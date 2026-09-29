"""Fixed loss and behavioral evaluation for the A320-family language model.

This module deliberately keeps validation and final-test evaluation separate.
The final test split is never loaded by training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from tokenizer import tokenizer_fingerprint

FIXED_EVALUATION_FORMAT = "fixed-evaluation-v1"
BEHAVIORAL_FORMAT = "a320-behavioral-benchmark-v1"
BEHAVIORAL_CATEGORIES = {
    "factual",
    "why",
    "how",
    "comparisons",
    "troubleshooting",
    "system_explanations",
    "mcdu_fmgs",
    "fcu_fma",
    "flight_phases",
    "performance",
    "simulator_scenarios",
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary_path = path + ".partial"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("format") != "document-split-v2":
        raise ValueError("Evaluation requires a document-split-v2 dataset manifest.")
    for split in ("train", "validation", "test"):
        if split not in manifest.get("splits", {}):
            raise ValueError(f"Dataset manifest is missing the {split!r} split.")
    return manifest


def _resolve_from_manifest(manifest_path: str, relative_path: str) -> str:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(manifest_path).parent / candidate)


def create_fixed_evaluation_set(
    manifest_path: str,
    split: str,
    output_path: str,
    context_length: int = 1024,
    max_windows: int = 256,
) -> Dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Fixed evaluation sets can only use validation or test.")
    if context_length < 1 or max_windows < 1:
        raise ValueError("context_length and max_windows must be positive.")
    manifest = load_manifest(manifest_path)
    split_info = manifest["splits"][split]
    token_path = _resolve_from_manifest(manifest_path, split_info["path"])
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Token split {token_path!r} does not exist.")
    actual_hash = sha256_file(token_path)
    if actual_hash != split_info["sha256"]:
        raise ValueError(f"{split} token file hash does not match the manifest.")
    tokens = np.memmap(token_path, dtype=np.dtype(manifest["preprocessing"]["dtype"]), mode="r")
    max_start = int(tokens.shape[0]) - context_length - 1
    if max_start < 0:
        raise ValueError(f"{split} split is too short for context length {context_length}.")
    count = min(max_windows, max_start + 1)
    if count == 1:
        offsets = [0]
    else:
        offsets = [
            int(round(index * max_start / (count - 1)))
            for index in range(count)
        ]
    metadata = {
        "format": FIXED_EVALUATION_FORMAT,
        "role": "final_test" if split == "test" else "checkpoint_validation",
        "split": split,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "token_path": split_info["path"],
        "token_sha256": actual_hash,
        "dtype": manifest["preprocessing"]["dtype"],
        "context_length": context_length,
        "offsets": offsets,
        "selection": "deterministic_evenly_spaced_offsets",
        "max_windows": max_windows,
        "dataset_policy": (
            "This set may be used for checkpoint selection."
            if split == "validation"
            else "This set is final-only and must not be used for training or selection."
        ),
    }
    atomic_json_write(output_path, metadata)
    return metadata


def load_fixed_evaluation_set(
    path: str,
    manifest_path: str,
    expected_split: str,
    context_length: int,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("format") != FIXED_EVALUATION_FORMAT:
        raise ValueError(f"Unsupported fixed evaluation set format in {path!r}.")
    if metadata.get("split") != expected_split:
        raise ValueError(
            f"Evaluation set {path!r} is for {metadata.get('split')!r}, "
            f"not {expected_split!r}."
        )
    expected_role = (
        "checkpoint_validation" if expected_split == "validation" else "final_test"
    )
    if metadata.get("role") != expected_role:
        raise ValueError(
            f"Evaluation set {path!r} has role {metadata.get('role')!r}; "
            f"expected {expected_role!r}."
        )
    fixed_context_length = metadata.get("context_length")
    if not isinstance(fixed_context_length, int) or context_length > fixed_context_length:
        raise ValueError(
            f"Evaluation set context length {fixed_context_length} cannot support "
            f"the requested context length {context_length}."
        )
    if metadata.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Evaluation set {path!r} belongs to a different manifest.")
    manifest = load_manifest(manifest_path)
    split_info = manifest["splits"][expected_split]
    token_path = _resolve_from_manifest(manifest_path, split_info["path"])
    if metadata.get("token_sha256") != split_info["sha256"]:
        raise ValueError(f"Evaluation set {path!r} has a stale split hash.")
    if sha256_file(token_path) != metadata["token_sha256"]:
        raise ValueError(f"Token file for evaluation set {path!r} has changed.")
    offsets = metadata.get("offsets")
    if not isinstance(offsets, list) or not offsets:
        raise ValueError(f"Evaluation set {path!r} has no fixed offsets.")
    token_count = os.path.getsize(token_path) // np.dtype(metadata["dtype"]).itemsize
    max_start = token_count - context_length - 1
    if any(not isinstance(offset, int) or offset < 0 or offset > max_start for offset in offsets):
        raise ValueError(f"Evaluation set {path!r} contains an invalid offset.")
    return {
        **metadata,
        "manifest_path": manifest_path,
        "token_path_resolved": token_path,
        "fixed_context_length": fixed_context_length,
        "context_length": context_length,
    }


def _autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "none":
        return nullcontext()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported AMP dtype {dtype_name!r}.")
    if device.type != "cuda":
        raise ValueError(f"AMP dtype {dtype_name!r} requires CUDA.")
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def evaluate_fixed_token_loss(
    model: torch.nn.Module,
    evaluation_set: Dict[str, Any],
    batch_size: int,
    device: torch.device,
    amp_dtype: str,
) -> Dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    tokens = np.memmap(
        evaluation_set["token_path_resolved"],
        dtype=np.dtype(evaluation_set["dtype"]),
        mode="r",
    )
    context_length = int(evaluation_set["context_length"])
    offsets = evaluation_set["offsets"]
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(offsets), batch_size):
        batch_offsets = offsets[start : start + batch_size]
        x = np.stack([tokens[offset : offset + context_length] for offset in batch_offsets])
        y = np.stack(
            [tokens[offset + 1 : offset + context_length + 1] for offset in batch_offsets]
        )
        input_ids = torch.from_numpy(x.astype(np.int64, copy=False)).to(device)
        targets = torch.from_numpy(y.astype(np.int64, copy=False)).to(device)
        with _autocast_context(device, amp_dtype):
            _, loss = model(input_ids, targets)
        token_count = int(input_ids.numel())
        total_nll += float(loss.item()) * token_count
        total_tokens += token_count
    if was_training:
        model.train()
    cross_entropy = total_nll / max(total_tokens, 1)
    perplexity = math.exp(cross_entropy) if cross_entropy < 700 else float("inf")
    return {
        "split": evaluation_set["split"],
        "role": evaluation_set["role"],
        "windows": len(offsets),
        "tokens": total_tokens,
        "cross_entropy": cross_entropy,
        "perplexity": perplexity,
    }


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _contains_alias(text: str, aliases: Iterable[str]) -> bool:
    return any(_normalized(alias) in text for alias in aliases)


def score_behavioral_answer(item: Dict[str, Any], answer: str) -> Dict[str, Any]:
    normalized_answer = _normalized(answer)
    facts = item["expected_key_facts"]
    fact_results = []
    for fact in facts:
        matched = _contains_alias(normalized_answer, fact["aliases"])
        fact_results.append({"fact": fact["fact"], "matched": matched})
    matched_count = sum(result["matched"] for result in fact_results)
    forbidden_hits = [
        claim for claim in item.get("forbidden_claims", [])
        if _normalized(claim) in normalized_answer
    ]
    topic_hits = [
        term for term in item.get("topic_terms", [])
        if _normalized(term) in normalized_answer
    ]
    requirement_hits = [
        requirement for requirement in item.get("question_requirements", [])
        if _normalized(requirement) in normalized_answer
    ]
    sentences = [part.strip() for part in re.split(r"[.!?]+", answer) if part.strip()]
    anchors = list(item.get("topic_terms", []))
    anchors.extend(alias for fact in facts for alias in fact["aliases"])
    anchors.extend(["check", "monitor", "approved", "procedure", "depends", "configuration"])
    unsupported_sentences = [
        sentence for sentence in sentences
        if len(sentence.split()) >= 6 and not _contains_alias(_normalized(sentence), anchors)
    ]
    contradiction_rate = len(forbidden_hits) / max(len(facts), 1)
    fact_coverage = matched_count / max(len(facts), 1)
    return {
        "relevance": {
            "score": min(len(topic_hits) / max(len(item.get("topic_terms", [])), 1), 1.0),
            "topic_terms_found": topic_hits,
            "interpretation": "automatic topic-overlap proxy",
        },
        "factual_correctness": {
            "score": max(0.0, fact_coverage - contradiction_rate),
            "key_fact_coverage": fact_coverage,
            "forbidden_claims_found": forbidden_hits,
            "interpretation": "key-fact coverage minus explicit forbidden-claim hits; review required for full factual judgment",
        },
        "completeness": {
            "score": fact_coverage,
            "matched_key_facts": matched_count,
            "total_key_facts": len(facts),
        },
        "hallucination": {
            "unsupported_sentence_rate": len(unsupported_sentences) / max(len(sentences), 1),
            "unsupported_sentences": unsupported_sentences,
            "interpretation": "heuristic risk flag, not a claim that every unmatched sentence is false",
        },
        "question_following": {
            "score": len(requirement_hits) / max(len(item.get("question_requirements", [])), 1),
            "requirements_found": requirement_hits,
            "interpretation": "automatic requirement-overlap proxy",
        },
    }


def validate_behavioral_benchmark(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        benchmark = json.load(file)
    if benchmark.get("format") != BEHAVIORAL_FORMAT:
        raise ValueError(f"Unsupported behavioral benchmark format in {path!r}.")
    questions = benchmark.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("Behavioral benchmark must contain questions.")
    categories = {item.get("category") for item in questions}
    missing_categories = sorted(BEHAVIORAL_CATEGORIES - categories)
    if missing_categories:
        raise ValueError(f"Behavioral benchmark is missing categories: {missing_categories}.")
    for item in questions:
        if not item.get("id") or not item.get("question"):
            raise ValueError("Every behavioral question needs an id and question.")
        if not item.get("expected_key_facts"):
            raise ValueError(f"Question {item['id']!r} has no expected key facts.")
        for fact in item["expected_key_facts"]:
            if not fact.get("fact") or not fact.get("aliases"):
                raise ValueError(f"Question {item['id']!r} has an incomplete key fact.")
    return benchmark


def summarize_behavioral(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dimensions = {
        "relevance": [result["automatic"]["relevance"]["score"] for result in results],
        "factual_correctness": [
            result["automatic"]["factual_correctness"]["score"] for result in results
        ],
        "completeness": [result["automatic"]["completeness"]["score"] for result in results],
        "hallucination": [
            result["automatic"]["hallucination"]["unsupported_sentence_rate"]
            for result in results
        ],
        "question_following": [
            result["automatic"]["question_following"]["score"] for result in results
        ],
    }
    return {
        dimension: {
            "mean": sum(values) / max(len(values), 1),
            "count": len(values),
            "interpretation": (
                "lower is better risk" if dimension == "hallucination"
                else "higher is better proxy"
            ),
        }
        for dimension, values in dimensions.items()
    }


def evaluate_behavioral(
    model: torch.nn.Module,
    tokenizer: Any,
    benchmark: Dict[str, Any],
    generate_tokens_function: Any,
    device: torch.device,
    max_tokens: int,
) -> Dict[str, Any]:
    results = []
    for item in benchmark["questions"]:
        prompt = f"Question: {item['question']}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        output_ids = generate_tokens_function(
            model,
            prompt_ids,
            max_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(output_ids[len(prompt_ids):]).strip()
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
            "expected_key_facts": item["expected_key_facts"],
            "automatic": score_behavioral_answer(item, answer),
            "manual_review": {
                "required": True,
                "dimensions": [
                    "relevance",
                    "factual_correctness",
                    "completeness",
                    "hallucination",
                    "question_following",
                ],
                "note": "Automatic results are transparent proxies; record expert judgments separately rather than collapsing them into one score.",
            },
        })
    return {
        "benchmark_format": BEHAVIORAL_FORMAT,
        "question_count": len(results),
        "results": results,
        "summary_by_dimension": summarize_behavioral(results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("create-fixed", "validation", "final-test", "behavioral", "all"),
        default="all",
    )
    parser.add_argument("--manifest", default="dataset_manifest.json")
    parser.add_argument("--validation-set", default="evaluation/validation_eval.json")
    parser.add_argument("--test-set", default="evaluation/test_eval.json")
    parser.add_argument(
        "--benchmark",
        default="evaluation/a321neo_behavioral_benchmark.json",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="evaluation/evaluation_report.json")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    if args.mode == "create-fixed":
        validation = create_fixed_evaluation_set(
            args.manifest, "validation", args.validation_set,
            args.context_length, args.max_windows,
        )
        test = create_fixed_evaluation_set(
            args.manifest, "test", args.test_set,
            args.context_length, args.max_windows,
        )
        print(f"Created validation set with {len(validation['offsets'])} windows.")
        print(f"Created final test set with {len(test['offsets'])} windows.")
        return

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for evaluation modes.")
    from generate import load_model, load_tokenizer_for_model, generate_tokens

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer_for_model(model)
    report: Dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "device": str(device),
        "model_config": model.config.to_dict(),
    }
    if args.mode in {"validation", "all"}:
        validation_set = load_fixed_evaluation_set(
            args.validation_set, args.manifest, "validation", args.context_length
        )
        report["validation"] = evaluate_fixed_token_loss(
            model, validation_set, args.batch_size, device, model.config.amp_dtype
        )
    if args.mode in {"final-test", "all"}:
        test_set = load_fixed_evaluation_set(
            args.test_set, args.manifest, "test", args.context_length
        )
        report["final_test"] = evaluate_fixed_token_loss(
            model, test_set, args.batch_size, device, model.config.amp_dtype
        )
    if args.mode in {"behavioral", "all"}:
        benchmark = validate_behavioral_benchmark(args.benchmark)
        report["behavioral"] = evaluate_behavioral(
            model, tokenizer, benchmark, generate_tokens, device, args.max_tokens
        )
    atomic_json_write(args.output, report)
    for name in ("validation", "final_test"):
        if name in report:
            metrics = report[name]
            print(
                f"{name}: cross_entropy={metrics['cross_entropy']:.6f} "
                f"perplexity={metrics['perplexity']:.6f} "
                f"windows={metrics['windows']}"
            )
    if "behavioral" in report:
        print("behavioral summary (separate dimensions; no composite score):")
        for dimension, values in report["behavioral"]["summary_by_dimension"].items():
            print(f"  {dimension}: {values['mean']:.4f} ({values['interpretation']})")
    print(f"Evaluation report written to {args.output}")


if __name__ == "__main__":
    main()