"""Audit and conservatively deduplicate the active training corpus.

The source corpus is never modified.  The command writes a report first and
then writes a cleaned corpus plus a family manifest.  Cleaning removes only
records that are exact, near-exact, or template-equivalent duplicates within
the same technical topic.  It does not rewrite retained records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


REPORT_FORMAT = "corpus-quality-report-v1"
FAMILY_MANIFEST_FORMAT = "corpus-family-manifest-v1"
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85
NEAR_DUPLICATE_SHINGLE_SIZE = 5

# These are the subject labels used by the generated examples.  Aliases are
# deliberately explicit: broad noun extraction would accidentally merge
# different aircraft concepts.
TOPIC_ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("turbofan engines", ("turbofan engines",)),
    ("navigation database", ("navigation database",)),
    ("weight and balance", ("weight and balance",)),
    ("airspeed and mach", ("airspeed and mach",)),
    ("energy management", ("energy management",)),
    ("flight director", ("flight director",)),
    ("ceo/neo distinction", ("ceo/neo distinction", "ceo and neo")),
    ("a320 family", ("a320 family",)),
    ("a321neo", ("a321neo",)),
    ("electrical system", ("electrical system",)),
    ("hydraulic system", ("hydraulic system",)),
    ("pneumatic system", ("pneumatic system",)),
    ("fuel system", ("fuel system",)),
    ("autopilot", ("autopilot",)),
    ("autothrust", ("autothrust",)),
    ("anti-ice", ("anti-ice",)),
    ("fmgs", ("fmgs",)),
    ("mcdu", ("mcdu",)),
    ("fcu", ("fcu",)),
    ("fma", ("fma",)),
    ("ils", ("ils",)),
    ("sidestick", ("sidestick",)),
    ("fly-by-wire", ("fly-by-wire",)),
    ("sids and stars", ("sids and stars",)),
    ("rnav and rnp", ("rnav and rnp", "rnav/rnp")),
    ("dispatch", ("dispatch",)),
    ("lift and drag", ("lift and drag",)),
    ("pressurization", ("pressurization",)),
)
SORTED_ALIASES = sorted(
    ((alias, topic) for topic, aliases in TOPIC_ALIASES for alias in aliases),
    key=lambda item: len(item[0]),
    reverse=True,
)

BOILERPLATE_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    (
        "generic training context",
        re.compile(
            r"in this training context, during .*? where the learner is "
            r"focusing on .*? while checking .*?\.",
            re.IGNORECASE,
        ),
    ),
    (
        "simulator caveat",
        re.compile(
            r"a simulator can help explore the relationship, but its default "
            r"behavior is not automatically an approved real-aircraft procedure\.",
            re.IGNORECASE,
        ),
    ),
    (
        "configuration caveat",
        re.compile(
            r"exact behavior depends on aircraft standard, software, engine option, "
            r"database, operator procedure, and the current flight condition\.",
            re.IGNORECASE,
        ),
    ),
    (
        "learning-angle label",
        re.compile(r"the learning angle here is [^.]+\.", re.IGNORECASE),
    ),
    (
        "training-variation label",
        re.compile(r"training variation \d+:", re.IGNORECASE),
    ),
    (
        "scenario scaffolding",
        re.compile(
            r"scenario: imagine a study example .*? the trainee observes "
            r"unexpected behavior involving .*?\.",
            re.IGNORECASE,
        ),
    ),
)

STOPWORDS = set(
    """
    a an and are as at be because by can could does for from has have how i if
    in is it its keep more not of on or should so that the their them then
    this to was what when where which while why with would you your
    """.split()
)
GENERIC_WORDS = set(
    """
    user assistant scenario imagine study training variation context learner
    focusing checking question answer angle perspective explanation beginner
    useful understand think reason aircraft state otherwise unchanged expected
    observed result concept learned memorizing general principle exact
    configuration software standard procedure modeled model system topic
    """.split()
)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]{2,}")
PREAMBLE_PREFIXES = (
    "this is an educational language-model corpus",
    "it is not an fcom",
    "the source corpus was cleaned",
    "public source basis:",
)


def read_records(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    return [record.strip() for record in text.split("\n\n") if record.strip()]


def canonical(text: str) -> str:
    return " ".join(text.casefold().split())


def topic_for(text: str) -> str:
    lowered = text.casefold()
    for alias, topic in SORTED_ALIASES:
        if re.search(r"(?<![\w/-])" + re.escape(alias) + r"(?![\w/-])", lowered):
            return topic
    return "<unknown>"


def _replace_topics(text: str) -> str:
    for alias, _topic in SORTED_ALIASES:
        text = re.sub(
            r"(?<![\w/-])" + re.escape(alias) + r"(?![\w/-])",
            "<topic>",
            text,
        )
    return text


def template_signature(text: str) -> str:
    """Normalize generated slots while retaining wording and answer structure."""
    value = canonical(text)
    value = re.sub(
        r"training variation\s+\d+:\s*[^.]+\.",
        "training variation <variation>.",
        value,
    )
    value = re.sub(
        r"in this training context, during [^,]+, where the learner is "
        r"focusing on [^,]+ while checking [^.]+\.",
        "in this training context, <context>.",
        value,
    )
    value = re.sub(r"from (?:a|the) [^?.,]+ perspective", "from <perspective>", value)
    value = re.sub(r"from the angle of [^?.,]+", "from the angle of <angle>", value)
    value = re.sub(r"during [a-z][^?.,]+? study", "during <study> study", value)
    value = re.sub(
        r"after a small configuration change — exercise \d+",
        "after <change>",
        value,
    )
    value = re.sub(r"exercise \d+", "exercise <num>", value)
    value = _replace_topics(value)
    value = re.sub(r"\b\d+\b", "<num>", value)
    return value


def family_id(signature: str) -> str:
    return "family-" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def family_records(records: Sequence[str]) -> Tuple[List[str], Dict[str, List[int]]]:
    signatures = [template_signature(record) for record in records]
    ids = [family_id(signature) for signature in signatures]
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, identifier in enumerate(ids):
        groups[identifier].append(index)
    return ids, dict(groups)


def assistant_spans(text: str) -> Iterable[str]:
    matches = list(
        re.finditer(
            r"Assistant:\s*(.*?)(?=\s+(?:User|Scenario):|$)",
            text,
            flags=re.IGNORECASE,
        )
    )
    for match in matches:
        answer = canonical(match.group(1))
        if answer:
            yield answer


def variation_signature(text: str) -> str | None:
    match = re.search(r"training variation\s+\d+:\s*(.+)", text, re.IGNORECASE)
    if not match:
        return None
    return template_signature(match.group(1))


def informative_words(text: str) -> set[str]:
    return {
        word.casefold()
        for word in WORD_RE.findall(text)
        if word.casefold() not in STOPWORDS and word.casefold() not in GENERIC_WORDS
    }


def specificity_score(text: str) -> Tuple[int, int, int]:
    """Prefer a representative with more non-scaffolding lexical content."""
    words = informative_words(text)
    return (len(words), len(words), len(text))


def shingles(text: str, size: int = NEAR_DUPLICATE_SHINGLE_SIZE) -> set[Tuple[str, ...]]:
    tokens = canonical(text).split()
    if len(tokens) < size:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(left: set[Tuple[str, ...]], right: set[Tuple[str, ...]]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def near_duplicate_stats(
    records: Sequence[str],
    candidate_groups: Dict[Tuple[str, str], List[int]],
) -> Dict[str, object]:
    pair_count = 0
    records_in_pairs: set[int] = set()
    for indices in candidate_groups.values():
        if len(indices) < 2:
            continue
        indexed_shingles = [(index, shingles(records[index])) for index in indices]
        for left_position, (left_index, left_shingles) in enumerate(indexed_shingles):
            for right_index, right_shingles in indexed_shingles[left_position + 1 :]:
                if jaccard(left_shingles, right_shingles) >= NEAR_DUPLICATE_JACCARD_THRESHOLD:
                    pair_count += 1
                    records_in_pairs.update((left_index, right_index))
    return {
        "threshold": NEAR_DUPLICATE_JACCARD_THRESHOLD,
        "shingle_size": NEAR_DUPLICATE_SHINGLE_SIZE,
        "pair_count": pair_count,
        "record_count": len(records_in_pairs),
    }


def sentence_counts(records: Sequence[str]) -> Counter[str]:
    sentences: Counter[str] = Counter()
    for record in records:
        for sentence in re.split(r"(?<=[.!?])\s+", record):
            sentence = canonical(sentence)
            if sentence:
                sentences[sentence] += 1
    return sentences


def boilerplate_counts(records: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label, pattern in BOILERPLATE_PATTERNS:
        counts[label] = sum(len(pattern.findall(record)) for record in records)
    return counts


def low_information_indices(records: Sequence[str], preamble_count: int) -> List[int]:
    """Flag only records with almost no lexical content after generic words."""
    flagged: List[int] = []
    for index, record in enumerate(records):
        if index < preamble_count:
            continue
        words = informative_words(record)
        if len(words) < 12:
            flagged.append(index)
    return flagged


def duplicate_groups(records: Sequence[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[canonical(record)].append(index)
    return dict(groups)


def choose_clean_indices(
    records: Sequence[str],
    family_ids: Sequence[str],
    preamble_count: int,
) -> Tuple[List[int], Dict[str, int]]:
    """Keep one representative per template family and technical topic."""
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if index < preamble_count:
            continue
        groups[(family_ids[index], topic_for(record))].append(index)

    selected = set(range(min(preamble_count, len(records))))
    for indices in groups.values():
        representative = max(
            indices,
            key=lambda index: (specificity_score(records[index]), -index),
        )
        selected.add(representative)
    return sorted(selected), {
        "template_equivalent_groups": sum(len(v) > 1 for v in groups.values()),
        "template_equivalent_records_removed": sum(
            len(v) - 1 for v in groups.values() if len(v) > 1
        ),
    }


def count_topics(records: Sequence[str]) -> Dict[str, int]:
    return dict(sorted(Counter(topic_for(record) for record in records).items()))


def count_families(
    family_ids: Sequence[str], indices: Iterable[int] | None = None
) -> Dict[str, int]:
    selected = indices if indices is not None else range(len(family_ids))
    return dict(sorted(Counter(family_ids[index] for index in selected).items()))


def build_report(
    source: Path,
    records: Sequence[str],
    cleaned_records: Sequence[str],
    family_ids: Sequence[str],
    family_groups: Dict[str, List[int]],
    selected_indices: Sequence[int],
    removal_counts: Dict[str, int],
    preamble_count: int,
) -> Dict[str, object]:
    exact = duplicate_groups(records)
    exact_groups = [indices for indices in exact.values() if len(indices) > 1]
    candidate_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, family in enumerate(family_ids):
        candidate_groups[(family, topic_for(records[index]))].append(index)
    near = near_duplicate_stats(records, candidate_groups)
    answers = Counter(
        answer
        for record in records
        for answer in assistant_spans(record)
    )
    variations = Counter(
        signature
        for record in records
        if (signature := variation_signature(record)) is not None
    )
    sentences = sentence_counts(records)
    repeated_sentences = [
        {"count": count, "sentence": sentence}
        for sentence, count in sentences.most_common(30)
        if count >= 20
    ]
    low_info = low_information_indices(records, preamble_count)

    before_bytes = sum(len(record.encode("utf-8")) for record in records)
    after_bytes = sum(len(record.encode("utf-8")) for record in cleaned_records)
    report: Dict[str, object] = {
        "format": REPORT_FORMAT,
        "source": str(source),
        "policy": {
            "original_preserved": True,
            "retained_records_are_unmodified": True,
            "technical_topic_is_part_of_deduplication_key": True,
            "cleaning_rule": (
                "Keep one deterministic representative for each normalized "
                "template family and technical topic; preserve preamble records."
            ),
            "near_duplicate_rule": (
                f"{NEAR_DUPLICATE_SHINGLE_SIZE}-word shingle Jaccard >= "
                f"{NEAR_DUPLICATE_JACCARD_THRESHOLD:.2f}"
            ),
        },
        "before": {
            "record_count": len(records),
            "utf8_bytes": before_bytes,
            "topic_counts": count_topics(records),
            "template_family_count": len(family_groups),
            "template_family_counts": dict(
                sorted(
                    ((family, len(indices)) for family, indices in family_groups.items()),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        },
        "after": {
            "record_count": len(cleaned_records),
            "utf8_bytes": after_bytes,
            "topic_counts": count_topics(cleaned_records),
            "template_family_count": len(set(family_ids[index] for index in selected_indices)),
            "template_family_counts": count_families(family_ids, selected_indices),
        },
        "duplicates": {
            "exact_duplicate_groups": len(exact_groups),
            "exact_duplicate_records_beyond_first": sum(
                len(indices) - 1 for indices in exact_groups
            ),
            "near_duplicate": near,
            "template_equivalent_groups": removal_counts["template_equivalent_groups"],
            "template_equivalent_records_removed": removal_counts[
                "template_equivalent_records_removed"
            ],
            "repeated_answer_groups": sum(count > 1 for count in answers.values()),
            "records_with_repeated_answers": sum(
                count for count in answers.values() if count > 1
            ),
            "repeated_training_variation_structures": sum(
                count > 1 for count in variations.values()
            ),
            "records_with_repeated_training_variation_structures": sum(
                count for count in variations.values() if count > 1
            ),
        },
        "boilerplate": {
            "pattern_counts": boilerplate_counts(records),
            "repeated_sentences_at_least_20": repeated_sentences,
            "low_information_candidate_count": len(low_info),
            "low_information_candidate_indices": low_info[:100],
        },
        "removals": {
            "records_removed": len(records) - len(cleaned_records),
            "utf8_bytes_removed": before_bytes - after_bytes,
            "removed_by_reason": {
                "template_equivalent_duplicate": removal_counts[
                    "template_equivalent_records_removed"
                ],
                "other": 0,
            },
        },
    }
    return report


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: Dict[str, object]) -> None:
    before = report["before"]
    after = report["after"]
    duplicates = report["duplicates"]
    boilerplate = report["boilerplate"]
    removals = report["removals"]
    lines = [
        "# Corpus cleaning report",
        "",
        f"- Source: `{report['source']}`",
        "- The original source was preserved.",
        "- Retained records were copied without technical rewriting.",
        "",
        "## Cleaning policy",
        "",
        report["policy"]["cleaning_rule"],
        "",
        "The technical topic is included in the deduplication key. Therefore,",
        "different questions about the same aircraft system are not removed merely",
        "because they share a subject. Only records with the same normalized",
        "template family and subject compete for one representative.",
        "",
        "## Statistics before and after",
        "",
        "| Measure | Before | After |",
        "|---|---:|---:|",
        f"| Records | {before['record_count']:,} | {after['record_count']:,} |",
        f"| UTF-8 bytes | {before['utf8_bytes']:,} | {after['utf8_bytes']:,} |",
        f"| Template families | {before['template_family_count']:,} | {after['template_family_count']:,} |",
        "",
        "## Duplicate and repetition counts",
        "",
        f"- Exact duplicate groups: **{duplicates['exact_duplicate_groups']:,}**",
        f"- Exact duplicate records beyond the first: **{duplicates['exact_duplicate_records_beyond_first']:,}**",
        f"- Strict near-duplicate pairs: **{duplicates['near_duplicate']['pair_count']:,}** "
        f"({duplicates['near_duplicate']['record_count']:,} records; "
        f"{report['policy']['near_duplicate_rule']})",
        f"- Template-equivalent groups: **{duplicates['template_equivalent_groups']:,}**",
        f"- Template-equivalent records removed: **{duplicates['template_equivalent_records_removed']:,}**",
        f"- Repeated answer groups: **{duplicates['repeated_answer_groups']:,}**",
        f"- Records participating in repeated answers: **{duplicates['records_with_repeated_answers']:,}**",
        f"- Repeated `Training variation` structures: **{duplicates['repeated_training_variation_structures']:,}**",
        f"- Records participating in repeated variation structures: "
        f"**{duplicates['records_with_repeated_training_variation_structures']:,}**",
        "",
        "## Boilerplate and low-information analysis",
        "",
        "| Pattern | Occurrences |",
        "|---|---:|",
    ]
    for label, count in boilerplate["pattern_counts"].items():
        lines.append(f"| {label} | {count:,} |")
    lines.extend(
        [
            "",
            f"Low-information candidates (flagged, not independently deleted): "
            f"**{boilerplate['low_information_candidate_count']:,}**.",
            "They are only removed when they are also template-equivalent duplicates;",
            "unique records remain available for review.",
            "",
            "## Removal summary",
            "",
            f"- Records removed: **{removals['records_removed']:,}**",
            f"- UTF-8 bytes removed: **{removals['utf8_bytes_removed']:,}**",
            "- Removal reason: template-equivalent duplicate within the same technical topic.",
            "",
            "## Template-family counts",
            "",
            "The complete before/after family counts are in the JSON report.",
            "The largest families before cleaning are:",
            "",
        ]
    )
    largest_families = sorted(
        before["template_family_counts"].items(),
        key=lambda item: (-item[1], item[0]),
    )
    for family, count in largest_families[:25]:
        lines.append(f"- `{family}`: {count:,}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `training_data_cleaned.txt` — cleaned corpus; blank-line-delimited records.",
            "- `training_data_cleaned.families.json` — record-to-family assignments.",
            "- `corpus_cleaning_report.json` — machine-readable full report.",
            "- `corpus_cleaning_report.md` — this human-readable report.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_family_manifest(
    path: Path,
    source: Path,
    records: Sequence[str],
    family_ids: Sequence[str],
) -> None:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, identifier in enumerate(family_ids):
        groups[identifier].append(index)
    payload = {
        "format": FAMILY_MANIFEST_FORMAT,
        "source_path": str(source),
        "document_count": len(records),
        "family_count": len(groups),
        "document_families": list(family_ids),
        "families": [
            {
                "family_id": identifier,
                "document_indices": indices,
                "document_count": len(indices),
            }
            for identifier, indices in sorted(groups.items())
        ],
    }
    write_json(path, payload)


def run(
    source: Path,
    output: Path,
    report_path: Path,
    report_json_path: Path,
    family_manifest_path: Path,
) -> Dict[str, object]:
    records = read_records(source)
    if not records:
        raise ValueError(f"Corpus {source} contains no records.")
    first_generated = 0
    while (
        first_generated < len(records)
        and canonical(records[first_generated]).startswith(PREAMBLE_PREFIXES)
    ):
        first_generated += 1
    family_ids, family_groups = family_records(records)
    selected_indices, removal_counts = choose_clean_indices(
        records, family_ids, first_generated
    )
    cleaned_records = [records[index] for index in selected_indices]

    # The report is written before the cleaned corpus and manifest.  The source
    # remains untouched in all cases.
    report = build_report(
        source,
        records,
        cleaned_records,
        family_ids,
        family_groups,
        selected_indices,
        removal_counts,
        first_generated,
    )
    write_json(report_json_path, report)
    write_markdown(report_path, report)

    output.write_text("\n\n".join(cleaned_records) + "\n", encoding="utf-8")
    cleaned_family_ids = [family_ids[index] for index in selected_indices]
    write_family_manifest(
        family_manifest_path, output, cleaned_records, cleaned_family_ids
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="training_data.txt")
    parser.add_argument("--output", default="training_data_cleaned.txt")
    parser.add_argument("--report", default="corpus_cleaning_report.md")
    parser.add_argument("--report-json", default="corpus_cleaning_report.json")
    parser.add_argument(
        "--family-manifest",
        default="training_data_cleaned.families.json",
    )
    args = parser.parse_args()
    report = run(
        Path(args.input),
        Path(args.output),
        Path(args.report),
        Path(args.report_json),
        Path(args.family_manifest),
    )
    print(
        f"Cleaned {report['before']['record_count']:,} records to "
        f"{report['after']['record_count']:,}; "
        f"removed {report['removals']['records_removed']:,} "
        "template-equivalent duplicates."
    )


if __name__ == "__main__":
    main()