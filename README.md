# Local Subword GPT

This is a complete, local, decoder-only Transformer language model written in
Python and PyTorch. It starts with randomly initialized weights, learns a
compact byte-subword tokenizer from your text, trains locally, and does not
call OpenAI, Anthropic, Gemini, Hugging Face pretrained models, or any
inference API.

The default model is intentionally small:

| Setting | Default |
| --- | ---: |
| Vocabulary | 4,096 learned byte-subword tokens |
| Context length | 1,024 tokens |
| Embedding dimension | 384 |
| Transformer layers | 8 |
| Attention heads | 6 |
| Feed-forward dimension | 1,536 SwiGLU hidden units |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Dropout | 0.05 |

## Install and run

Use Python 3.10+ in a virtual environment. Install a CUDA-enabled PyTorch
build if you have an NVIDIA GPU and want GPU training; the code automatically
selects CUDA when that build and a visible GPU are available.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Prepare the included sample text. This first assigns complete blank-line-
delimited documents to three disjoint partitions, trains `tokenizer.json`
using TRAIN documents only, and then creates three separate compact `uint16`
token files:

```bash
python prepare_dataset.py
```

Verify the prepared artifacts before training. The verifier fails with all
detected inconsistencies, including stale hashes, dtype/size mismatches,
out-of-vocabulary IDs, source/split overlap, train-only tokenizer provenance,
and EOS placement:

```bash
python verify_dataset.py
```

Put your own UTF-8 text in `training_data.txt`, or point the script at another
file or directory. A file is interpreted as blank-line-delimited documents;
each file in a directory is processed the same way. The preprocessing pass
keeps only a bounded tokenizer sample and one document in memory at a time.
Document assignment is deterministic from document bytes and `--split-seed`.
EOS is written once per document, never once per I/O chunk.

The generated artifacts are:

```text
train_tokens.bin
validation_tokens.bin
test_tokens.bin
tokenizer.json
dataset_manifest.json
```

The manifest records split counts, token counts, source and output hashes,
tokenizer provenance, preprocessing settings, and the isolation policy.
`train.py` rejects the old single-file token-offset metadata instead of
silently using it.

Train from scratch. The default schedule includes gradient accumulation,
warmup, cosine decay, mixed precision on CUDA, validation, and atomic
checkpoints. The default budget is calculated from the current TRAIN token
count, not a fixed step count:

```bash
python train.py
```

For the current included corpus and defaults, the schedule is approximately:

```text
unique TRAIN tokens:          4,276,031
tokens per optimizer update:     32,768
updates per epoch:                    131
initial budget:                        5 token-equivalent epochs
optimizer updates:                    655
total tokens processed:        21,463,040
effective epochs:                   5.019
```

`train.py` prints the same values from the manifest at startup. It tracks
token-equivalent epoch progress because batches are sampled randomly from the
TRAIN token file. Use `--target-epochs` to choose a different data-sized
budget. `--training-steps` remains available only as an explicit override and
will warn when it repeatedly cycles through TRAIN excessively. Progress lines
log `step`, `epoch`, `tokens processed`, `train loss`, `validation loss`,
`train perplexity`, `validation perplexity`, and `learning rate`. Validation
loss does not automatically extend the run.

For a quick smoke test:

```bash
python train.py --training-steps 5 --eval-interval 2 --eval-steps 1 \
  --checkpoint-interval 5 --batch-size 2 \
  --gradient-accumulation-steps 1 --context-length 256
```

Resume from the newest periodic checkpoint:

```bash
python train.py --resume
```

Create the immutable token-level evaluation windows for the current dataset:

```bash
python evaluation.py --mode create-fixed
```

Training uses only `evaluation/validation_eval.json` for checkpoint selection.
The training process never loads the TEST token file or final-test evaluation
windows. The final test windows in `evaluation/test_eval.json` are reserved
for evaluation after training, hyperparameters, prompts, and
instruction-tuning decisions are complete.

Evaluate a checkpoint with cross-entropy and perplexity reported separately:

```bash
python evaluation.py --mode validation --checkpoint checkpoints/best_model.pt
python evaluation.py --mode final-test --checkpoint checkpoints/best_model.pt
```

Run the manually curated A321neo/A320-family behavioral benchmark:

```bash
python evaluation.py --mode behavioral --checkpoint checkpoints/best_model.pt
```

The behavioral report keeps relevance, factual correctness, completeness,
hallucination risk, and question-following as separate dimensions. Automatic
checks are transparent proxies and each answer is marked for expert review;
there is no composite score. The final TEST set and behavioral benchmark are
not tuning data.

Start at step zero with new random weights. Existing checkpoints are not
deleted; a periodic checkpoint for the same step may be replaced:

```bash
python train.py --reset
```

Generate a continuation:

```bash
python generate.py --prompt "The aircraft" --max_tokens 300 \
  --temperature 0.8 --top-k 50 --top-p 0.92
```

Omit `--prompt` to type one interactively. `--temperature 0` uses greedy
decoding. Smaller temperatures are more conservative; larger values are more
random. `top-k` restricts sampling to the k most likely next tokens.

Start the short-history chat loop:

```bash
python chat.py
```

Or run one message:

```bash
python chat.py --message "Tell me about airplanes" --max_tokens 100
```

Benchmark the actual machine:

```bash
python benchmark.py
python benchmark.py --steps 10 --batch-size 2 --inference-only
python benchmark.py --checkpoint checkpoints/best_model.pt
```

Inspect the architecture and calculated parameter count without training:

```bash
python inspect_model.py
python inspect_model.py --checkpoint checkpoints/best_model.pt
```

Change the most useful training settings without editing code:

```bash
python train.py --batch-size 16 --learning-rate 0.0003 \
  --target-epochs 5 --checkpoint-interval 1000
```

All defaults live in `config.py`. Edit that file to change model size,
context length, dropout, layers, heads, embedding dimension, dataset manifest,
and training defaults in one place. Command-line values are convenient
temporary overrides for the training settings.

## Project layout

* `config.py` — one central configuration dataclass.
* `tokenizer.py` — dependency-free learned byte-subword tokenizer with byte fallback.
* `prepare_dataset.py` — deterministic document splitting, tokenizer training, and token preprocessing.
* `verify_dataset.py` — strict source, tokenizer, split, token-file, hash, and EOS verification.
* `dataset.py` — NumPy `memmap` batches for one pre-split token file.
* `model.py` — RoPE attention, RMSNorm, SwiGLU, and weight tying.
* `train.py` — AdamW, warmup/cosine decay, accumulation, AMP,
  validation-only checkpoint selection, and checkpoints with no TEST loader
  state.
* `instruction_data.jsonl` — curated conversational instruction examples.
* `prepare_instruction_data.py` — role-token vocabulary extension and
  deterministic instruction train/validation splitting.
* `finetune.py` — assistant-only masked instruction fine-tuning and best
  instruction checkpoint selection.
* `generate.py` — temperature/top-k/top-p/repetition-aware generation.
* `chat.py` — a bounded-history local chat wrapper.
* `benchmark.py` — actual forward/backward or inference throughput.
* `training_data.txt` — the active sample corpus.
* `train_tokens.bin`, `validation_tokens.bin`, `test_tokens.bin` — separate generated partitions.
* `dataset_manifest.json` — split, tokenizer, preprocessing, checksum, and
  isolation metadata.
* `evaluation/a321neo_behavioral_benchmark.json` — manually curated
  A321neo/A320-family questions with expected key facts.
* `checkpoints/` — periodic checkpoints and `best_model.pt`.

## Making it genuinely capable

Architecture improvements help, but data and training budget dominate quality.
For a useful assistant rather than a text continuation demo:

1. Use a large, clean, deduplicated corpus.
2. Keep train and validation documents separate; do not randomly split copies
   of the same document.
3. Pretrain on raw text, then fine-tune on high-quality conversations.
4. Format conversations with explicit system, user, assistant, and end-of-turn
   markers, and mask the loss so instruction tuning focuses on assistant text.
5. Evaluate against a fixed VALIDATION prompt set after each training change.
6. Keep the final TEST set sealed until the model, hyperparameters, prompts,
   and instruction-tuning decisions are frozen.

## Instruction fine-tuning

The repository includes a curated conversational dataset in
`instruction_data.jsonl`. It covers factual, explanatory, comparison,
scenario, troubleshooting, MCDU/FMGS, FCU/FMA, flight-planning, and simulator
questions. Prepare it and run the separate fine-tuning stage with:

```bash
python prepare_instruction_data.py
python finetune.py --checkpoint checkpoints/best_model.pt
```

Preparation creates real control-token IDs for `<|user|>`, `<|assistant|>`,
and `<|end|>` instead of encoding those markers as ordinary text. Fine-tuning
preserves the pretrained vocabulary prefix, uses a lower default learning rate
(`5e-5` versus the pretraining default `3e-4`), and masks user prompt and
role-prefix labels with `-100`. Assistant response tokens and `<|end|>` are
supervised. The best instruction-validation checkpoint is saved to:

```text
checkpoints/instruction_best_model.pt
```

The final TEST split and final evaluation benchmark are not used for
instruction data, validation, or checkpoint selection.

The current repository includes a small sample corpus for reproducibility and
smoke tests. It cannot produce a broadly knowledgeable assistant without a
larger corpus and a longer run.

## How the model works

### 1. Tokenization

`tokenizer.encode("é")` first encodes the Unicode string as UTF-8, producing
the bytes `0xC3 0xA9`, or token IDs `[195, 169]`. ASCII characters usually
take one token; many Unicode characters take multiple tokens. `decode` reverses
this operation and replaces invalid incomplete UTF-8 sequences gracefully.

### 2. Token and positional embeddings

The token embedding table has shape `256 x 192`. Each byte ID selects one
learned 192-dimensional vector. A second table has one learned vector for
each position from 0 through 255. The model adds token and position vectors,
so the same byte can mean something different at different positions.

### 3. Query, Key, and Value

For every position, a learned linear layer produces three vectors:

* **Query (Q):** what this position is looking for.
* **Key (K):** what information this position offers for matching.
* **Value (V):** the information passed along if it is relevant.

For one attention head:

```text
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

`Q K^T` gives a compatibility score between positions. Dividing by
`sqrt(d_k)` prevents dot products from becoming too large. Softmax turns the
scores into weights that sum to one, and the weighted values are mixed into
the current representation.

### 4. Causal multi-head self-attention

Self-attention lets every position use information from the same sequence.
This implementation splits the embedding into six heads, computes attention
independently in each head, concatenates the results, and applies an output
projection. Different heads can learn different relationships.

The lower-triangular causal mask changes every score for a future position to
negative infinity before softmax. Therefore position 5 can attend to
positions 0–5, but never positions 6 and onward. This is essential: during
training, the target byte at a future position is present in the batch, but
the model must not be allowed to read it while predicting the current byte.

### 5. Residual connections and layer normalization

Each Transformer block uses:

```text
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

The residual additions provide short paths for information and gradients.
Layer normalization keeps activations in a useful range. This project uses
pre-normalization, a common stable arrangement for Transformers.

### 6. Feed-forward network

After attention, every position independently passes through:

```text
Linear(192 -> 768) -> GELU -> Linear(768 -> 192)
```

The attention operation mixes information between positions; the feed-forward
network transforms each position's representation more deeply.

### 7. Transformer stack and logits

Six blocks repeatedly apply attention and the feed-forward transformation.
The final layer normalization is followed by a linear language-model head
that produces 256 logits at every position. A logit is an unnormalized score
for one possible next byte. The head shares its weights with the token
embedding table, a parameter-saving technique often used in language models.

### 8. Loss and learning

Softmax converts logits into probabilities. For each position, cross-entropy
loss is the negative log probability assigned to the actual next byte. If the
correct byte has probability `p`, its contribution is `-log(p)`.

For a sequence `t[0], ..., t[n]`, the dataset creates:

```text
X = t[i : i + 256]
Y = t[i + 1 : i + 257]
```

The model predicts all 256 targets in parallel during training. PyTorch
backpropagation computes gradients of the average cross-entropy with respect
to every parameter. AdamW uses those gradients and moving estimates of their
first and second moments to update the weights. Weight decay is applied
separately as a regularizer. Repeating this process makes likely training
patterns receive higher probability.

### 9. Autoregressive generation

Generation starts with the prompt's byte IDs. The model predicts a
distribution for the next byte, samples one byte using temperature and
optional top-k filtering, appends it, and repeats. If the sequence grows
past 256 tokens, only the most recent 256 tokens are used as context. The
resulting byte stream is decoded as UTF-8.

## Checkpoints and inspection

Training writes files such as:

```text
checkpoints/checkpoint_0001000.pt
checkpoints/checkpoint_0002000.pt
checkpoints/best_model.pt
```

Each periodic/best checkpoint stores model weights, optimizer state, step,
configuration, and recent loss values. It stores TRAIN and VALIDATION loader
state only; it contains no TEST loader or TEST RNG state. `best_model.pt` is
only updated when a new validation loss is lower than the previous best.
`train.py` prints the programmatically calculated parameter count, architecture
settings, device, losses, token throughput, ETA, and (on CUDA) allocated GPU
memory.

## Important limitations

* More data does not automatically make a tiny model intelligent. Capacity,
  optimization, and context length matter.
* Dataset quality matters: the model learns statistical patterns, including
  errors and biases, from the text supplied to it.
* You are responsible for the licensing and privacy of training data.
* Byte-level tokenization is transparent but inefficient: modern BPE or
  subword tokenizers usually represent common text with fewer tokens.
* This is an educational miniature LLM, not a production or frontier model.
  The included corpus is far too small for broad knowledge or reliable chat.
* CPU training with the default configuration can be slow. Use the smoke-test
  command first, then increase the corpus and training budget. A CUDA build of
  PyTorch can make a substantial difference when a compatible NVIDIA GPU is
  available.

## Reproducibility and safety

The training seed defaults to `1337`, but GPU kernels and hardware can still
introduce small differences. Checkpoints are written through a temporary file
and an atomic rename so an interrupted write does not leave a half checkpoint.
The code executes only local Python/PyTorch/NumPy operations and makes no
network requests.