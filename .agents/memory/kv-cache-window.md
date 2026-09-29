---
name: Sliding-window KV cache
description: Why bounded autoregressive KV caches retain absolute RoPE positions when generation exceeds the attention window.
---

For generation beyond the configured context length, evict the oldest cached
key/value entries and retain their original RoPE positions. Recomputing the
entire rebased window is not equivalent to one-token decoding because cached
higher-layer states already include the evicted token's causal influence.

**Why:** A rolling window changes the historical context of every retained
token. Reprocessing the window would be more exact relative to a fresh
forward pass, but it defeats the one-new-token generation path and is not the
standard KV-cache semantics.

**How to apply:** Keep the cache bounded to the model context, use explicit
position IDs for newly decoded tokens, and compare cached-vs-uncached logits
within the active context window. Test that generation continues safely after
cache eviction without exceeding the configured context length.