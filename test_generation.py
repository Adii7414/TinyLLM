"""Correctness tests for cached autoregressive generation."""

import unittest

import torch

from config import Config
from generate import generate_tokens
from model import GPTModel


class GenerationTests(unittest.TestCase):
    @staticmethod
    def make_model() -> GPTModel:
        config = Config(
            vocab_size=23,
            context_length=6,
            embedding_dim=12,
            num_layers=2,
            num_heads=3,
            feed_forward_dim=24,
            dropout=0.0,
        )
        torch.manual_seed(7)
        return GPTModel(config).eval()

    def test_cached_and_uncached_generation_match_across_context_rollover(self) -> None:
        model = self.make_model()
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]

        torch.manual_seed(123)
        uncached = generate_tokens(
            model,
            prompt,
            max_tokens=8,
            temperature=0.8,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.08,
            use_kv_cache=False,
        )
        torch.manual_seed(123)
        cached = generate_tokens(
            model,
            prompt,
            max_tokens=8,
            temperature=0.8,
            top_k=5,
            top_p=0.9,
            repetition_penalty=1.08,
            use_kv_cache=True,
        )

        self.assertEqual(cached, uncached)

    def test_cached_prefill_logits_match_full_forward(self) -> None:
        model = self.make_model()
        prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        full_logits, _ = model(prompt)
        cached_logits, _, cache = model(prompt, use_cache=True)

        self.assertTrue(torch.allclose(full_logits, cached_logits, atol=1e-6, rtol=1e-5))
        self.assertEqual(len(cache), model.config.num_layers)
        self.assertEqual(cache[0][0].size(2), prompt.size(1))

    def test_cached_decode_logits_match_full_forward_before_context_limit(self) -> None:
        model = self.make_model()
        generated = [1, 2, 3]
        prompt = torch.tensor([generated], dtype=torch.long)
        cached_logits, _, cache = model(prompt, use_cache=True)

        for _ in range(2):
            full_logits, _ = model(
                torch.tensor([generated], dtype=torch.long)
            )
            self.assertTrue(
                torch.allclose(
                    full_logits[:, -1],
                    cached_logits[:, -1],
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
            next_token = int(torch.argmax(cached_logits[:, -1]).item())
            generated.append(next_token)
            cached_logits, _, cache = model(
                torch.tensor([[next_token]], dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                position_offset=cache[0][0].size(2),
            )

    def test_cached_generation_stops_at_eos(self) -> None:
        model = self.make_model()
        prompt = [1, 2, 3, 4]
        prompt_tensor = torch.tensor([prompt], dtype=torch.long)
        first_logits, _ = model(prompt_tensor)
        expected_eos = int(torch.argmax(first_logits[0, -1]).item())

        cached = generate_tokens(
            model,
            prompt,
            max_tokens=5,
            temperature=0,
            eos_token_id=expected_eos,
            use_kv_cache=True,
        )
        uncached = generate_tokens(
            model,
            prompt,
            max_tokens=5,
            temperature=0,
            eos_token_id=expected_eos,
            use_kv_cache=False,
        )

        self.assertEqual(cached, uncached)
        self.assertEqual(cached[-1], expected_eos)
        self.assertEqual(len(cached), len(prompt) + 1)


if __name__ == "__main__":
    unittest.main()