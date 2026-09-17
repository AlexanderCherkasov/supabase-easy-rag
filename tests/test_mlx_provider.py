import unittest
from unittest.mock import MagicMock

import mlx.core as mx

from supabase_easy_rag.providers.mlx_provider import MlxQwenEmbeddingProvider


class TestMlxQwenEmbeddingProvider(unittest.TestCase):
    def test_dynamic_batch_sizing_heuristics(self):
        provider = MlxQwenEmbeddingProvider(min_batch_size=16, max_batch_size=32, lazy_load=True)

        # Empty texts -> min_batch_size
        self.assertEqual(provider._calculate_dynamic_batch_size([]), 16)

        # Very long texts (> 1000 chars avg) -> min_batch_size 16
        long_texts = ["a" * 1200 for _ in range(20)]
        self.assertEqual(provider._calculate_dynamic_batch_size(long_texts), 16)

        # Short texts (< 300 chars avg) -> max_batch_size 32
        short_texts = ["hello world" for _ in range(40)]
        self.assertEqual(provider._calculate_dynamic_batch_size(short_texts), 32)

        # Intermediate texts (e.g. 650 chars avg) -> dynamic between 16 and 32
        medium_texts = ["m" * 650 for _ in range(20)]
        batch_size = provider._calculate_dynamic_batch_size(medium_texts)
        self.assertGreaterEqual(batch_size, 16)
        self.assertLessEqual(batch_size, 32)

    def test_embed_batch_tokens_bf16_and_normalization(self):
        provider = MlxQwenEmbeddingProvider(lazy_load=True, use_bf16=True)

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda t: [101, 102, 103] if len(t) > 5 else [101, 102]
        mock_tokenizer.pad_token_id = 0

        # Fake callable model returning hidden states of shape (batch, seq_len, hidden_dim)
        class FakeModel:
            def __call__(self, tokens_mx):
                b, s = tokens_mx.shape
                return mx.ones((b, s, 8), dtype=mx.bfloat16)

        provider._tokenizer = mock_tokenizer
        provider._model = FakeModel()

        texts = ["short", "longer text query"]
        embeddings = provider.embed_texts(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(len(embeddings[0]), 8)
        self.assertEqual(len(embeddings[1]), 8)

        # In BF16 precision (16-bit float with 7-bit mantissa), precision is ~0.001 (places=3)
        vec1 = mx.array(embeddings[0])
        norm1 = float(mx.linalg.norm(vec1))
        self.assertAlmostEqual(norm1, 1.0, places=3)

        vec2 = mx.array(embeddings[1])
        norm2 = float(mx.linalg.norm(vec2))
        self.assertAlmostEqual(norm2, 1.0, places=3)

    def test_embed_query(self):
        provider = MlxQwenEmbeddingProvider(lazy_load=True, use_bf16=True)
        provider.embed_texts = MagicMock(return_value=[[0.1] * 8])

        # 1. Raw query without instruction
        res = provider.embed_query("search query")
        self.assertEqual(res, [0.1] * 8)
        provider.embed_texts.assert_called_once_with(["search query"])

        # 2. Explicit instruction passed to embed_query
        provider.embed_texts.reset_mock()
        inst = "Given a web search query, retrieve relevant passages that answer the query"
        res_inst = provider.embed_query("search query", instruction=inst)
        self.assertEqual(res_inst, [0.1] * 8)
        expected_query = f"Instruct: {inst}\nQuery: search query"
        provider.embed_texts.assert_called_once_with([expected_query])

        # 3. Provider with default_instruction configured
        provider_default = MlxQwenEmbeddingProvider(
            lazy_load=True,
            use_bf16=True,
            default_instruction=inst,
        )
        provider_default.embed_texts = MagicMock(return_value=[[0.1] * 8])
        res_def = provider_default.embed_query("search query")
        self.assertEqual(res_def, [0.1] * 8)
        provider_default.embed_texts.assert_called_once_with([expected_query])

    def test_int8_quantization_flag(self):
        provider = MlxQwenEmbeddingProvider(lazy_load=True, quantize_int8=True)
        self.assertTrue(provider.quantize_int8)
        provider_no_quant = MlxQwenEmbeddingProvider(lazy_load=True, quantize_int8=False)
        self.assertFalse(provider_no_quant.quantize_int8)

    def test_custom_dtype_and_quantization(self):
        # Test Literal dtype options
        p_f16 = MlxQwenEmbeddingProvider(lazy_load=True, dtype="float16", quantization=None)
        self.assertEqual(p_f16.dtype, "float16")
        self.assertIsNone(p_f16.quantization)
        self.assertFalse(p_f16.use_bf16)
        self.assertFalse(p_f16.quantize_int8)

        p_f32 = MlxQwenEmbeddingProvider(lazy_load=True, dtype="float32", quantization=4)
        self.assertEqual(p_f32.dtype, "float32")
        self.assertEqual(p_f32.quantization, 4)

    def test_batch_size_and_max_length(self):
        p_fixed = MlxQwenEmbeddingProvider(
            lazy_load=True,
            batch_size=20,
            max_length=4096,
        )
        self.assertEqual(p_fixed.batch_size, 20)
        self.assertEqual(p_fixed.max_length, 4096)
        # Fixed batch size should override dynamic batch sizing
        self.assertEqual(p_fixed._calculate_dynamic_batch_size(["sample"] * 100), 20)


