from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from supabase_easy_rag.providers.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class MlxQwenEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using Apple Silicon MLX with Qwen3-Embedding-0.6B.

    Features:
    - Apple Silicon Metal acceleration via MLX
    - BF16 precision (mx.bfloat16) or INT8 quantization (bits=8) for memory efficiency & performance
    - Dynamic batching (16–32 items) based on sequence count/lengths
    - L2 normalization for cosine similarity search
    - Lazy model loading so initialization is lightweight
    """

    def __init__(
        self,
        model_path_or_repo: str | Path = "Qwen/Qwen3-Embedding-0.6B",
        min_batch_size: int = 16,
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
        use_bf16: bool = True,
        quantize_int8: bool = True,
        lazy_load: bool = True,
        default_instruction: str | None = None,
    ) -> None:
        self.model_path_or_repo = str(model_path_or_repo)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.use_bf16 = use_bf16
        self.quantize_int8 = quantize_int8
        self.default_instruction = default_instruction

        self._model: Any = None
        self._tokenizer: Any = None

        if not lazy_load:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import mlx.core as mx  # noqa: F401
            import mlx.nn as nn
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "MLX packages are required for MlxQwenEmbeddingProvider. "
                "Install with `pip install mlx mlx-lm tokenizers` on Apple Silicon."
            ) from exc

        logger.info("Loading Qwen embedding model via MLX from %s...", self.model_path_or_repo)
        
        # Load tokenizer
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path_or_repo))

        # Check model config and load weights directly into Qwen3Model
        import json
        from pathlib import Path
        import mlx.core as mx
        from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

        model_dir = Path(self.model_path_or_repo)
        config_path = model_dir / "config.json" if model_dir.is_dir() else None
        
        if config_path and config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            weights_file = model_dir / "model.safetensors"
            weights = mx.load(str(weights_file)) if weights_file.exists() else None
        else:
            weights = None
            cfg = None

        if weights is not None and cfg is not None:
            # Construct model args matching Qwen3
            model_args = ModelArgs(
                model_type=cfg.get("model_type", "qwen3"),
                hidden_size=cfg["hidden_size"],
                num_hidden_layers=cfg["num_hidden_layers"],
                intermediate_size=cfg["intermediate_size"],
                num_attention_heads=cfg["num_attention_heads"],
                rms_norm_eps=cfg["rms_norm_eps"],
                vocab_size=cfg["vocab_size"],
                num_key_value_heads=cfg["num_key_value_heads"],
                max_position_embeddings=cfg.get("max_position_embeddings", 32768),
                rope_theta=cfg.get("rope_theta", 1000000),
                head_dim=cfg.get("head_dim", 128),
                tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            )
            model = Qwen3Model(model_args)

            loaded_weights = {}
            for k, v in weights.items():
                if k.startswith("model."):
                    loaded_weights[k[6:]] = v
                else:
                    loaded_weights[k] = v

            model.load_weights(list(loaded_weights.items()))
        else:
            from mlx_lm import load
            model, _ = load(self.model_path_or_repo)

        if self.quantize_int8:
            logger.info("Quantizing model weights to INT8 (bits=8)...")
            nn.quantize(model, bits=8)

        self._model = model

    def _calculate_dynamic_batch_size(self, texts: Sequence[str]) -> int:
        """Dynamically compute batch size between min_batch_size (16) and max_batch_size (32)

        Heuristic:
        - If texts are long on average (> 1000 characters), use min_batch_size (16) to avoid memory spikes.
        - If texts are short (< 300 characters), use max_batch_size (32) for maximum throughput.
        - Linearly interpolate in between.
        """
        if not texts:
            return self.min_batch_size

        sample_size = min(len(texts), 32)
        avg_len = sum(len(texts[i]) for i in range(sample_size)) / sample_size

        if avg_len >= 1000:
            return self.min_batch_size
        if avg_len <= 300:
            return self.max_batch_size

        # Interpolate between 16 and 32
        ratio = (1000 - avg_len) / (1000 - 300)
        dynamic_size = int(self.min_batch_size + ratio * (self.max_batch_size - self.min_batch_size))
        return max(self.min_batch_size, min(self.max_batch_size, dynamic_size))

    def _embed_batch_tokens(self, batch_texts: list[str]) -> list[list[float]]:
        import mlx.core as mx

        self._ensure_loaded()

        encoded_tokens = [self._tokenizer.encode(t) for t in batch_texts]
        max_len = max((len(tok) for tok in encoded_tokens), default=0)
        pad_id = getattr(self._tokenizer, "pad_token_id", 0) or 0

        # Pad tokens to same length
        padded = []
        masks = []
        for tok in encoded_tokens:
            pad_len = max_len - len(tok)
            padded.append(tok + [pad_id] * pad_len)
            masks.append([1.0] * len(tok) + [0.0] * pad_len)

        tokens_mx = mx.array(padded, dtype=mx.int32)
        mask_mx = mx.array(masks, dtype=mx.bfloat16 if self.use_bf16 else mx.float32)

        # Forward pass through model
        if hasattr(self._model, "model") and hasattr(self._model.model, "embed_tokens"):
            hidden_states = self._model.model(tokens_mx)
        else:
            hidden_states = self._model(tokens_mx)

        if hasattr(hidden_states, "last_hidden_state"):
            hidden_states = hidden_states.last_hidden_state

        if self.use_bf16 and hidden_states.dtype != mx.bfloat16:
            hidden_states = hidden_states.astype(mx.bfloat16)

        # Masked mean pooling
        expanded_mask = mask_mx[:, :, None]
        sum_embeddings = mx.sum(hidden_states * expanded_mask, axis=1)
        sum_mask = mx.clip(mx.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        norms = mx.linalg.norm(pooled, axis=-1, keepdims=True)
        normalized = pooled / mx.clip(norms, a_min=1e-9, a_max=None)

        # Convert to float32 Python lists
        embeddings = normalized.astype(mx.float32).tolist()
        return embeddings

    def embed_query(self, query: str, instruction: str | None = None) -> list[float]:
        """Generate an embedding vector for a search query.

        If instruction is provided (or default_instruction is set), uses Qwen instruction formatting.
        Otherwise, embeds the raw query directly (which delivers 80.5%+ retrieval accuracy on Qwen3-0.6B).
        """
        inst = instruction if instruction is not None else self.default_instruction
        if inst:
            formatted_query = f"Instruct: {inst}\nQuery: {query}"
        else:
            formatted_query = query
        results = self.embed_texts([formatted_query])
        if not results:
            raise RuntimeError("Embedding provider returned empty result for query")
        return results[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        text_list = list(texts)
        dynamic_batch_size = self._calculate_dynamic_batch_size(text_list)
        all_embeddings: list[list[float]] = []

        for i in range(0, len(text_list), dynamic_batch_size):
            batch = text_list[i : i + dynamic_batch_size]
            batch_vectors = self._embed_batch_tokens(batch)
            all_embeddings.extend(batch_vectors)

        return all_embeddings
