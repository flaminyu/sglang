from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Model architecture information."""

    hf_config: Optional[object] = None
    model_path: str = ""
    attention_arch: str = "MHA"  # MHA or MLA
    context_len: int = 4096
    hidden_size: int = 4096
    head_dim: int = 128
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: int = 32
    v_head_dim: int = 128
    vocab_size: int = 32000
    torch_dtype: str = "float16"
    # MLA-specific fields
    qk_rope_head_dim: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    kv_lora_rank: Optional[int] = None

    def is_mla(self) -> bool:
        """Check if the model uses MLA attention architecture."""
        return self.attention_arch == "MLA"

    def is_mha(self) -> bool:
        """Check if the model uses MHA attention architecture."""
        return self.attention_arch == "MHA"
