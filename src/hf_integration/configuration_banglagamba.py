from transformers import PretrainedConfig

class BanglaGambaConfig(PretrainedConfig):
    model_type = "banglagamba"

    def __init__(
        self,
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        d_head=64,
        d_ff=2560,
        vocab_size=48000,
        seq_len=2048,
        dropout=0.0,
        bias=False,
        layer_types=None,
        mamba_d_state=128,
        mamba_expand=2,
        mamba_headdim=64,
        mamba_ngroups=1,
        mamba_chunk_size=64,
        rope_base=10000.0,
        rms_norm_eps=1e-5,
        qk_norm=True,
        tie_embeddings=True,
        **kwargs,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.max_position_embeddings = seq_len
        self.dropout = dropout
        self.bias = bias

        # Default layer pattern if none provided: 12 layers alternating Mamba and Attention
        if layer_types is None:
            self.layer_types = [
                "mamba", "attn", "mamba", "attn", "mamba", "attn",
                "mamba", "attn", "mamba", "attn", "mamba", "attn",
            ]
        else:
            self.layer_types = layer_types

        self.mamba_d_state = mamba_d_state
        self.mamba_expand = mamba_expand
        self.mamba_headdim = mamba_headdim
        self.mamba_ngroups = mamba_ngroups
        self.mamba_chunk_size = mamba_chunk_size
        self.rope_base = rope_base
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.tie_embeddings = tie_embeddings

        # Super init handles kwargs drop-in, serialization, etc.
        super().__init__(**kwargs)
