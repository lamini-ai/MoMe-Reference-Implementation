from transformers import PretrainedConfig

class LlamaMoMEConfig(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=4096,
        sequence_length=512,
        intermediate_size=14336,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        bos_token_id=128000,
        eos_token_id=[128001, 128008, 128009],
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 8.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        attention_bias=False,
        attention_dropout=0.0,
        pretraining_tp=1,
        torch_dtype="bfloat16",
        transformers_version="4.43.1",
        r_value=32,
        index_k=2,
        mlp_bias=False,
        **kwargs
    ):  
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pretraining_tp = pretraining_tp
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version
        self.r_value = r_value
        self.index_k = index_k
        self.mlp_bias = mlp_bias

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )