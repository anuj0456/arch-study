from transformers.configuration_utils import PretrainedConfig


class KiteFishConfig(PretrainedConfig):

    model_type = "kitefish_v1"

    def __init__(
        self,
        vocab_size=128262, # vocab size
        hidden_size=2048, # d_ff same as paper
        num_hidden_layers=18, # num of layers or n, originally = 6, Total number of encode & decoder stack on each other
        num_attention_heads=16, #number of heads or h, originally = 8
        intermediate_size=8192, #feed-forward block
        max_position_embeddings=512, #same as paper, d_model
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

        super().__init__(**kwargs)