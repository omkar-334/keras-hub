import numpy as np

from keras_hub.src.models.deberta_v3.deberta_v3_backbone import DebertaV3Backbone
from keras_hub.src.utils.preset_utils import HF_TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = DebertaV3Backbone


def convert_backbone_config(transformers_config):
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
    }


def convert_weights(backbone, loader, transformers_config):
    # Embedding layer
    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key="embeddings.word_embeddings.weight",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").beta,
        hf_weight_key="embeddings.LayerNorm.bias",
    )
    loader.port_weight(
        keras_variable=backbone.get_layer("embeddings_layer_norm").gamma,
        hf_weight_key="embeddings.LayerNorm.weight",
    )

    def transpose_and_reshape(x, shape):
        return np.reshape(np.transpose(x), shape)

    # Attention blocks
    for i in range(backbone.num_layers):
        block = backbone.get_layer(f"disentangled_attention_encoder_layer_{i}")
        attn = block._self_attention_layer
        hf_prefix = "encoder.layer."
        # Attention layers
        loader.port_weight(
            keras_variable=attn._query_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.query_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn._key_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.key_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn._value_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.self.value_proj.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn._output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=attn._output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.dense.bias",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.LayerNorm.bias",
        )
        loader.port_weight(
            keras_variable=block._self_attention_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.attention.output.LayerNorm.weight",
        )
        # Intermediate layer
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.bias",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.intermediate.dense.weight",
            hook_fn=transpose_and_reshape,
        )

        loader.port_weight(
            keras_variable=block._feedforward_output_dense.bias,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.bias",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._feedforward_output_dense.kernel,
            hf_weight_key=f"{hf_prefix}{i}.output.dense.weight",
            hook_fn=transpose_and_reshape,
        )
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.output.LayerNorm.bias",
        )
        loader.port_weight(
            keras_variable=block._feedforward_layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.output.LayerNorm.weight",
        )
        # Relative Embeddings
        loader.port_weight(
            keras_variable=backbone.get_layer("relative_embeddings").rel_embeddings,
            hf_weight_key=f"{hf_prefix}{i}.rel_embeddings.weight",
        )
        loader.port_weight(
            keras_variable=backbone.get_layer("relative_embeddings").layer_norm.beta,
            hf_weight_key=f"{hf_prefix}{i}.LayerNorm.bias",
        )
        loader.port_weight(
            keras_variable=backbone.get_layer("relative_embeddings").layer_norm.gamma,
            hf_weight_key=f"{hf_prefix}{i}.LayerNorm.weight",
        )


def convert_tokenizer(cls, preset, **kwargs):
    transformers_config = load_json(preset, HF_TOKENIZER_CONFIG_FILE)
    return cls(
        get_file(preset, "vocab.txt"),
        lowercase=transformers_config["do_lower_case"],
        **kwargs,
    )
