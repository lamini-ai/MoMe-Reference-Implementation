import numpy as np
from transformers import PretrainedConfig
from raft_model.configuration_raft import LlamaRaftConfig
from raft_model.lamini_index import LaminiIndex
import torch.nn.functional as F

import torch
import torch.nn as nn
import copy
import logging
from typing import Optional, List, Dict, Union, Tuple
from transformers.cache_utils import Cache
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class LoraHeadAdaptor(nn.Module):
    def __init__(self, layer, r_value):
        super().__init__()
        self.layer = layer
        self.hidden_size = layer.weight.shape
        # Add a mome attention layer
        self.mlp_lora_in = nn.Linear(self.hidden_size[1], r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, self.hidden_size[0], bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_lora_out.weight.data.zero_()

    # Call layer with all inputs and kwargs
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        results = self.layer(hidden_states)
        lora_results = self.mlp_lora_in(hidden_states)
        lora_results = self.mlp_lora_out(lora_results)

        return results + lora_results

class LoraMLPAdaptor(nn.Module):
    def __init__(self, layer, r_value):
        super().__init__()
        self.layer = layer
        # Get the hidden size
        hidden_size = get_hidden_size(layer)  
        # Add LoRA layers
        self.mlp_lora_in = nn.Linear(hidden_size, r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, hidden_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_lora_out.weight.data.zero_()

    def forward(        
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
        ):
        base_model_results = self.layer(hidden_states)
        lora_results = self.mlp_lora_in(hidden_states)
        lora_results = self.mlp_lora_out(lora_results)
        return base_model_results + lora_results

class LaminiRaftForCausalLM(PreTrainedModel):
    config_class = LlamaRaftConfig
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, config):
        super().__init__(config)
        self.raft_model = None
        self.post_init()

    def initialize(self, model):
        cloned_model = clone_module(model)
        self.embeddings = {
            "key_embeddings": [],
            "value_embeddings": [],
            "embedding_indices": [],
        }
        self.index = LaminiIndex.load_index("nithya/checkpoints/index")
        freeze_all_model_params(cloned_model)

        self.raft_model = add_raft_adaptors_to_each_layer(
        cloned_model, self.config, self.embeddings, self.index)
    
        self.raft_model = add_lora_adaptors_to_mlp_layer(
            self.raft_model,
            LlamaRaftConfig,
        )
        self.raft_model = add_extra_lora_adapters_to_head(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", self.raft_model, LlamaRaftConfig
        )

        mark_only_adapters_as_trainable(self.raft_model)

        if hasattr(self.raft_model, "enable_input_require_grads"):
            self.raft_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.raft_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.raft_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
    def generate(self, input_ids, do_sample, max_new_tokens, return_dict_in_generate):
        return self.raft_model.generate(
            input_ids=input_ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=return_dict_in_generate,
        )

def get_hidden_size(layer):
    logger.debug(f"getting hidden size for layer: {layer}")
    if hasattr(layer, "attention"):
        return get_hidden_size(layer.attention)

    if hasattr(layer, "hidden_size"):
        logger.debug(f"hidden size: {layer.hidden_size} from layer.hidden_size")
        return layer.hidden_size

    if hasattr(layer, "q_proj"):
        logger.debug(f"hidden size: {layer.q_proj.weight.shape[1]} from layer.q_proj")
        return layer.q_proj.weight.shape[1]

    if hasattr(layer, "out_proj"):
        logger.debug(
            f"hidden size: {layer.out_proj.weight.shape[1]} from layer.out_proj"
        )
        return layer.out_proj.weight.shape[1]

    if hasattr(layer, "c_fc"):
        logger.debug(f"hidden size: {layer.c_fc.weight.shape[1]} from layer.c_fc")
        return layer.c_fc.weight.shape[1]

    if hasattr(layer, "fc2"):
        logger.debug(f"hidden size: {layer.fc2.weight.shape[0]} from layer.fc2")
        return layer.fc2.weight.shape[0]

    if hasattr(layer, "gate_up_proj"):
        logger.debug(
            f"hidden size: {layer.gate_up_proj.weight.shape[1]} from layer.gate_up_proj"
        )
        return layer.gate_up_proj.weight.shape[1]

    assert hasattr(layer, "c_proj")
    logger.debug(f"hidden size: {layer.c_proj.weight.shape[1]} from layer.c_proj")
    return layer.c_proj.weight.shape[1]

def clone_module(module):
    """Make a shallow copy of a module recursively."""
    shallow_copy = copy.copy(module)
    shallow_copy._parameters = shallow_copy._parameters.copy()
    shallow_copy._buffers = shallow_copy._buffers.copy()
    shallow_copy._modules = shallow_copy._modules.copy()

    for child_name, child in shallow_copy.named_children():
        shallow_copy._modules[child_name] = clone_module(child)
    return shallow_copy

def add_raft_adaptors_to_each_layer(model, config, embeddings, index): 
    """The RaftAdaptor wraps and replaces layers if they are attention layers."""
    for name, layer in model.named_modules():
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_self_attn(name, layer, model, config, embeddings, index)
    return model

def is_tiny_lm_head_layer(base_model_name: str, name: str):
    # if "hf-internal-testing" not in base_model_name:
    #    return False
    return any(prefix in name for prefix in ["lm_head"])
    
MOME_ADAPTER_PREFIXES = [
    "mome_attention.attn",
    "mome_attention.query_projection_lora_in",
    "mome_attention.query_projection_lora_out",
    "mome_layer_norm",
    "mlp_lora_in",
    "mlp_lora_out",
    "mlp_layer_norm",
    "lm_head_lora_in",
    "lm_head_lora_out",
]

def is_mome_adapter_layer(name: str):
    return any(prefix in name for prefix in MOME_ADAPTER_PREFIXES)

def add_lora_adaptors_to_mlp_layer(model: PreTrainedModel, config: LlamaRaftConfig):
    """Add LoRa adapters to MLP layers."""
    for name, layer in model.named_children():  
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_mlp(name, layer, model, config)
    return model

def add_extra_lora_adapters_to_head(
    base_model_name: str, model: PreTrainedModel, config: LlamaRaftConfig):
    """Add LoRa adapters to the head."""
    logger.info("Adding LoRa adapters to the head")
    for name, layer in model.named_modules():
        if is_tiny_lm_head_layer(base_model_name, name):
            logger.info(f"Wrapping layer {name} with LoraHeadAdaptor")
            recursive_setattr(model, name, LoraHeadAdaptor(layer, 64))
    return model

def try_to_update_self_attn(name, layer, model, config: LlamaRaftConfig, embeddings, index):
    """Try to wrap the layer with a RaftAdaptor."""
    if not is_self_attn_layer(layer, name):
        return

    logger.info(f"Wrapping layer {name} with RaftAdaptor")

    # Wrap the layer with a RaftAdaptor
    recursive_setattr(model, name, RaftAdaptor(layer, embeddings, index))

def try_to_update_mlp(name, layer, model, config):
    """Try to wrap the layer with a LoraMLPAdaptor."""
    if not is_mlp_layer(name):
        return

    logger.info(f"Wrapping layer {name} with LoraMLPAdaptor")

    # Wrap the layer with a RAFTAdaptor
    recursive_setattr(model, name, LoraMLPAdaptor(layer, 64))

def freeze_all_model_params(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad = False
        logger.debug(
            "Before Params: " + str(n) + " requires grad: " + str(p.requires_grad)
        )
def mark_only_adapters_as_trainable(model: nn.Module):
    for n, p in model.named_parameters():
        if is_mome_adapter_layer(n):
            p.requires_grad = True

def is_self_attn_layer(layer, name):
    """Check if it is a huggingface attention layer."""
    name_suffix = name.split(".")[-1]

    # huggingface calls it self_attn
    if name_suffix == "self_attn":
        return True

    return False

def is_mlp_layer(name):
    """Check if it is a huggerface mlp layer."""
    name_suffix = name.split(".")[-1]

    if name_suffix == "mlp":
        return True

    return False

def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

class RaftAdaptor(nn.Module):
    def __init__(self, layer, embeddings, index):
        super().__init__()
        self.layer = layer

        # Add a raft attention layer
        self.raft_attention = RaftAttentionLayer(
            hidden_size=layer.hidden_size,
            r_value = 64,
            raft_embedding_seq_length = 512,
            device=layer.q_proj.weight.device,
            embeddings=embeddings,
            index=index,
            index_k = 2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        layer_outputs = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # project the raft attention output to the same size as the transformer attention output
        raft_attention_output = self.raft_attention(hidden_states)

        if raft_attention_output.size(1) != layer_outputs[0].size(1):
            raft_attention_output = raft_attention_output[:, :layer_outputs[0].size(1), :]

        combined_output = layer_outputs[0] + raft_attention_output
        return (
            combined_output,) + layer_outputs[1:]

raft_embedding_batch_size = 1
raft_embedding_seq_length = 512
attention_head_count = 8


class RaftAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        r_value,
        raft_embedding_seq_length,
        device,
        embeddings,
        index: LaminiIndex,
        index_k,
    ):
        super().__init__()
        self.index = index
        self.index_k = index_k
        self.index_dimension = min(
            index.embedding_dimension, hidden_size
        )
        self.embeddings = embeddings
        self.key_embedding = nn.Parameter(
            torch.zeros(
                1,
                raft_embedding_seq_length * self.index_k,
                self.index_dimension,
            )
        )
        self.value_embedding = nn.Parameter(
            torch.zeros(
                1,
                raft_embedding_seq_length * self.index_k,
                self.index_dimension,
            )
        )
        self.embedding_index = len(embeddings["key_embeddings"])
        self.embeddings["key_embeddings"].append(self.key_embedding)
        self.embeddings["value_embeddings"].append(self.value_embedding)
        self.embeddings["embedding_indices"].append([])
        self.query_projection_lora_in = nn.Linear(hidden_size, r_value, bias=False)
        self.query_projection_lora_out = nn.Linear(
            r_value, self.index_dimension, bias=False
        )
        self.value_projection_lora_in = nn.Linear(self.index_dimension, r_value, bias=False)
        self.value_projection_lora_out = nn.Linear(r_value, hidden_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        self.value_projection_lora_out.weight.data.zero_()
        self.query_projection_lora_out.weight.data.zero_()

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        query = self.get_query(hidden_states)
        key, value = self.get_key_and_value(query)
        
        key = key.to(query.dtype)
        value = value.to(query.dtype)
        
        # Ensure all tensors have the same sequence length
        seq_len = hidden_states.size(1)
        query = query[:, :seq_len, :]
        key = key[:, :seq_len, :]
        value = value[:, :seq_len, :]
        
        raft_attention_output = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attention_mask,
            dropout_p=0.1,
            is_causal=True,
            scale=None,
        )
        projected_raft_attention_output = self.project_value(raft_attention_output)
        return projected_raft_attention_output

    def project_value(self, value):
        value = self.value_projection_lora_in(value)
        value = self.value_projection_lora_out(value)
        return value

    def get_query(self, hidden_states):
        query = self.query_projection_lora_in(hidden_states)
        query = self.query_projection_lora_out(query)
        return query

    def get_key_and_value(self, query):
        key, value, indices = self.get_key_and_value_from_index(query)
        self.embeddings["embedding_indices"][self.embedding_index] = indices
        batch_size = key.shape[0]
        k_times_sequence_length = key.shape[1]
        if self.training:
            with torch.no_grad():
                self.key_embedding[:batch_size, :k_times_sequence_length, :].copy_(key)
                self.value_embedding[:batch_size, :k_times_sequence_length, :].copy_(value)
                del key
                del value
            return (
                self.key_embedding[:batch_size, :k_times_sequence_length, :],
                self.value_embedding[:batch_size, :k_times_sequence_length, :],
            )
        else:
            return key, value

    def get_key_and_value_from_index(self, query):
        batch_size, sequence_length, embedding_dimension = query.shape
        device = query.device
        query_new = query.view(batch_size * sequence_length, embedding_dimension)
        
        with torch.no_grad():
            query_new = query_new.float().cpu().numpy()
            key, value, indices = self.index.get_key_and_value(query_new, k=self.index_k)
            
            # Ensure key and value have the correct shape
            key = np.array(key).reshape(batch_size, sequence_length * self.index_k, -1)
            value = np.array(value).reshape(batch_size, sequence_length * self.index_k, -1)
            
            key = torch.from_numpy(key).to(device)
            value = torch.from_numpy(value).to(device)
            
            # Ensure the last dimension matches self.index_dimension
            if key.shape[-1] > self.index_dimension:
                key = key[:, :, :self.index_dimension]
            if value.shape[-1] > self.index_dimension:
                value = value[:, :, :self.index_dimension]
            
            # If the last dimension is smaller, pad it
            if key.shape[-1] < self.index_dimension:
                key = nn.functional.pad(key, (0, self.index_dimension - key.shape[-1]))
            if value.shape[-1] < self.index_dimension:
                value = nn.functional.pad(value, (0, self.index_dimension - value.shape[-1]))
        
        return key, value, indices