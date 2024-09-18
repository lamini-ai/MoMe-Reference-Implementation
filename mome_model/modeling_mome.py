import os
import copy
import torch
import logging
import inspect
import numpy as np
from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.utils import PushToHubMixin
from safetensors.torch import load_file as safe_load_file
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM

from mome_model.lamini_index import LaminiIndex
from mome_model.configuration_mome import LlamaMoMEConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def infer_device():
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"

def load_base_model_and_tokenizer(base_model_name, device):
    token = os.getenv("HF_API_TOKEN")
    if not token:
        raise ValueError("HF_API_TOKEN environment variable is not set")

    logger.debug(f"Loading base model and tokenizer... {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)

    return {
        "model": base_model,
        "tokenizer": tokenizer,
        "model_name": base_model_name,
        "device": device,
    }

def load_mome_model_for_inference(base_model, path):
    logger.debug(f"Loading MoME model for inference from path: {path}")
    model = PretrainedLaminiMoMEForCausalLM.from_pretrained(base_model, os.path.abspath(path))
    model.device = base_model.device
    prepare_mome_model_for_inference(base_model, model)
    return model

def prepare_mome_model_for_inference(base_model, model):
    attributes_to_copy = [
        "generation_config", "config", "_validate_model_class", "_validate_model_kwargs",
        "_prepare_model_inputs", "_prepare_attention_mask_for_generation",
        "_validate_generated_length", "_extract_past_from_model_output",
        "_get_logits_processor", "_get_stopping_criteria", "prepare_inputs_for_generation",
        "_update_model_kwargs_for_generation", "_get_initial_cache_position",
        "_supports_default_dynamic_cache"
    ]
    
    for attr in attributes_to_copy:
        setattr(model, attr, getattr(base_model, attr))
    
    if hasattr(model, "_get_generation_mode"):
        model._get_generation_mode = base_model._get_generation_mode
    
    logger.info(f"Loaded MoME model: {model}")
    return model

def load_mome_weights(model_id: str, device: Optional[str] = None) -> dict:
    device = device or infer_device()
    safetensors_path = os.path.join(model_id, "adapter_model.safetensors")
    
    if os.path.exists(safetensors_path):
        return safe_load_file(safetensors_path, device=device)
    else:
        raise FileNotFoundError(f"MoME weights not found at path: {model_id}")

def set_mome_model_state_dict_for_inference(model, peft_model_state_dict, base_model_name):
    state_dict = {k.split("mome_model.")[1]: v for k, v in peft_model_state_dict.items() 
                  if is_mome_adapter_layer(k) or is_tiny_lm_head_layer(base_model_name, k)}
    
    state_dict, mismatched_keys = find_mismatched_keys(model, state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)

    if mismatched_keys:
        mismatched_warning = "\n".join(
            [f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
             for key, shape1, shape2 in mismatched_keys]
        )
        msg = (f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint "
               f"and are being ignored because you passed `ignore_mismatched_sizes=True`: {mismatched_warning}.")
        raise ValueError(msg)
    return load_result

def clone_module(module):
    shallow_copy = copy.copy(module)
    shallow_copy._parameters = shallow_copy._parameters.copy()
    shallow_copy._buffers = shallow_copy._buffers.copy()
    shallow_copy._modules = shallow_copy._modules.copy()

    for child_name, child in shallow_copy.named_children():
        shallow_copy._modules[child_name] = clone_module(child)
    return shallow_copy

def add_mome_adaptors_to_each_layer(model, config, embeddings, index):
    for name, layer in model.named_modules():
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_self_attn(name, layer, model, config, embeddings, index)
    return model

def add_lora_adaptors_to_mlp_layer(model: PreTrainedModel, config: LlamaMoMEConfig):
    for name, layer in model.named_modules():  
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_mlp(name, layer, model, config)
    return model

def add_extra_lora_adapters_to_head(base_model_name: str, model: PreTrainedModel, config: LlamaMoMEConfig):
    logger.info("Adding LoRa adapters to the head")
    for name, layer in model.named_modules():
        if is_tiny_lm_head_layer(base_model_name, name):
            logger.info(f"Wrapping layer {name} with LoraHeadAdaptor")
            recursive_setattr(model, name, LoraHeadAdaptor(layer, 32))
    return model

def try_to_update_self_attn(name, layer, model, config: LlamaMoMEConfig, embeddings, index):
    if not is_self_attn_layer(layer, name):
        return

    logger.info(f"Wrapping layer {name} with MoMEAdaptor")
    recursive_setattr(model, name, MoMEAdaptor(
        layer,
        embeddings,
        index,
        config.r_value,
        config.sequence_length,
        config.index_k,
        requires_attention_output=get_requires_output_attentions(layer),
    ))

def try_to_update_mlp(name, layer, model, config):
    if not is_mlp_layer(name):
        return

    logger.info(f"Wrapping layer {name} with LoraMLPAdaptor")
    recursive_setattr(model, name, LoraMLPAdaptor(layer, 32))

def is_tiny_lm_head_layer(base_model_name: str, name: str):
    return any(prefix in name for prefix in ["lm_head"])

def recursive_setattr(obj, attr, value):
    attr = attr.split(".", 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)

def is_self_attn_layer(layer, name):
    return name.split(".")[-1] == "self_attn"

def get_hidden_size(layer):
    logger.debug(f"getting hidden size for layer: {layer}")
    if hasattr(layer, "attention"):
        return get_hidden_size(layer.attention)
    
    attributes_to_check = [
        ("hidden_size", lambda x: x),
        ("q_proj", lambda x: x.weight.shape[1]),
        ("out_proj", lambda x: x.weight.shape[1]),
        ("c_fc", lambda x: x.weight.shape[1]),
        ("fc2", lambda x: x.weight.shape[0]),
        ("gate_up_proj", lambda x: x.weight.shape[1]),
        ("c_proj", lambda x: x.weight.shape[1])
    ]
    
    for attr, get_size in attributes_to_check:
        if hasattr(layer, attr):
            size = get_size(getattr(layer, attr))
            logger.debug(f"hidden size: {size} from layer.{attr}")
            return size
    
    raise AttributeError("Could not determine hidden size for layer")

def get_device(layer):
    return next(layer.parameters()).device

def is_mlp_layer(name):
    return name.split(".")[-1] == "mlp"

def is_mome_adapter_layer(name: str):
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
    return any(prefix in name for prefix in MOME_ADAPTER_PREFIXES)

def find_mismatched_keys(
    model: torch.nn.Module,
    peft_model_state_dict: dict[str, torch.Tensor],
    ignore_mismatched_sizes: bool = False,
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    if not ignore_mismatched_sizes:
        return peft_model_state_dict, []

    mismatched = []
    state_dict = model.state_dict()
    for key, tensor in peft_model_state_dict.items():
        if key not in state_dict:
            continue

        if (state_dict[key].shape[-1] == 1) and (state_dict[key].numel() * 2 == tensor.numel()):
            continue

        if state_dict[key].shape != tensor.shape:
            mismatched.append((key, tensor.shape, state_dict[key].shape))

    for key, _, _ in mismatched:
        del peft_model_state_dict[key]

    return peft_model_state_dict, mismatched

class PretrainedLaminiMoMEForCausalLM(PushToHubMixin, torch.nn.Module):
    config_class = LlamaMoMEConfig
    base_model_prefix = "mome_model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def __init__(self, base_model: PreTrainedModel, config: LlamaMoMEConfig):
        super().__init__()
        logger.debug(f"PretrainedLaminiMoMEForCausalLM name_or_path {base_model}")
        logger.debug(f"PretrainedLaminiMoMEForCausalLM config {config}")
        index_path = "model/index"
        self.index = LaminiIndex.load_index(index_path)

        self.embeddings = {
            "key_embeddings": [],
            "value_embeddings": [],
            "embedding_indices": [],
        }

        cloned_model = clone_module(base_model)
        self.mome_model = add_mome_adaptors_to_each_layer(cloned_model, config, self.embeddings, self.index)
        self.mome_model = add_lora_adaptors_to_mlp_layer(self.mome_model, config)
        self.mome_model = add_extra_lora_adapters_to_head(base_model.name_or_path, self.mome_model, config)
        self.load_adapter("model/checkpoint", cloned_model.name_or_path)

    @classmethod
    def from_pretrained(cls, model, model_id):
        config = LlamaMoMEConfig.from_pretrained(model_id)
        return cls(model, config)

    def load_adapter(self, model_id: str, base_model_name: str, is_trainable: bool = False):
        torch_device = infer_device()
        adapters_weights = load_mome_weights(model_id, device=torch_device)
        load_result = set_mome_model_state_dict_for_inference(self.mome_model, adapters_weights, base_model_name)
        if not is_trainable:
            self.mome_model.eval()
        return load_result

    def forward(self, *args, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
        return self.mome_model.forward(*args, **kwargs)

    def generate(self, input_ids, do_sample, max_new_tokens, return_dict_in_generate):
        logger.debug(f"input_ids in generate: {input_ids} on device: {input_ids.device}")
        return self.mome_model.generate(
            input_ids=input_ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=return_dict_in_generate,
        )

class MoMEAdaptor(nn.Module):
    def __init__(self, layer, embeddings, index, r_value, mome_embedding_seq_length, index_k, requires_attention_output):
        super().__init__()
        self.layer = layer
        self.requires_attention_output = requires_attention_output
        self.mome_attention = MoMEAttentionLayer(
            hidden_size=get_hidden_size(layer),
            r_value=r_value,
            mome_embedding_seq_length=mome_embedding_seq_length,
            device=get_device(layer),
            embeddings=embeddings,
            index=index,
            index_k=index_k,
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
                output_attentions: bool = False, use_cache: bool = False, **kwargs):
        self.update_kwargs(kwargs, past_key_value=past_key_value, position_ids=position_ids)
        if self.requires_attention_output:
            output_attentions = True
        layer_outputs = self.layer(hidden_states=hidden_states, attention_mask=attention_mask,
                                   output_attentions=output_attentions, use_cache=use_cache, **kwargs)
        mome_attention_output = self.mome_attention(hidden_states)
        layer_and_adaptor_sum = layer_outputs[0] + mome_attention_output
        return (layer_and_adaptor_sum,) + layer_outputs[1:]

    def update_kwargs(self, kwargs, past_key_value, position_ids):
        args = inspect.getfullargspec(self.layer.forward).args
        if past_key_value is not None:
            if "past_key_value" in args:
                kwargs["past_key_value"] = past_key_value
            elif "layer_past" in args:
                kwargs["layer_past"] = past_key_value
            else:
                assert False, "Could not figure out how to pass past_key_value"
        if position_ids is not None and "position_ids" in args:
            kwargs["position_ids"] = position_ids

class MoMEAttentionLayer(nn.Module):
    def __init__(self, hidden_size, r_value, mome_embedding_seq_length, device, embeddings, index: LaminiIndex, index_k):
        super().__init__()
        self.index = index
        self.index_k = index_k
        self.index_dimension = min(index.embedding_dimension, hidden_size)
        self.embeddings = embeddings
        self.key_embedding = nn.Parameter(torch.zeros(1, mome_embedding_seq_length * self.index_k, self.index_dimension))
        self.value_embedding = nn.Parameter(torch.zeros(1, mome_embedding_seq_length * self.index_k, self.index_dimension))
        self.embedding_index = len(embeddings["key_embeddings"])
        self.embeddings["key_embeddings"].append(self.key_embedding)
        self.embeddings["value_embeddings"].append(self.value_embedding)
        self.embeddings["embedding_indices"].append([])
        self.query_projection_lora_in = nn.Linear(hidden_size, r_value, bias=False)
        self.query_projection_lora_out = nn.Linear(r_value, self.index_dimension, bias=False)
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
        
        mome_attention_output = torch.nn.functional.scaled_dot_product_attention(
            query=query, key=key, value=value, attn_mask=attention_mask,
            dropout_p=0.1, is_causal=True, scale=None,
        )
        projected_mome_attention_output = self.project_value(mome_attention_output)
        return projected_mome_attention_output

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
        logger.debug(f"Query shape: {query.shape}")
        logger.debug(f"batch_size: {batch_size}, sequence_length: {sequence_length}, embedding_dimension: {embedding_dimension}")
        logger.debug(f"self.index_dimension: {self.index_dimension}")
        logger.debug(f"self.index_k: {self.index_k}")

        device = query.device
        query_new = query.view(batch_size * sequence_length, embedding_dimension)
        logger.debug(f"query_new shape: {query_new.shape}")

        with torch.no_grad():
            query_new = query_new.float().cpu().numpy()
            key, value, indices = self.index.get_key_and_value(query_new, k=self.index_k)
            
            logger.debug(f"key shape from index: {np.array(key).shape}")
            logger.debug(f"value shape from index: {np.array(value).shape}")

            key = np.array(key)
            value = np.array(value)

            expected_kv_pairs = batch_size * sequence_length * self.index_k
            if key.shape[0] != expected_kv_pairs:
                logger.warning(f"Expected {expected_kv_pairs} key-value pairs, but got {key.shape[0]}. Adjusting...")
                key = np.repeat(key, repeats=self.index_k, axis=0)[:expected_kv_pairs]
                value = np.repeat(value, repeats=self.index_k, axis=0)[:expected_kv_pairs]

            key = torch.from_numpy(key).to(device)
            value = torch.from_numpy(value).to(device)

            logger.debug(f"key shape before reshape: {key.shape}")
            
            key = key.view(batch_size, sequence_length * self.index_k, self.index_dimension)
            value = value.view(batch_size, sequence_length * self.index_k, self.index_dimension)

            logger.debug(f"key shape after reshape: {key.shape}")
            logger.debug(f"value shape after reshape: {value.shape}")

        return key, value, indices

class LoraMLPAdaptor(nn.Module):
    def __init__(self, layer, r_value):
        super().__init__()
        self.layer = layer
        hidden_size = get_hidden_size(layer)
        self.mlp_lora_in = nn.Linear(hidden_size, r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, hidden_size, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_lora_out.weight.data.zero_()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
                output_attentions: bool = False, use_cache: bool = False, **kwargs):
        base_model_results = self.layer(hidden_states)
        lora_results = self.mlp_lora_in(hidden_states)
        lora_results = self.mlp_lora_out(lora_results)
        layer_and_adaptor_sum = base_model_results + lora_results
        return layer_and_adaptor_sum

class LoraHeadAdaptor(nn.Module):
    def __init__(self, layer, r_value):
        super().__init__()
        self.layer = layer
        self.hidden_size = layer.weight.shape
        self.mlp_lora_in = nn.Linear(self.hidden_size[1], r_value, bias=False)
        self.mlp_lora_out = nn.Linear(r_value, self.hidden_size[0], bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        self.mlp_lora_out.weight.data.zero_()

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None, past_key_value: Optional[Cache] = None,
                output_attentions: bool = False, use_cache: bool = False, **kwargs):
        results = self.layer(hidden_states)
        lora_results = self.mlp_lora_in(hidden_states)
        lora_results = self.mlp_lora_out(lora_results)
        return results + lora_results

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

def get_requires_output_attentions(layer):
    return type(layer).__name__.find("SpdaAttention") != -1