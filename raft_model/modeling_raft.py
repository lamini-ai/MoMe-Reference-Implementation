import numpy as np
from transformers import PretrainedConfig
from raft_model.configuration_raft import LlamaRaftConfig
from raft_model.lamini_index import LaminiIndex

import torch
import torch.nn as nn
import copy
import logging
from typing import Optional, List, Dict, Union, Tuple
from transformers.cache_utils import Cache
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def initialize(self, model):
        cloned_model = clone_module(model)
        self.embeddings = {
            "key_embeddings": [],
            "value_embeddings": [],
            "embedding_indices": [],
        }
        self.index = LaminiIndex.load_index("/home/llama/MoME-Reference-Implementation/nithya/checkpoints/index")
        self.raft_model = add_raft_adaptors_to_each_layer(
            cloned_model, self.embeddings, self.index
        )

    def get_input_embeddings(self):
        return self.raft_model.embed_tokens

    def set_input_embeddings(self, value):
        self.raft_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        outputs = self.raft_model(
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

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


def clone_module(module):
    """Make a shallow copy of a module recursively."""
    shallow_copy = copy.copy(module)
    shallow_copy._parameters = shallow_copy._parameters.copy()
    shallow_copy._buffers = shallow_copy._buffers.copy()
    shallow_copy._modules = shallow_copy._modules.copy()

    for child_name, child in shallow_copy.named_children():
        shallow_copy._modules[child_name] = clone_module(child)
    return shallow_copy


def add_raft_adaptors_to_each_layer(model, embeddings, index):
    """The RaftAdaptor wraps and replaces layers if they are attention layers."""
    for name, layer in model.named_modules():
        logger.info(f"Checking layer {name}, type: {type(layer)}")
        try_to_update_self_attn(name, layer, model, embeddings, index)
    return model


def try_to_update_self_attn(name, layer, model, embeddings, index):
    """Try to wrap the layer with a RaftAdaptor."""
    if not is_self_attn_layer(name):
        return

    logger.info(f"Wrapping layer {name} with RaftAdaptor")

    # Wrap the layer with a RaftAdaptor
    recursive_setattr(model, name, RaftAdaptor(layer, embeddings, index))


def is_self_attn_layer(name):
    """Check if it is a huggerface attention layer."""
    name_suffix = name.split(".")[-1]

    # huggingface calls it self_attn
    if name_suffix == "self_attn":
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
            device=layer.q_proj.weight.device,
            embeddings=embeddings,
            index=index,
        )

        # LayerNorm
        self.layer_norm = nn.LayerNorm(layer.hidden_size, eps=1e-12)

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
        self_attention_output, self_attention_weights, present_key_value = self.layer(
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

        # sum the two attentions
        return (
            self_attention_output + self.layer_norm(raft_attention_output),
            self_attention_weights,
            present_key_value,
        )


raft_embedding_batch_size = 1
raft_embedding_seq_length = 512
attention_head_count = 8


class RaftAttentionLayer(nn.Module):
    def __init__(self, hidden_size, device, embeddings, index):
        super().__init__()

        self.index = index

        self.embeddings = embeddings

        self.key_embedding = nn.Parameter(
            torch.zeros(
                raft_embedding_batch_size, raft_embedding_seq_length, hidden_size
            )
        )

        self.value_embedding = nn.Parameter(
            torch.zeros(
                raft_embedding_batch_size, raft_embedding_seq_length, hidden_size
            )
        )

        self.embedding_index = len(embeddings["key_embeddings"])

        self.embeddings["key_embeddings"].append(self.key_embedding)
        self.embeddings["value_embeddings"].append(self.value_embedding)
        self.embeddings["embedding_indices"].append([])

        # A linear layer to project the query into the space of the index
        self.query_projection = nn.Linear(hidden_size, 384)

        # Add a standard self attention layer for raft
        self.attn = nn.MultiheadAttention(hidden_size, attention_head_count)

        # Add a linear layer to project the raft attention output to the same size as the transformer attention output
        self.raft_attention_projection = nn.Linear(hidden_size, hidden_size)

        # initialize the projection layer to zeros
        self.raft_attention_projection.weight.data.zero_()

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
        query = self.get_query(hidden_states)
        key, value = self.get_key_and_value(query)

        # project the raft attention output to the same size as the transformer attention output
        raft_attention_output = self.attn(query=query, key=key, value=value)[0]

        raft_attention_output = self.raft_attention_projection(raft_attention_output)

        return raft_attention_output

    def get_query(self, hidden_states):
        # get the query from the hidden states
        # hidden states dimensions are [seq_len, batch_size, hidden_size]
        return self.query_projection(hidden_states)

    def get_key_and_value(self, query):
        key, value, indices = self.get_key_and_value_from_index(query)

        self.embeddings["embedding_indices"][self.embedding_index] = indices

        # assign into the raft embedding space
        print(f"key embedding weight shape: {self.key_embedding.shape}")

        # Get the sequence length
        sequence_length = key.shape[1]

        with torch.no_grad():
            self.key_embedding[:, :sequence_length, :].copy_(key)
            self.value_embedding[:, :sequence_length, :].copy_(value)

        return (
            self.key_embedding[:, :sequence_length, :],
            self.value_embedding[:, :sequence_length, :],
        )

    def get_key_and_value_from_index(self, query):
        batch_size = query.shape[0]
        sequence_length = query.shape[1]
        embedding_dimension = query.shape[2]

        device = query.device

        query = query.view(batch_size * sequence_length, embedding_dimension)

        # get the key and value from the index, no gradients
        with torch.no_grad():
            # convert query to float32
            query = query.float()

            # convert query to a numpy array
            query = query.cpu().numpy()

            key, value, indices = self.index.get_key_and_value(query, k=1)

            # convert key and values, which are lists, to numpy arrays
            key = np.array(key)
            value = np.array(value)

            # convert key and value to torch tensors
            key = torch.from_numpy(key).to(device)
            value = torch.from_numpy(value).to(device)

            key = key.view(batch_size, sequence_length, embedding_dimension)
            value = value.view(batch_size, sequence_length, embedding_dimension)

        print(f"key shape: {key.shape}, value shape: {value.shape}")

        return key, value, indices
