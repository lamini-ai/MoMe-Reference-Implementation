import torch
import torch.nn as nn
from transformers import LlamaPreTrainedModel, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_raft import RaftConfig
from typing import Optional, List, Union, Tuple
from .lamini_index import LaminiIndex

class RaftAttentionLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.index = index
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.key_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.value_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.shape
        query = self.query_proj(hidden_states)

        # Retrieve from index
        retrieved_embeddings = self.retrieve_from_index(query)

        # Project retrieved embeddings
        key = self.key_proj(retrieved_embeddings)
        value = self.value_proj(retrieved_embeddings)

        # Reshape for attention
        query = query.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, value)

        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        output = self.output_proj(context)

        return output

    def retrieve_from_index(self, query):
        # Flatten query for retrieval
        flat_query = query.view(-1, self.hidden_size).detach().cpu().numpy()
        
        # Retrieve from index
        retrieved_keys, retrieved_values, _ = self.index.get_key_and_value(flat_query, k=1)
        
        # Convert back to tensor and reshape
        retrieved_embeddings = torch.tensor(retrieved_values, device=query.device).view_as(query)
        
        return retrieved_embeddings

class RaftModel(LlamaModel):
    config_class = RaftConfig

    def __init__(self, config):
        super().__init__(config)
        self.index = LaminiIndex.load_index(config.index_path)
        self.raft_layers = nn.ModuleList([RaftAttentionLayer(config, self.index) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = super().forward(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]

        for raft_layer in self.raft_layers:
            raft_output = raft_layer(hidden_states)
            hidden_states = hidden_states + raft_output

        outputs = (hidden_states,) + outputs[1:]
        return outputs

class RaftForCausalLM(LlamaForCausalLM):
    config_class = RaftConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = RaftModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids,
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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

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

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        return super().prepare_inputs_for_generation(input_ids, past=past, **kwargs)

# Register the model with Auto classes
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("raft", RaftConfig)
AutoModel.register(RaftConfig, RaftModel)
AutoModelForCausalLM.register(RaftConfig, RaftForCausalLM)