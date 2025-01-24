# coding=utf-8

from copy import deepcopy
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers import LlamaForCausalLM, GPT2LMHeadModel, LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, BaseModelOutputWithPast, CausalLMOutputWithPast


def make_head(n_embd: int, out: int, E: int):
    return nn.Sequential(
        nn.GELU(),
        nn.Linear(n_embd, E),
        nn.GELU(),
        nn.Linear(E, out),
    )


class ILQLHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        feature_size: int,
        n_qs: int = 1,
        alpha: float = 0.1,
        E: int = 16,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.n_qs = n_qs
        self.alpha = alpha
        self.E = E

        output_size = self.vocab_size * self.feature_size
        self.q_heads = nn.ModuleList(
            make_head(self.hidden_size, output_size, self.E) for _ in range(n_qs)
        )
        self.target_q_heads = nn.ModuleList(deepcopy(q_head) for q_head in self.q_heads)

        for q_head in self.target_q_heads:
            q_head.requires_grad_(False)

    def forward(
        self,
        hs: torch.Tensor,
    ):
        """
        Args:
            hs (torch.Tensor): [bsz, seq_len, hidden_size]

        Returns:
            qs (torch.Tensor): [bsz, seq_len, vocab_size, feature_size]
            target_qs (torch.Tensor): [bsz, seq_len, vocab_size, feature_size]
        """
        bsz, seq_len, _ = hs.shape

        qs = tuple(q_head(hs) for q_head in self.q_heads)
        target_qs = tuple(q_head(hs) for q_head in self.target_q_heads)

        assert qs[0].shape[-1] == self.vocab_size * self.feature_size
        qs = tuple(q.view(bsz, seq_len, self.vocab_size, self.feature_size) for q in qs)
        target_qs = tuple(q.view(bsz, seq_len, self.vocab_size, self.feature_size) for q in target_qs)

        return qs[0], target_qs[0]

    def _sync_target_q_heads(self, alpha):
        for target_q_head, q_head in zip(self.target_q_heads, self.q_heads):
            for target_param, copy_param in zip(
                target_q_head.parameters(), q_head.parameters()
            ):
                target_param.data.copy_(
                    (alpha * copy_param.data) + (1.0 - alpha) * target_param.data
                )

    def sync_target_q_heads(self):
        self._sync_target_q_heads(self.alpha)

    def copy_q_heads_to_target_q_heads(self):
        self._sync_target_q_heads(1.0)


class CausalLMWithValueHeads(LlamaModel):
    """This is a wrapper around huggingface AutoModelForCausalLM with two additional scalar heads"""

    def __init__(self, config, feature_size, E=16, alpha=0.1):
        super().__init__(config)

        self.alpha = alpha
        self.feature_size = feature_size
        # self.n_embd = self.transformer.config.n_embd

        # vocab_size = self.transformer.config.vocab_size
        self.ilql_heads = ILQLHeads(
            self.config.hidden_size, self.vocab_size, self.feature_size, alpha=self.alpha, E=E,
        )

    def sync_target_q_heads(self):
        self.ilql_heads.sync_target_q_heads()

    def copy_q_heads_to_target_q_heads(self):
        self.ilql_heads.copy_q_heads_to_target_q_heads()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        return_outputs=False,
    ):
        """
        Returns:
            qs (Tensor): [bsz, tgt_len, vocab_size, feature_size]
            target_qs (Tensor): [bsz, tgt_len, vocab_size, feature_size]

        """
        outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # last_hidden_state: [bsz, max_len, hidden_size]
        # qs, target_qs = self.ilql_heads(out.last_hidden_state)
        qs, target_qs = self.ilql_heads(outputs.last_hidden_state)

        if return_outputs:
            return qs, target_qs, outputs
        else:
            return qs, target_qs


# class CausalLMWithValueHeadsInference(CausalLMWithValueHeads):
#     """This is a wrapper around CausalLMWithValueHeadsInference for inference"""

#     def set_reward_model(self, reward_model):
#         self.reward_model = reward_model

class CausalLMWithValueHeadsInference(CausalLMWithValueHeads):
    _tied_weights_keys = ["lm_head.weight"]

    # def __init__(self, config, feature_size, E=16, alpha=0.1):
    #     super().__init__(config, feature_size, E=16, alpha=0.1)
    #     self.model = LlamaModel(config)
    #     self.vocab_size = config.vocab_size
    #     # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    #     self.llm_head = None

    #     # Initialize weights and apply final processing
    #     print("test")
    #     self.post_init()

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    # def get_input_embeddings(self):
    #     return self.model.embed_tokens

    # def set_input_embeddings(self, value):
    #     self.model.embed_tokens = value

    # def get_output_embeddings(self):
    #     return self.lm_head

    # def set_output_embeddings(self, new_embeddings):
    #     self.lm_head = new_embeddings

    # def set_decoder(self, decoder):
    #     self.model = decoder

    # def get_decoder(self):
    #     return self.model

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
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        # hidden_states = outputs[0]

        logits, _, outputs = super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_outputs=True,
        )

        if self.config.pretraining_tp > 1:
            # lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            # logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            # logits = torch.cat(logits, dim=-1)
            raise NotImplementedError
        else:
            # logits, _ = self.ilql_heads(hidden_states)
            if type(logits) == tuple:
                logits = logits[0]

            if self.reward_model is not None:
                assert logits.dim() == 4, f"{logits.dim()}"
                logits = -self.reward_model(logits).squeeze(dim=-1)
            else:
                raise AttributeError

        # logits = logits.float()

        loss = None
        if labels is not None:
            raise NotImplementedError

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

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

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
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