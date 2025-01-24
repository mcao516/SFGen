# coding=utf-8

from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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


class CausalLMWithValueHeads(GPT2LMHeadModel):
    """This is a wrapper around huggingface AutoModelForCausalLM with two additional scalar heads"""

    def __init__(self, config, feature_size, E=16, alpha=0.1):
        super().__init__(config)

        self.alpha = alpha
        self.feature_size = feature_size
        self.n_embd = self.transformer.config.n_embd

        vocab_size = self.transformer.config.vocab_size
        self.ilql_heads = ILQLHeads(
            self.n_embd, vocab_size, self.feature_size, alpha=self.alpha, E=E,
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
    ):
        """
        Returns:
            qs (Tensor): [bsz, tgt_len, vocab_size, feature_size]
            target_qs (Tensor): [bsz, tgt_len, vocab_size, feature_size]

        """
        out = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        # last_hidden_state: [bsz, max_len, hidden_size]
        qs, target_qs = self.ilql_heads(out.last_hidden_state)

        return qs, target_qs


class CausalLMWithValueHeadsInference(CausalLMWithValueHeads):
    """This is a wrapper around CausalLMWithValueHeadsInference for inference"""

    def set_reward_model(self, reward_model):
        self.reward_model = reward_model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        with torch.no_grad():
            lm_logits, _ = self.ilql_heads(hidden_states)
            if type(lm_logits) == tuple:
                lm_logits = lm_logits[0]

            if self.reward_model is not None:
                assert lm_logits.dim() == 4, f"{lm_logits.dim()}"
                lm_logits = -self.reward_model(lm_logits).squeeze(dim=-1)

            # lm_logits_for_test = self.lm_head(hidden_states)
            # assert lm_logits.shape == lm_logits_for_test.shape, \
            #     "{} - {}".format(lm_logits.shape, lm_logits_for_test.shape)

        loss = None
        if labels is not None:
            raise NotImplementedError

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )