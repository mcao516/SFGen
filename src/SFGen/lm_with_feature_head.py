# coding=utf-8

from copy import deepcopy
import torch
from torch import nn
from transformers import GPT2LMHeadModel


def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(n_embd, out),
    )


class CausalLMWithFeatureHead(GPT2LMHeadModel):
    """This is a wrapper around huggingface AutoModelForCausalLM with feature head"""

    def __init__(self, config, feature_size):
        super().__init__(config)

        self.feature_size = feature_size
        self.n_embd = self.transformer.config.n_embd

        self.feature_head = make_head(self.n_embd, self.feature_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kargs,
    ):
        """
        Returns:
            features (Tensor): [bsz, tgt_len, feature_size]
        """
        out = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kargs
        )
        # last_hidden_state: [bsz, max_len, hidden_size]
        return self.feature_head(out.last_hidden_state)

    @property
    def dummy_inputs(self):
        return {"input_ids": torch.ones(1, 1, device=self.gpt.device, dtype=torch.long)}

    @property
    def device(self):
        return self.gpt.device