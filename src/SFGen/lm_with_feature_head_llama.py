# coding=utf-8

from copy import deepcopy
import torch
from torch import nn
from transformers import LlamaForCausalLM, GPT2LMHeadModel


def make_head(n_embd: int, out: int):
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(n_embd, out),
    )


class CausalLMWithFeatureHead(LlamaForCausalLM):
    """This is a wrapper around huggingface AutoModelForCausalLM with feature head"""

    def __init__(self, config, feature_size):
        super().__init__(config)

    # def build_feature_head(self, feature_size):
        self.feature_size = feature_size
        # self.feature_head = make_head(self.vocab_size, self.feature_size)
        self.feature_head = make_head(self.config.hidden_size, self.feature_size)

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
        assert self.feature_head is not None
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kargs
        )
        hidden_states = outputs[0]
        # last_hidden_state: [bsz, max_len, hidden_size]
        return self.feature_head(hidden_states)

    @property
    def dummy_inputs(self):
        return {"input_ids": torch.ones(1, 1, device=self.gpt.device, dtype=torch.long)}

    @property
    def device(self):
        return self.gpt.device