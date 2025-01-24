# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonteCarlo(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        loss_func="mse",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_func = loss_func

    def _mask_loss(self, loss, actions):
        assert loss.shape == actions.shape
        pad_mask = actions.eq(self.ignore_index)
        loss.masked_fill_(pad_mask, 0.0)

    def forward(self, psi, target_psi, actions, features, seq_lens, reduce=True):
        """
        Args:
            psi (Tensor): [bsz, tgt_len - 1, vocab_size, feature_size]
            target_psi (Tensor): [bsz, tgt_len - 1, vocab_size, feature_size]
            actions (Tensor): [bsz, tgt_len - 1]
            features (Tensor): [bsz, tgt_len, hidden_size]
            seq_lens (Tensor): [bsz]

        """
        phis = features[:, 1:, :]
        psi = psi[:, :-1, :, :].contiguous()

        bsz = psi.shape[0]
        indices = actions.unsqueeze(-1).unsqueeze(-1)
        indices = indices.expand(bsz, psi.shape[1], 1, psi.shape[-1])

        psi = psi.gather(dim=2, index=indices).squeeze(2)  # [bsz, tgt_len - 1, feature_size]

        phi_terminal = phis[torch.arange(bsz), seq_lens - 1, :]
        phi_terminal_expand = phi_terminal.unsqueeze(1).expand(psi.shape)
        assert psi.shape == phi_terminal_expand.shape, f"{psi.shape} - {phi_terminal_expand.shape}"

        if self.loss_func == "mse":
            loss = F.mse_loss(psi, phi_terminal_expand.detach(), reduction="none").sum(dim=-1)
        elif self.loss_func == "cos":
            cosine_sim = nn.CosineSimilarity(dim=2, eps=1e-6)
            loss = 1 - cosine_sim(psi, phi_terminal_expand.detach())
        else:
            raise ValueError(f"Unknown loss type: {self.loss_func}")

        self._mask_loss(loss, actions)

        if reduce:
            loss = loss.sum()

        return loss