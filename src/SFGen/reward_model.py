# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Reward(nn.Module):
    def __init__(
        self,
        feature_size=768,
        device=None
    ):
        super().__init__()
        self.feature_size = feature_size
        self.w = nn.Linear(feature_size, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if device:
            self.sigmoid.to(device)
            self.w.to(device)

    def calculate_reward(self, features):
        return self.w(features)

    def forward(self, features):
        """r = phi @ w

        Args:
            features (Tensor): [batch_size, feature_size]
 
        Returns:
            (Tensor): [batch_size, 1]
        """
        assert self.feature_size == features.shape[-1], f"{self.feature_size} : {features.shape[1]}"
        return self.sigmoid(self.w(features))


class RewardMultiTask(nn.Module):
    def __init__(
        self,
        feature_size=768,
        n_task=8,
        device=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.w = nn.Linear(feature_size, n_task, bias=False)
        if device:
            self.w.to(device)

    def forward(self, features):
        """r = phi @ w

        Args:
            features (Tensor): [batch_size, feature_size]
 
        Returns:
            (Tensor): [batch_size, n_task]
        """
        assert self.feature_size == features.shape[-1], f"{self.feature_size} : {features.shape[-1]}"
        return self.w(features)


class RewardMultiTaskInference(nn.Module):
    def __init__(
        self,
        feature_size=768,
        n_task=8,
        current_task=0,
        device=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.w = nn.Linear(feature_size, n_task, bias=False)
        self.n_task = n_task
        self.current_task = current_task

        if device:
            self.w.to(device)
    
    def set_current_task(self, task_id):
        assert 0 <= task_id < self.n_task, f"Task ID {task_id} is not valid."
        self.current_task = task_id

    def forward(self, features):
        """r = phi @ w

        Args:
            features (Tensor): [batch_size, seq_len, vocab_size, feature_size]

        Returns:
            (Tensor): [batch_size, seq_len, vocab_size, 1]
        """
        assert self.feature_size == features.shape[-1], f"{self.feature_size} : {features.shape[-1]}"
        if type(self.current_task) == int:
            return (1.0 - self.w(features))[..., self.current_task].unsqueeze(-1)
        elif type(self.current_task) == list:
            q_tasks = self.w(features)
            q = None
            for task_id in self.current_task:
                if q is None:
                    q = q_tasks[..., task_id]
                else:
                    # q += q_tasks[..., task_id]
                    q = torch.maximum(q, q_tasks[..., task_id])

            return q.unsqueeze(-1)
            # return q / len(self.current_task)
        else:
            raise Exception(f"Unknown self.current_task type: {self.current_task}")