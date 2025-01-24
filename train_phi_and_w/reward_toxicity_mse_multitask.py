import os
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from SFGen.utils import process_prompt
from SFGen.lm_with_feature_head import CausalLMWithFeatureHead


TOXICITY_TYPE = [
    "TOXICITY",
    "IDENTITY_ATTACK",
    "THREAT",
    "SEVERE_TOXICITY",
    "FLIRTATION",
    "PROFANITY",
    "INSULT",
    "SEXUALLY_EXPLICIT",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_path",
        type=str,
        default="realtoxicityprompts-new/rescored/generations/prompted/prompted_gens_gpt2.jsonl",
        help="Path of the training prompts.",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name in the saveing path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt2",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
    )
    args = parser.parse_args()
    return args


def get_rewrad_vector(sample):
    reward_vec = []
    for tp in TOXICITY_TYPE:
        assert tp in sample.keys(), f"{tp}: {sample.keys()}"
        if sample[tp] is None:
            raise Exception(f"Score for '{tp}' is None!")
        if sample[tp] >= 0.5:
            reward_vec.append(1.0)
        else:
            reward_vec.append(0.0)
    assert len(reward_vec) == len(TOXICITY_TYPE)
    return reward_vec


def get_features(text, model, tokenizer):
    """
    For a given text, return last layer's hidden state of the language model.
    
    Returns:
        (Tensor): [batch_size, hidden_size]
    """
    encoded_input = tokenizer(text, padding='longest', add_special_tokens=False, return_tensors="pt")
    encoded_input = encoded_input.to("cuda")

    # output: [bsz, seq_len, hidden_size]
    output = model(**encoded_input, return_dict=True, output_hidden_states=True)
    return output[:, -1, :]


class Reward(nn.Module):
    def __init__(
        self,
        feature_size=768,
        n_task=8,
        device=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.w = nn.Linear(feature_size, n_task, bias=False)
        self.w.to(device)

    def forward(self, features):
        """r = phi @ w

        Args:
            features (Tensor): [batch_size, feature_size]
 
        Returns:
            (Tensor): [batch_size, n_task]
        """
        assert self.feature_size == features.shape[1], "Hidden size doesn't match!"
        return self.w(features)


def train(model, reward_model, tokenizer, training_set, optimizer, criterion, batch_size=32):
    model.train()
    reward_model.train()

    batch, epoch_loss = [], []
    for sample in tqdm(training_set):
        batch.append(sample)

        if len(batch) == batch_size:        
            optimizer.zero_grad()

            text, rewards = [b[0] for b in batch], [b[1] for b in batch]
            features = get_features(text, model, tokenizer)

            predictions = reward_model(features)
            rewards = torch.tensor(rewards).to("cuda")

            loss = criterion(predictions, rewards)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            batch = []
    
    return sum(epoch_loss) / len(epoch_loss)


def evaluate(model, reward_model, tokenizer, eval_set, criterion, batch_size=256):
    model.eval()
    reward_model.eval()

    mse_errors, preds, rewards = [], [], []
    batch = []
    for sample in tqdm(eval_set):
        batch.append(sample)

        if len(batch) == batch_size:
            # batch_rewards: [bsz, n_task]
            batch_rewards = torch.tensor([b[1] for b in batch]).to("cuda")

            with torch.no_grad():
                features = get_features([b[0] for b in batch], model, tokenizer)
                batch_preds = reward_model(features)  # [bsz, n_task]
            
            loss = criterion(batch_preds, batch_rewards)

            preds.extend(batch_preds.tolist())
            rewards.extend(batch_rewards.tolist())
            mse_errors.append(loss.item())

            batch = []

    return mse_errors, preds, rewards


def main(args):
    # read dataset
    prompts = []
    with open(args.prompts_path, 'r') as f:
        prompts = [json.loads(line) for line in f]
    print(f"- Total training prompts loaded: {len(prompts)}")

    # build training & evaluation set
    print("- Bulding training & evaluation dataset...")
    training_set, test_set = [], []

    clean_counter, except_counter = 0, 0
    reward_type_count = np.zeros(len(TOXICITY_TYPE))

    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]["text"]
        for gen in p["generations"]:
            try:
                reward_vector = get_rewrad_vector(gen)

                # filter out low toxicity data
                if sum(reward_vector) < 1.0:
                    if random.random() < 1.0:
                        clean_counter += 1
                        continue

                reward_type_count += np.asarray(reward_vector)
                text_and_score = (process_prompt(prompt_text + gen['text']), reward_vector)

                if random.random() < 0.1:
                    test_set.append(text_and_score)
                else:
                    training_set.append(text_and_score)

            except Exception as e:
                except_counter += 1

    print(f"- Total clean samples removed: {clean_counter}")
    print(f"- Total None samples removed: {except_counter}")

    for i, tp in enumerate(TOXICITY_TYPE):
        print("- {}: {:.4f}".format(tp.lower(), reward_type_count[i] / (len(training_set) + len(test_set))))

    print(f"- Training set size: {len(training_set)}")
    print(f"- Test set size: {len(test_set)}")

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = CausalLMWithFeatureHead.from_pretrained(
        args.model_name_or_path,
        config=config,
        feature_size=args.feature_size
    )
    model.to("cuda")
    print(f'- {args.model_name_or_path} loaded!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_model = Reward(feature_size=args.feature_size, device=device)
    criterion = nn.MSELoss(reduction='mean')
    criterion = criterion.to(device)
    optimizer = optim.AdamW([
        {"params": reward_model.parameters()},
        {"params": model.parameters()}
    ])

    for e in range(args.num_epoch):
        print("- Epoch: {:}".format(e))
        print("- Training...")
        avg_loss = train(
            model,
            reward_model,
            tokenizer,
            training_set,
            optimizer,
            criterion,
            batch_size=args.train_batch_size
        )
        print("- Training loss: {:.5f}".format(avg_loss))
        print("- Evaluating...")
        mse_errors, preds, rewards = evaluate(
            model,
            reward_model,
            tokenizer,
            test_set,
            criterion,
            batch_size=args.eval_batch_size
        )
        print("- Evaluation loss: {:.5f}".format(sum(mse_errors) / len(mse_errors)))

        assert len(preds) == len(rewards)

        print("- Evaluation accuracy:")
        accuracy_dict = {k:0 for k in TOXICITY_TYPE}
        for ls, ps in zip(rewards, preds):
            for i in range(len(TOXICITY_TYPE)):
                true_l = 1 if ls[i] > 0.5 else 0
                pred_l = 1 if ps[i] > 0.5 else 0
                if pred_l == true_l:
                    accuracy_dict[TOXICITY_TYPE[i]] += 1

        for tp in accuracy_dict.keys():
            print(f"{tp}: {accuracy_dict[tp] / len(rewards)}")

        # save checkpoints
        save_path = f"save/{args.model_name}-with-feature{args.feature_size}-head-mse-multitask-skip-clean/epoch_{e}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        torch.save(
            reward_model.state_dict(), f"{save_path}/reward_model.pt"
        )
        print(f"- Checkpoint saved at: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
