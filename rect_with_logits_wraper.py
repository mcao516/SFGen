# import os
# import json
# import torch
# import transformers

# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# from generation import CustomizedGenerationMixin

# SCRATCH = os.environ['SCRATCH']
# SEED = 0

# torch.manual_seed(SEED)
# transformers.set_seed(SEED)


# # Read language model
# checkpoint = "gpt2"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForCausalLM.from_pretrained(checkpoint)

# # read prompts
# prompts_path = os.path.join(SCRATCH, 'successor-features/dataset/toxic_prompts-10k_test_uppercase.jsonl')
# data = []
# with open(prompts_path, 'r') as f:
#     for line in f:
#         data.append(json.loads(line)['prompt']['text'])

# # generation
# for prompt in data[:10]:
#     inputs = tokenizer(prompt, return_tensors="pt")

#     outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, pad_token_id=tokenizer.eos_token_id)
#     print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

#     generator = CustomizedGenerationMixin(model)
#     outputs = generator.generate(**inputs, max_new_tokens=20, do_sample=True, pad_token_id=tokenizer.eos_token_id)
#     print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
#     print()

# ==================================================================================================
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    LogitsWarper,
    LogitsProcessor,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.utils import ModelOutput
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from reward_model import Reward, RewardMultiTaskInference
from lm_with_value_heads import CausalLMWithValueHeadsInference

SCRATCH = os.environ['SCRATCH']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= READ REWARD MODEL ==========================================
FEATURE_SIZE = 64

# rewrad_model_dir = os.path.join(
#     SCRATCH,
#     "successor-features/save/lm-with-feature64-head-mse/reward_model.pt"
# )

rewrad_model_dir = os.path.join(
    SCRATCH,
    "successor-features/save/lm-with-feature64-head-mse-multitask-skip-clean/reward_model.pt"
)

TOXICITY_TYPE_MAP = {
    "TOXICITY": 0,
    "IDENTITY_ATTACK": 1,
    "THREAT": 2,
    "SEVERE_TOXICITY": 3,
    "FLIRTATION": 4,
    "PROFANITY": 5,
    "INSULT": 6,
    "SEXUALLY_EXPLICIT": 7,
}

# reward_model = Reward(feature_size=FEATURE_SIZE).to(device)
reward_model = RewardMultiTaskInference(
    feature_size=FEATURE_SIZE,
    n_task=8,
    current_task=[TOXICITY_TYPE_MAP["TOXICITY"]],
    device=device
)
reward_model.load_state_dict(torch.load(rewrad_model_dir))
# ===============================================================================================

# ==================================== READ SECURITY MODEL ==============================================
model_dir = os.path.join(
    SCRATCH,
    "DeD_models/DeBias/successor_feature_epoch-3_plr-0.1_biploar2_mse_multi_feature64_skipclean_gpt2/"
)

security_model = CausalLMWithValueHeadsInference.from_pretrained(model_dir, feature_size=FEATURE_SIZE, E=32)
security_model.set_reward_model(reward_model)
security_model.to(device)
security_model.eval()

print('- Security model loaded!')
# ==================================== GREEDY SEARCH ====================================

class SecurityLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for LM rectification. See [Systematic Rectification of Language
    Models via Dead-end Analysis](https://arxiv.org/abs/2302.14003) for more information.

    Args:
        eps (`float`, *optional*, defaults to 0):
            The threshold removes any action with a value of 1 + Q(s, a) below it.
        p_min (`float`, *optional*, defaults to 0):
            The minimum value that used to clamp 1 + Q(s, a).
        p_max (`float`, *optional*, defaults to 1):
            The maximum value that used to clamp 1 + Q(s, a).
    """

    def __init__(
        self,
        rectlm,
        eps: float = 0.0,
        p_min: float = 0.0,
        p_max: float = 1.0,
    ):
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max
        self.rectlm = rectlm

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
                search or log softmax for each vocabulary token when using beam search

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
        """
        security_scores = self.rectlm(input_ids).logits[:, -1, :]
        lm_probs = nn.functional.softmax(scores, dim=-1)

        security_scores = torch.clamp(security_scores, max=0.0, min=-1.0)
        secure_probs = (1 + security_scores - self.eps) / (1 - self.eps)
        secure_probs = torch.clamp(secure_probs, max=self.p_max, min=self.p_min)

        for _ in range(3):
            mask = lm_probs > secure_probs
            lm_probs[mask] = secure_probs[mask]

            # re-normalize
            lm_probs = lm_probs / lm_probs.sum(dim=1, keepdim=True)

        return lm_probs


# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

# input_prompt = """The male fled the scene on a bicycle, but not before he shouted "I'll"""
# input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
# input_ids = input_ids.to(device)

# # instantiate logits processors
# logits_processor = LogitsProcessorList(
#     [
#         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
#         SecurityLogitsProcessor(security_model, eps=0.55, p_min=0.0, p_max=1.0),
#     ]
# )

# stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=40)])

# outputs = model.greedy_search(
#     input_ids,
#     logits_processor=logits_processor,
#     stopping_criteria=stopping_criteria,
# )

# output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(output)

# ==================================== SAMPLE ====================================

class SecurityTopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] for LM rectification. See [Systematic Rectification of Language
    Models via Dead-end Analysis](https://arxiv.org/abs/2302.14003) for more information.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        eps (`float`, *optional*, defaults to 0):
            The threshold removes any action with a value of 1 + Q(s, a) below it.
        p_min (`float`, *optional*, defaults to 0):
            The minimum value that used to clamp 1 + Q(s, a).
        p_max (`float`, *optional*, defaults to 1):
            The maximum value that used to clamp 1 + Q(s, a).
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: int,
        rectlm,
        filter_value: float = -float("Inf"),
        eps: float = 0.0,
        p_min: float = 0.0,
        p_max: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max
        self.rectlm = rectlm

        self.security_kwargs = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
                search or log softmax for each vocabulary token when using beam search

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.
        """
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        lm_probs = nn.functional.softmax(scores, dim=-1)

        self.rectlm.prepare_inputs_for_generation(input_ids, **self.security_kwargs)
        security_outputs = self.rectlm(input_ids)
        security_scores = security_outputs.logits[:, -1, :]
        self.security_kwargs = self._update_model_kwargs_for_generation(
            security_outputs, self.security_kwargs
        )

        security_scores = torch.clamp(security_scores, max=0.0, min=-1.0)
        secure_probs = (1 + security_scores - self.eps) / (1 - self.eps)
        secure_probs = torch.clamp(secure_probs, max=self.p_max, min=self.p_min)

        for _ in range(3):
            mask = lm_probs > secure_probs
            lm_probs[mask] = secure_probs[mask]

            # re-normalize
            lm_probs = lm_probs / lm_probs.sum(dim=1, keepdim=True)

        scores = torch.log(lm_probs)
        return scores

    @staticmethod
    def _update_model_kwargs_for_generation(
        outputs: ModelOutput, model_kwargs: Dict[str, Any], is_encoder_decoder: bool = False
    ) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
model.to(device)

# set pad_token_id to eos_token_id because GPT2 does not have a EOS token
model.config.pad_token_id = model.config.eos_token_id
model.generation_config.pad_token_id = model.config.eos_token_id

input_prompt =  """The male fled the scene on a bicycle, but not before he shouted "I'll"""
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(device)

# instantiate logits processors
logits_processor = LogitsProcessorList(
    [
        MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
    ]
)
# instantiate logits processors
# TopKLogitsWarper(50),
logits_warper = LogitsProcessorList(
    [
        SecurityTopKLogitsWarper(50, security_model, eps=0.55, p_min=1e-5, p_max=1.0),
    ]
)

stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=50)])

print("- Generating...")
torch.manual_seed(1)
for _ in tqdm(range(1)):
    outputs = model.sample(
        input_ids,
        logits_processor=logits_processor,
        logits_warper=logits_warper,
        stopping_criteria=stopping_criteria,
    )

output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(output)