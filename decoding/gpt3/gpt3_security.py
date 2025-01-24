# coding=utf-8

import os
import json
import torch
import openai
import argparse

from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

openai.api_key = os.getenv("OPENAI_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpt3_generate(
    prompt,
    max_tokens=20,
    num_lprobs=1,
    num_returns=10,
    top_p=0.9
):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            suffix=None,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=top_p,
            logprobs=num_lprobs,
            n=num_returns
        )
    except Exception as err:
        return None

    return [r["text"] for r in response["choices"]]


def gpt3_one_step(context, num_lprobs=100, sort_lprobs=True):
    """
    returns:
        - token:  it
        - lprobs: {
            ' that': -1.3344219,
            ' it': -1.4906405,
            ' being': -1.521714,
            ' my': -1.8878764,
            ' who': -2.544157,
            ' what': -3.785404,
            ' myself': -3.8253057,
            ' the': -5.473457,
            ' where': -6.029905,
            ' me': -6.3665986,
        }
    """
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=context,
        suffix=None,
        max_tokens=1,
        temperature=1.0,
        logprobs=num_lprobs
    )
    token = response["choices"][0]["text"]
    lprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]
    if sort_lprobs:
        lprobs = {k: v for k, v in sorted(lprobs.items(), key=lambda item: item[1], reverse=True)}

    return token, lprobs


def gpt3_generation_with_security(
    prompt_text,
    model=None,
    tokenizer=None,
    max_steps=20,
    num_lprobs=100,
    apply_security=True,
    num_returns=1,
    threshold=0.0,
    secure_prob_min=0.0
):
    step_count = 0
    
    continuation = ""
    while True:
        token, lprobs = gpt3_one_step(prompt_text, num_lprobs=num_lprobs, sort_lprobs=False)
        
        if apply_security:
            assert model is not None and tokenizer is not None
            input_ids = tokenizer.encode(prompt_text, return_tensors='pt').cuda()
            security_outputs = model(
                input_ids
            )
            Q_dead_end = security_outputs.logits[:, -1, :]
            secure_probs = (1 + Q_dead_end - threshold) / (1 - threshold)
            secure_probs = torch.clamp(secure_probs, max=1.0, min=secure_prob_min)

            # convert lprobs to tensor
            probs = torch.zeros_like(secure_probs)
            for k, v in lprobs.items():
                probs[0][tokenizer.encode(k)[0]] = v
            probs = probs / probs.sum(dim=1, keepdim=True)

            for _ in range(3):
                mask = probs > secure_probs
                probs[mask] = secure_probs[mask]
                probs = probs / probs.sum(dim=1, keepdim=True)
            
            next_token_id = torch.argmax(probs, dim=-1)
            next_token_str = tokenizer.decode(next_token_id)
        else:
            next_token_str = token
        
        prompt_text += next_token_str
        continuation += next_token_str

        step_count += 1
        if step_count == max_steps:
            break

    return [continuation]


def process_prompt(text):
    text = text.replace('’', "'")
    text = text.replace('‘', "'")
    text = text.replace('“', '"')
    text = text.replace('”', '"')
    return text


def main(args):
    # load prompts
    with open(args.prompts_file, 'r') as f:
        prompts = [json.loads(line) for line in f]

    # load security model
    tokenizer = GPT2Tokenizer.from_pretrained(args.security_model_dir)
    security_model = GPT2LMHeadModel.from_pretrained(args.security_model_dir)
    security_model.to(device)
    security_model.eval()

    # generate
    inputs = [process_prompt(s['prompt']['text']) for s in prompts]

    generations = []
    for input_text in tqdm(inputs):
        output = gpt3_generation_with_security(
            input_text,
            model=security_model,
            tokenizer=tokenizer,
            max_steps=20,
            num_lprobs=100,
            apply_security=True,
            num_returns=1,
            threshold=args.secure_threshold,
            secure_prob_min=args.secure_prob_min,
        )
        generations.append(output)

    assert len(generations) == len(inputs)

    # add generations
    for gen, pm in zip(generations, prompts):
        pm["generations"] = [{"text": g} for g in gen]

    print("Generations saved at: ", args.save_file)
    with open(args.save_file, 'w') as wf:
        for item in prompts:
            wf.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file",
        type=str,
        default='realtoxicityprompts-new/rescored/prompts.jsonl'
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default='realtoxicityprompts-new/rescored/prompts_test.jsonl'
    )
    parser.add_argument(
        "--num_returns",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--secure_threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--secure_prob_min",
        type=float,
        default=1e-8,
    )
    args = parser.parse_args()
    main(args)