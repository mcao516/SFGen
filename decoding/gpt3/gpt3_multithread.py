# coding=utf-8

import os
import json
import openai
import argparse

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from itertools import repeat

from SFGen.utils import process_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")


def gpt3_generate(
    prompt,
    max_tokens=20,
    num_lprobs=1,
    num_returns=10,
    top_p=0.9,
    engine="gpt-3.5-turbo"
):
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            suffix=None,
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=top_p,
            logprobs=num_lprobs,
            n=num_returns
        )
    except Exception as err:
        print(err)
        return None

    return [r["text"] for r in response["choices"]]


def main(args):
    with open(args.prompts_file, 'r') as f:
        prompts = [json.loads(line) for line in f]
    print(f"- read {len(prompts)} prompts")

    inputs = [process_prompt(s['prompt']['text'], args.prefix) for s in prompts]

    generations = []
    with Pool(10) as pool:
        for r in tqdm(pool.imap(partial(
            gpt3_generate,
            max_tokens=args.max_tokens,
            num_lprobs=1,
            num_returns=args.num_returns,
            top_p=args.top_p,
            engine=args.engine), inputs), total=len(inputs)
        ):
            generations.append(r)

        # for r in tqdm(pool.imap(gpt3_generate, inputs), total=len(inputs)):
        #     generations.append(r)

        # for r in tqdm(pool.starmap(gpt3_generate, 
        #     zip(inputs, 
        #         repeat(args.max_tokens),
        #         repeat(1),
        #         repeat(args.num_returns),
        #         repeat(args.top_p),
        #         repeat(args.engine))), total=len(inputs)):
        #     generations.append(r)

    for i in range(len(inputs)):
        if generations[i] is None:
            generations[i] = gpt3_generate(
                inputs[i],
                max_tokens=args.max_tokens,
                num_lprobs=1,
                num_returns=args.num_returns,
                top_p=args.top_p,
                engine=args.engine
            )
    assert len(generations) == len(prompts), f"{len(generations)} - {len(prompts)}"

    # save generations
    for gen, p in zip(generations, prompts):
        p["generations"] = [{"text": g} for g in gen]

    with open(args.save_file, 'w') as wf:
        for item in prompts:
            wf.write(json.dumps(item) + "\n")

    print("done. generations saved at: ", args.save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts_file",
        type=str,
        default='prompts.jsonl'
    )
    parser.add_argument(
        "--save_file",
        type=str,
        default='generations.jsonl'
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
        "--prefix",
        type=str,
        default=None
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo"
    )
    args = parser.parse_args()
    main(args)