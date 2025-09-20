#!/usr/bin/env python
# -*- coding: utf-8

import os
import json
import time
import uuid
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_fp', type=str, required=True,
        help='Path to the batch JSON file; each record must contain body.messages[0].content'
    )
    parser.add_argument(
        '--save_fp', type=str, required=True,
        help='Path where the final scored results will be saved'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=2000,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--model', type=str, default="Qwen/Qwen2.5-72B-Instruct",
        help='Model used to evaluate as a judge'
    )
    parser.add_argument(
        '--dataset', type=str, default="cosmos",
        help='The dataset to judge'
    )

    args = parser.parse_args()

    # Load batch data
    data = []
    with open(args.batch_fp, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    dataset = args.dataset
    seeds = {
        "cosmos": 1,
        "drop": 18,
        "esnli": 9,
        "gsm8k": 30,
    }

    seed = seeds[dataset]

    train_data, test_data = train_test_split(
        data,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
    )

    # Initialize model and tokenizer
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME", None),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME", None),
    )
    model.eval()

    all_results = []
    for idx, item in enumerate(tqdm(test_data, desc="Scoring")):
        # Use the full content from the original messages as prompt
        prompt = item["body"]["messages"][0]["content"]

        # Tokenize and move to device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[-1]

        # Run generation and collect scores
        with torch.no_grad():
            generation = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=20 
            )

        generated_texts = []
        for i in range(generation.sequences.shape[0]):
            seq_ids = generation.sequences[i]
            generated_ids = seq_ids[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)

        item_output = dict(item)
        item_output["judge_responses"] = generated_texts
        all_results.append(item_output)
    
    # Write out final JSON
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nScoring complete! Results saved to {args.save_fp}")



if __name__ == "__main__":
    main()
