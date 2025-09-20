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
        '--max_new_tokens', type=int, default=10,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of top token logprobs to record at each generation step'
    )
    parser.add_argument(
        '--model', type=str, default="Qwen/Qwen2.5-72B-Instruct",
        help='Model used to evaluate as a judge'
    )

    args = parser.parse_args()

    # Load batch data
    with open(args.batch_fp, 'r', encoding='utf-8') as f:
        data = json.load(f)

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
    for idx, item in enumerate(tqdm(data, desc="Scoring")):
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
                output_scores=True,
                top_k=0,
                do_sample=False
            )

        # Extract generated sequence and per-step logits
        sequences = generation.sequences
        scores = generation.scores

        # Decode only the newly generated tokens
        generated_ids = sequences[0, input_length:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute logprobs and top-k candidates at each step
        tokens, token_logprobs, top_logprobs = [], [], []
        for step_idx, step_scores in enumerate(scores):
            log_probs = F.log_softmax(step_scores, dim=-1)

            token_id = sequences[0, input_length + step_idx].unsqueeze(0)
            lp = log_probs.gather(1, token_id.unsqueeze(1)).item()

            topk_lp, topk_ids = log_probs.topk(args.top_k, dim=-1)
            topk_lp = topk_lp[0].tolist()
            topk_ids = topk_ids[0].tolist()
            topk_dict = {tokenizer.convert_ids_to_tokens(tok): logp
                         for tok, logp in zip(topk_ids, topk_lp)}

            tokens.append(tokenizer.convert_ids_to_tokens(token_id.item()))
            token_logprobs.append(lp)
            top_logprobs.append(topk_dict)

        # Format completion record
        completion_record = {
            "id": f"localcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "logprobs": {
                    "tokens": tokens,
                    "token_logprobs": token_logprobs,
                    "top_logprobs": top_logprobs
                },
                "finish_reason": "length"
            }]
        }

        # Merge with original item
        item_output = dict(item)
        item_output["judge"] = generated_text
        item_output["logprobs"] = completion_record["choices"][0]["logprobs"]
        all_results.append(item_output)

    # Write out final JSON
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nScoring complete! Results saved to {args.save_fp}")


if __name__ == "__main__":
    main()
