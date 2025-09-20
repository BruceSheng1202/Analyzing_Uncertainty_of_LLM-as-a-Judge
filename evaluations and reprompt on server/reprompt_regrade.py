#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import re
import time
import uuid
import pandas as pd
import numpy as np
import argparse
from R2CCP.main import R2CCP
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

def score_and_extract(item, prompt, model, tokenizer, max_new_tokens=10000, top_k=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[-1]
    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            top_k=0,
            do_sample=False
        )

    # Extract generated sequence and per-step logits
    sequences = generation.sequences        # shape [1, input_length + new_tokens]
    scores    = generation.scores           # list of length new_tokens, each [1, vocab_size]

    # Decode only the newly generated tokens
    generated_ids = sequences[0, input_length:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    tokens        = []
    token_logprobs = []
    top_logprobs   = []
    for step_idx, step_scores in enumerate(scores):
        # step_scores: [1, vocab_size]
        log_probs = F.log_softmax(step_scores, dim=-1)  # shape [1, vocab_size]

        # actual token id & its logprob
        token_id = sequences[0, input_length + step_idx].unsqueeze(0)
        lp = log_probs.gather(1, token_id.unsqueeze(1)).item()

        # top-k tokens and their logprobs
        topk_lp, topk_ids = log_probs.topk(top_k, dim=-1)
        topk_lp   = topk_lp[0].tolist()
        topk_ids  = topk_ids[0].tolist()
        topk_dict = {
            tokenizer.convert_ids_to_tokens(tok): logp
            for tok, logp in zip(topk_ids, topk_lp)
        }

        tokens.append(tokenizer.convert_ids_to_tokens(token_id.item()))
        token_logprobs.append(lp)
        top_logprobs.append(topk_dict)

    
    m = re.search(r'\d+(?:\.\d+)?', generated_text)
    raw_score = float(m.group()) if m else None
    target_logits = top_logprobs[1] 
    # Format one “completion” record similar to OpenAI API
    completion_record = {
        "id":              f"localcmpl-{uuid.uuid4().hex[:8]}",
        "object":          "chat.completion",
        "created":         int(time.time()),
        "model":           model,
        "prompt": prompt,
        "choices": [{
            "index":      0,
            "message":    {"role": "assistant", "content": generated_text},
            "target":     {
                "raw_score":    raw_score,
                "target_logits": target_logits
            },
            "finish_reason": "length"
        }]
    }

    item_output = dict(item)
    item_output["prompt"] = prompt
    item_output["judge"]    = generated_text
    item_output["target"] = completion_record["choices"][0]["target"]

    return item_output, raw_score, target_logits

def merge_intervals(sample_intervals):
    if not sample_intervals:
        return (1,5)
    lows = [low for low, high in sample_intervals]
    highs = [high for low, high in sample_intervals]
    return (min(lows), max(highs))

def boundary_adjustment(value, threshold=0.0):
    label_set=np.array([1, 1.33, 1.67, 2, 2.33, 2.67, 3, 3.33, 3.67, 4, 4.33, 4.67, 5])
    threshold_max = (label_set[-1] - label_set[0]) / (len(label_set) - 1) / 2
    threshold = min(threshold_max, threshold)
    adjusted_value = next((num for num in label_set if abs(num - value) < threshold+0.01), value)
    
    return adjusted_value

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dimension', type=str, required=True,
        help='which dimension to evaluate'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=10000,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Number of top token logprobs to record at each generation step'
    )
    parser.add_argument(
        '--model', type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        help='Model used to evaluate as a judge'
    )

    args = parser.parse_args()
    dimension = args.dimension
    os.makedirs('R2CCP_paths', exist_ok=True)

    prompt_fp = f"./summeval/prompts/summeval/{dimension[:3]}_detailed.txt"
    # Load dataset and prompt template
    with open(f"./summeval/data/summeval.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompt_template = open(prompt_fp, 'r', encoding='utf-8').read()

    cal_data, test_data = train_test_split(
            data, test_size=0.5, random_state=42
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
    raw_scores = []
    y_scores   = []
    logits_rows = []
    print(f"\nOffline calibration begins! Collecting logits for conformal prediction!")
    for idx, item in enumerate(tqdm(cal_data, desc="Scoring")):
        # Build the full prompt by filling in document and summary
        document = item["source"]
        summary  = item["system_output"]
        prompt   = prompt_template.replace("{{Document}}", document)\
                                  .replace("{{Summary}}", summary)

        item_output, raw_score, logits_vec = score_and_extract(item, prompt, model, tokenizer, 50, args.top_k)
        all_results.append(item_output)

        raw_scores.append(float(raw_score))
        logits_rows.append({str(k): v for k, v in logits_vec.items()})
        y_scores.append(item['scores'][dimension])
        
    # Write out final JSON
    with open(f"./reprompt_regrade/cal_results/Summeval_{dimension}", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nScoring for calibration complete!")

    # Calibration Process
    raw_df = pd.DataFrame({'raw_score': raw_scores})
    X_cal   = pd.DataFrame(logits_rows).reindex(columns=['1','2','3','4','5'])
    X_cal.fillna(math.log(1e-05), inplace=True)
    y_cal   = pd.DataFrame({dimension: y_scores})

    calibration_set = pd.concat([X_cal, y_cal], axis=1)
    calibration_set.to_csv(f"./reprompt_regrade/logits_results/Summeval_{dimension}_logits.csv", index=False)

    X_cal = X_cal.to_numpy().astype(np.float32)
    y_cal = y_cal.to_numpy().astype(np.float32)

    if os.path.exists('R2CCP_paths/model_save_destination.pth'):
        os.remove('R2CCP_paths/model_save_destination.pth')
    
    predictor = R2CCP({'model_path': 'R2CCP_paths/model_save_destination.pth', 'max_epochs': 100, 'alpha': 0.1})
    predictor.fit(X_cal, y_cal.flatten())

    all_results = []
    raw_scores = []
    y_scores   = []
    logits_rows = []
    print(f"\nOnline testing begins! Reprompt intervals to the judge!")
    for idx, item in enumerate(tqdm(test_data, desc="Scoring")):
        # Build the full prompt by filling in document and summary
        document = item["source"]
        summary  = item["system_output"]
        prompt   = prompt_template.replace("{{Document}}", document)\
                                  .replace("{{Summary}}", summary)

        item_output, raw_score, logits_vec = score_and_extract(item, prompt, model, tokenizer, args.max_new_tokens, args.top_k)
        all_results.append(item_output)

        raw_scores.append(float(raw_score))
        print(f"My initial score is {float(raw_score)}\n")
        y_scores.append(item['scores'][dimension])
        print(f"The ground truth is {item['scores'][dimension]}\n")
        logits_rows.append({str(k): v for k, v in logits_vec.items()})

        X_test = pd.DataFrame([logits_vec], columns=['1', '2', '3', '4', '5'])
        X_test.fillna(math.log(1e-05), inplace=True)
        interval = predictor.get_intervals(X_test)
        interval = [merge_intervals(sample_intervals) for sample_intervals in interval]
        low = boundary_adjustment(interval[0][0], threshold=0.5)
        up = boundary_adjustment(interval[0][1], threshold=0.5)
        interval = f"[{low:.2f}, {up:.2f}]"
        print(f"The interval for this evaluation is {interval}")

        # Reprompt
        reprompt = open("./reprompt.txt", 'r', encoding='utf-8').read().replace("{{Interval}}", interval)
        reprompt = ( "Let me show you our evalutaion record. Based on all these information, make dicision and give me final score."
                + "Initial prompt: \n" + prompt + "\n" 
                + "Initial response: \n" + item_output["judge"] + "\n"
                + "Reprompt and Regrade: \n" + reprompt)
        rep_item, rep_raw_score, rep_logits_vec = score_and_extract(
            item, reprompt, model, tokenizer,
            1000, args.top_k
        )

        all_results.append(rep_item)

        raw_scores.append(rep_raw_score)
        print(f"My final score is {rep_raw_score}\n")
        if idx == 10:
            break
        
    # Write out final JSON
    with open(f"./reprompt_regrade/test_results/Summeval_{dimension}", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open("raw_scores.txt", "w") as file:
        for score in raw_scores:
            file.write(f"{score}\n")

if __name__ == "__main__":
    main()
