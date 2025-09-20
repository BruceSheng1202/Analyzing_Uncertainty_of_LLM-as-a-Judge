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


def score_and_extract(prompt, model, tokenizer, max_new_tokens=10000, top_k=10):
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

    
    # m = re.search(r'\d+(?:\.\d+)?', generated_text)
    # raw_score = float(m.group()) if m else None
    # target_logits = top_logprobs[10] 

     # 1)  raw_score_str
    m = re.search(r'Final Score\s*[::]\s*([1-5]+(?:\.\d+)?)', generated_text)
    if m:
        raw_score_str = m.group(1)
        raw_score     = float(raw_score_str)
    else:
        # last score
        nums = re.findall(r'[1-5]+(?:\.\d+)?', generated_text)
        raw_score_str = nums[-1] if nums else None
        raw_score     = float(raw_score_str) if raw_score_str else None

    # 2)  step index
    target_idx = None
    if raw_score_str is not None:
        for i, tok in enumerate(tokens):
            if tok.strip() == raw_score_str:
                target_idx = i
                break
    # first number token
    if target_idx is None:
        for i, tok in enumerate(tokens):
            if re.fullmatch(r'[1-5]+(?:\.\d+)?', tok.strip()):
                target_idx = i
                break

    # 3)  logits 
    if target_idx is not None and target_idx < len(top_logprobs):
        target_logits = top_logprobs[target_idx]
    else:
        #  fallback top_logprobs[0] 
        target_logits = top_logprobs[0]

    return generated_text, raw_score, target_logits

def softmax(x):
    x = np.asarray(x)
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

def boundary_adjustment(value, threshold=0.0):
    label_set=np.array([1, 2, 3, 4, 5])
    threshold_max = (label_set[-1] - label_set[0]) / (len(label_set) - 1) / 2
    threshold = min(threshold_max, threshold)
    adjusted_value = next((num for num in label_set if abs(num - value) <= threshold), value)
    adjusted_value = np.clip(adjusted_value, 1, 5)
    return adjusted_value

def main():
    parser = argparse.ArgumentParser()
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

    seeds = {
        # "cosmos": 1,
        # "drop": 18,
        "esnli": 9,
        "gsm8k": 30,
    }
    
    pattern = re.compile(
        r"""['"]?
        overall\W+quality
        ['"]?
        \s*[::]\s*
        ([1-5])
        """,
        re.IGNORECASE | re.VERBOSE
    )

    for dimension, seed in seeds.items(): 
        test_df = pd.read_csv(f"./reprompt_regrade/socreval/R2CCP_SocREval_{dimension}_{seed}.csv") 
        test_df['y_test'] = test_df['y_test'].astype(float)
        with open(f"./reprompt_regrade/socreval/dsr1_socreval_{dimension}.json","r",encoding="utf-8") as f:
            records = pd.DataFrame(json.load(f))
        logits = pd.read_csv(f"./reprompt_regrade/socreval/SocREval_{dimension}_logits.csv") 

        calibr_records, test_records = train_test_split(
            records,
            test_size=0.5,
            random_state=seed,
            shuffle=True
        )

        calibr_logits, test_logits = train_test_split(
            logits,
            test_size=0.5,
            random_state=seed,
            shuffle=True
        )

        results = []
        for i in tqdm(range(len(test_records))):            
            X_test = test_logits.iloc[i][['1','2','3','4','5']].to_numpy().astype(np.float32)
            X_test_softmax = softmax(X_test)
        
            init_score_weight = X_test_softmax @ np.array([1, 2, 3, 4, 5])
            
            row = test_df.iloc[i]
            low = boundary_adjustment(row['low'], threshold=0.5)
            up = boundary_adjustment(row['up'], threshold=0.5)
            interval = f"[{low:.2f}, {up:.2f}]"
            
            rec = test_records.iloc[i]
            prompt0 = rec['body']['messages'][0]['content']
            init_ans = rec['judge']

            if init_ans[-2].isdigit():
                init_raw_score = int(init_ans[-2])
            elif pattern.search(init_ans):
                m = pattern.search(init_ans)
                num_str = m.group(1)
                init_raw_score = int(num_str)

            reprompt = open("./reprompt.txt", 'r', encoding='utf-8').read().replace("{{Interval}}", interval)
            reprompt = ( "Let me show you our evalutaion record. Based on all these information, make dicision and give me final score."
                    + "Initial prompt: \n" + prompt0 + "\n" 
                    + "Initial response: \n" + init_ans + "\n"
                    + "Reprompt and Regrade: \n" + reprompt)
            
            gen_text, re_score_raw, re_logits = score_and_extract(
                reprompt, model, tokenizer,
                args.max_new_tokens, args.top_k
            )
            
            re_logits = {str(k): max(v,math.log(1e-5)) for k, v in re_logits.items()}
            for k in ['1','2','3','4','5']:
                if k not in re_logits or re_logits[k] is None:
                    re_logits[k] = math.log(1e-5)
            
            re_logits = np.array([re_logits[l] for l in ['1','2','3','4','5']], dtype=np.float64).reshape(1,-1)
            re_logits_softmax = softmax(re_logits)
            re_score_weight = re_logits_softmax @ np.array([1, 2, 3, 4, 5])

            results.append({
                'index':      i,
                'low':        low,
                'up':         up,
                'init_score_raw': init_raw_score,
                'init_score_weight': init_score_weight,
                're_score_raw':   re_score_raw,
                're_score_weight':  re_score_weight,
                'ground_truth': row['y_test'],
                'final_text': gen_text
            })

        res_df = pd.DataFrame(results)
        res_df.to_csv(f"bestseed_reprompt_socreval_{dimension}.csv", index=False)

if __name__ == "__main__":
    main()
