import os
import json
import time
import uuid
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm

def load_jsonl(fp):
    data = []
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prompt_fp', type=str, required=True,
        help='Template file with {{Document}} and {{Summary}} placeholders'
    )
    parser.add_argument(
        '--input_fp', type=str, required=True,
        help='Path to JSON or JSONL file with items: id, dialogue, summary, annotations, model_id'
    )
    parser.add_argument(
        '--save_fp', type=str, required=True,
        help='Path where the results will be saved'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=1000,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--top_k', type=int, default=10,
        help='Top k token logprobs to record per step'
    )
    parser.add_argument(
        '--model', type=str, default="Qwen/Qwen2.5-72B-Instruct",
        help='the judge model you wanna use to evaluate'
    )
    args = parser.parse_args()

    if args.input_fp.endswith('.jsonl'):
        data = load_jsonl(args.input_fp)
    else:
        with open(args.input_fp, 'r', encoding='utf-8') as f:
            data = json.load(f)

    prompt_template = open(args.prompt_fp, 'r', encoding='utf-8').read()

    # Initialize model
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
        cache_dir=os.environ.get("HF_HOME", None)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=os.environ.get("HF_HOME", None)
    )
    model.eval()

    results = []
    for item in tqdm(data, desc="Scoring"):
        document = item["dialogue"]
        summary  = item["summary"]
        prompt = prompt_template.replace("{{Document}}", document)\
                                .replace("{{Summary}}", summary)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[-1]

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                top_k=args.top_k,
                do_sample=True, # set to True to enable sampling (use temperature / top_p)
                temperature = 1
            )
        seq    = gen.sequences
        scores = gen.scores

        new_ids  = seq[0, input_len:].tolist()
        gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)

        tokens, token_lps, top_lps = [], [], []
        for step, sc in enumerate(scores):
            lp_dist = F.log_softmax(sc, dim=-1)
            tok_id  = seq[0, input_len + step].unsqueeze(0)
            lp      = lp_dist.gather(1, tok_id.unsqueeze(1)).item()

            topk_lp, topk_ids = lp_dist.topk(args.top_k, dim=-1)
            topk_dict = {
                tokenizer.convert_ids_to_tokens(t): l
                for t, l in zip(topk_ids[0].tolist(), topk_lp[0].tolist())
            }

            tokens.append(tokenizer.convert_ids_to_tokens(tok_id.item()))
            token_lps.append(lp)
            top_lps.append(topk_dict)

        out = {
            "id":           item.get("id"),
            "dialogue":     document,
            "summary":      summary,
            "annotations":  item.get("annotations"),
            "model_id":     item.get("model_id"),
            "judge":        gen_text,
            "logprobs": {
                "tokens":         tokens,
                "token_logprobs": token_lps,
                "top_logprobs":   top_lps
            }
        }
        results.append(out)
     


    os.makedirs(os.path.dirname(args.save_fp), exist_ok=True)
    with open(args.save_fp, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Scoring complete! Saved to {args.save_fp}")

if __name__ == "__main__":
    main()


