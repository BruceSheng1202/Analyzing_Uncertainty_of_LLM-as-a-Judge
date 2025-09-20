#!/usr/bin/env bash
set -euo pipefail
for dim in con coh flu rel; do
  echo "=== Processing dimension: $dim ==="
#   python qwen_eval.py \
#     	--prompt_fp   summeval/prompts/summeval/${dim}_detailed.txt \
#     	--summeval_fp summeval/data/summeval.json \
# 		--save_fp     results/oversampling/qwen_${dim}_detailed_qwen25_summeval.json
#   python qwen_eval.py \
#     	--prompt_fp   summeval/prompts/summeval/${dim}_detailed.txt \
#     	--summeval_fp summeval/data/summeval.json \
# 		--save_fp     results/oversampling/qwen_${dim}_detailed_dsr1_summeval.json \
# 		--model 	"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
#   python qwen_eval_dialsumm.py \
# 		--prompt_fp   summeval/prompts/summeval/${dim}_detailed.txt \
# 		--input_fp 		summeval/data/dialsumm.jsonl\
# 		--save_fp     results/oversampling/qwen_${dim}_detailed_qwen25_dialsumm.json
  python qwen_eval_dialsumm.py \
    	--prompt_fp   summeval/prompts/summeval/${dim}_detailed.txt \
  		--input_fp 		summeval/data/dialsumm.jsonl\
		--save_fp     results/oversampling/qwen_${dim}_detailed_dsr1_dialsumm.json \
		--model 	"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

  echo
done
