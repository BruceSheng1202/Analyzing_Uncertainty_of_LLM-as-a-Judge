#!/usr/bin/env bash

input_root="./GPT-4omini-batch/GPT-4omini-batch"
output_dir="./reasoning/tmp1"
models=("Qwen/Qwen2.5-72B-Instruct" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

mkdir -p "$output_dir"

mapfile -d '' json_files < <(
  find "$input_root" -type f -name 'Reasoning_*.jsonl' -print0
)

echo ">>> input_root = $input_root"
echo ">>> json_files found:"
for f in "${json_files[@]}"; do echo "    $f"; done
echo ">>> starting runs…"

for batch_fp in "${json_files[@]}"; do
  if [[ "$batch_fp" != *"GEval"* ]]; then
    echo ">>> Skipping $batch_fp (does not contain 'GEval')"
    continue
  fi
  script_dir=$(dirname "$batch_fp")
  for model in "${models[@]}"; do
    tag=$(basename "$model")
    name=$(basename "$batch_fp" .json)
    save_fp="$output_dir/${name}_${tag}.json"

    echo -e "\n>>> $batch_fp  →  $model"
    python "$script_dir/reasoning_eval.py" \
      --batch_fp "$batch_fp" \
      --save_fp "$save_fp" \
      --model "$model"
  done
done
