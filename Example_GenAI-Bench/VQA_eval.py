from datasets import load_dataset
dataset = load_dataset("BaiqiL/GenAI-Bench")

image_features = ['DALLE_3', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base', 'SDXL_Turbo']

import torch
import torch.nn.functional as F
import uuid
import time
import json
from PIL import Image
import numpy as np
from tqdm import tqdm  
import re
import argparse


class Args:
    def __init__(self):
        self.max_new_tokens = 128
        self.top_k = 10
        self.prompt_fp = "CoT_prompt.txt"
        self.model_name = "Qwen/Qwen2.5-VL-32B-Instruct"
        
        self._parse_args()
    
    def _parse_args(self):
        parser = argparse.ArgumentParser(description='Model inference arguments')
        parser.add_argument('--max_new_tokens', type=int, default=self.max_new_tokens)
        parser.add_argument('--top_k', type=int, default=self.top_k)
        parser.add_argument('--prompt_fp', type=str, default=self.prompt_fp)
        parser.add_argument('--model_name', type=str, default=self.model_name)
        
        args = parser.parse_args()
        
        self.max_new_tokens = args.max_new_tokens
        self.top_k = args.top_k
        self.prompt_fp = args.prompt_fp
        self.model_name = args.model_name

args = Args()
model_name = args.model_name

print("Loading model and processor...")
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

#  Processor
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)

#  Model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16, 
    device_map="auto"   
)
print("Model and processor loaded successfully.")



image_features = ['DALLE_3', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base', 'SDXL_Turbo']

all_dataset_results = []

dataset_length = len(dataset['train']) 

pbar_dataset = tqdm(total=dataset_length, desc="Processing Dataset Rows", unit="row")

for row_index in range(dataset_length):

    example = dataset['train'][row_index]
    prompt = example['Prompt']

    pbar_features = tqdm(total=len(image_features), desc=f"  Row {row_index + 1} Features", unit="feature", leave=False)
    
    results_for_current_row = {}

    for feature in image_features:
        
        pbar_features.set_description_str(f"  Row {row_index + 1}: {feature}")
        label = np.array(example['HumanRatings'][feature]).mean().round(2)
        item_output = {
            "dataset_row_index": row_index,
            "feature": feature,
            "original_prompt": prompt,
            "human_ratings": label
        }

        image_data = example[feature]

        # image
        try:
            if isinstance(image_data, torch.Tensor):
                #  CHW 
                if image_data.dtype == torch.float32 or image_data.dtype == torch.float64:
                    #  [0, 1] [0, 255] uint8
                    image_np = (image_data.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                else:
                    #  uint8
                    image_np = image_data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                image_pil = Image.fromarray(image_np)
            elif isinstance(image_data, np.ndarray):
                if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                    image_np = (image_data * 255).astype(np.uint8)
                else:
                    image_np = image_data.astype(np.uint8)
                image_pil = Image.fromarray(image_np)
            elif isinstance(image_data, Image.Image):
                image_pil = image_data
            else:
                raise TypeError(f"Unsupported image data type: {type(image_data)}")
        except Exception as e:
            error_msg = f"Error converting image data for {feature}: {e}"
            print(f"  Warning: raw {row_index}, feature {feature} error: {error_msg}")
            item_output["judge"] = f"Error: {error_msg}"
            item_output["logprobs"] = None
            results_for_current_row[feature] = item_output
            pbar_features.update(1)
            continue
        
        try:
            with open(args.prompt_fp, "r", encoding="utf-8") as file:
                template = file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"{args.prompt_fp} no found.")

        final_text = template.replace("{prompt}", prompt)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_pil,
                    },
                    {
                        "type": "text",
                        "text": final_text,
                    },
                ],
            }
        ]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False
                )

            #  logits
            sequences = generation.sequences
            scores = generation.scores
            generated_ids_trimmed = sequences[0, input_length:]
            full_generated_text = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

            target_score = None
            target_score_position = -1
        
            try:
                generated_tokens_list = [processor.decode([tid]) for tid in generated_ids_trimmed]
                if generated_tokens_list:
                    first_token = generated_tokens_list[0].strip()
                    if first_token.isdigit() and 1 <= int(first_token) <= 5:
                        target_score = int(first_token)
                        target_score_position = 0
                        target_token_id = generated_ids_trimmed[0].item()
                        print(f"evaluate row {row_index} generated by {feature}, score is {target_score}, while human rating is {label}.")
                    else:
                        # find ':'
                        colon_index = full_generated_text.find(':')
                        if colon_index != -1:
                            match = re.search(r'\d+', full_generated_text[colon_index+1:])
                            if match:
                                target_score_str = match.group() 
                                target_score = int(target_score_str) 
                                print(f"evaluate row {row_index} generated by {feature}, score is {target_score} , while human rating is {label}.")

                                generated_tokens_list = [processor.decode([tid]) for tid in generated_ids_trimmed]

                                target_token_str = str(target_score)
                                for i, token_str in enumerate(generated_tokens_list):
                                    cleaned_token_str = token_str.strip()
                                    if cleaned_token_str == target_token_str:
                                        target_score_position = i
                                        target_token_id = generated_ids_trimmed[i].item()
                                        break
                                else:
                                    print(f"  warning: no '{target_token_str}' found token。")
                            else:
                                print("  warning: no number found after :")
                        else:
                            print("  warning: no : found")
            except Exception as e:
                print(f"  error: {e}")

            #  token  logprob
            target_token_logprob = None
            target_token_top_logprobs = None # 
    
            if target_score_position != -1 and 0 <= target_score_position < len(scores):
                try:
                    step_idx = target_score_position
                    step_scores = scores[step_idx] #  logits [1, vocab_size]

                    log_probs = F.log_softmax(step_scores, dim=-1) # [1, vocab_size]

                    #  token ID  logprob
                    target_token_id_tensor = generated_ids_trimmed[step_idx].unsqueeze(0).unsqueeze(0) # [1, 1]
                    target_lp_tensor = log_probs.gather(1, target_token_id_tensor) # [1, 1]
                    target_token_logprob = target_lp_tensor.item()

                    # ---  top-k ---
                    vocab_size = log_probs.size(-1)
                    current_top_k = min(args.top_k, vocab_size) # args.top_k 
                    topk_lp, topk_ids = log_probs.topk(current_top_k, dim=-1) # [1, top_k]
                    topk_lp_squeezed = topk_lp.squeeze(0)   # [top_k]
                    topk_ids_squeezed = topk_ids.squeeze(0) # [top_k]
                    topk_lp_list = topk_lp_squeezed.tolist()
                    topk_ids_list = topk_ids_squeezed.tolist()
                    topk_tokens_list = processor.tokenizer.convert_ids_to_tokens(topk_ids_list)
                    target_token_top_logprobs = {
                        tok: logp for tok, logp in zip(topk_tokens_list, topk_lp_list)
                    }
                    # ---  ---

                except Exception as e:
                    print(f"  extracting token logprob : {e}")
            else:
                print(f"  fail to get logprob:  {target_score_position} invalid.")

            # 6.  logprobs data
            final_logprobs_data = {
                "target_score": target_score, 
                "target_token_top_logprobs": target_token_top_logprobs #  top-k
            }

            completion_record = {
                "id": f"localcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_generated_text},
                    "logprobs": final_logprobs_data,
                    "finish_reason": "length" if len(generated_ids_trimmed) == args.max_new_tokens else "stop"
                }]
            }

            item_output["judge"] = full_generated_text
            item_output["logprobs"] = completion_record["choices"][0]["logprobs"]

            results_for_current_row[feature] = item_output

        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            print(f"  ❌ error in row {row_index}, feature {feature} : {error_msg}")
            item_output["judge"] = error_msg
            item_output["logprobs"] = None
            results_for_current_row[feature] = item_output
        finally:
            pbar_features.update(1)
    
    pbar_features.close()

    current_row_results_list = list(results_for_current_row.values())
    all_dataset_results.extend(current_row_results_list)

    pbar_dataset.update(1)

pbar_dataset.close()

safe_model_name = model_name.replace('/', '_')
final_save_fp = f"{safe_model_name}_{args.prompt_fp}.json"
try:
    with open(final_save_fp, 'w', encoding='utf-8') as f:
        json.dump(all_dataset_results, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 VLM evaluation and  Logprob extraction finish! save to {final_save_fp}")
    print(f"    {len(all_dataset_results)} items in total.")
except Exception as e:
    print(f"\n❌ fails to save: {e}")
    print(json.dumps(all_dataset_results[:5], indent=2, ensure_ascii=False))

