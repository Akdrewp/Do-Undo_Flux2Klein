import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
INPUT_DIR = "./processed_tuples"
OUTPUT_DIR = "./final_dataset"
BATCH_SIZE = 32  # Change to 8 if you have 12GB+ VRAM
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
    attn_implementation="sdpa"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
processor.tokenizer.padding_side = "left"

system_instruction = (
  "You are an expert action describer for AI video datasets. Look at the 3 frames. "
  "Describe the physical movement, the specific grip or contact point, the direction of motion, and the final change in state. "
  "Do NOT describe lighting, shadows, camera angles, or background surfaces. "
  "Write exactly one highly detailed instruction sentence per action. Focus entirely on the mechanics of the manipulation."
  "\n\nExample 1:"
  "\nAction: 'open' followed by 'close' on a drawer"
  "\nForward: Open the top wooden drawer with the right hand by gripping the handle and pulling it straight backward until it is fully extended."
  "\nReverse: Close the top wooden drawer with the right hand by pushing the front panel forward until it is completely flush with the desk."
  "\n\nExample 2:"
  "\nAction: 'pickup' followed by 'putdown' on a mug"
  "\nForward: Pick up the ceramic mug with the right hand by grasping its side handle and lifting it vertically off the table."
  "\nReverse: Put down the ceramic mug with the right hand by lowering it vertically until its base rests flat on the table."
)

def process_batch(batch_paths):
    valid_paths = []
    messages_batch = []
    sample_ids = []

    for npz_path in batch_paths:
        try:
            data = np.load(npz_path)
            raw_frames = [data['Io'], data['context'][2], data['If']]
            
            frames = []
            for f in raw_frames:
                if f.dtype != np.uint8:
                    f = (f * 255).astype(np.uint8) if f.max() <= 1.0 else f.astype(np.uint8)
                frames.append(Image.fromarray(f))
            
            obj = str(data['obj'])
            vb_do, vb_undo = data['vb'] 
            
            user_query = (
                f"The action is '{vb_do}' followed by '{vb_undo}' on a {obj}. "
                "Based on these frames, provide the Forward and Reverse prompts in this exact format: "
                "Forward: [description]\nReverse: [description]"
            )

            content = [{"type": "image", "image": img, "max_pixels": 262144} for img in frames]
            content.append({"type": "text", "text": f"{system_instruction}\n\nTask: {user_query}"})
            
            messages_batch.append([{"role": "user", "content": content}])
            sample_ids.append(os.path.basename(npz_path).replace(".npz", ""))
            valid_paths.append(npz_path)
            
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            continue

    if not messages_batch:
        return []

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=300)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    responses = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    results = []
    for i, response in enumerate(responses):
        lines = response.split('\n')
        pf, pr = "", ""
        for line in lines:
            if line.startswith("Forward:"):
                pf = line.replace("Forward:", "").strip()
            elif line.startswith("Reverse:"):
                pr = line.replace("Reverse:", "").strip()
                
        results.append({
            "sample_id": sample_ids[i],
            "Pf": pf,
            "Pr": pr,
            "original_npz": valid_paths[i]
        })
        
    return results

# --- Setup and Filtering ---
# Grab every single .npz file in the directory
all_npz_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")])

# Filter out files that already have JSONs generated to allow easy resuming
pending_files = []
for filename in all_npz_files:
    sample_id = filename.replace(".npz", "")
    output_path = os.path.join(OUTPUT_DIR, f"{sample_id}_meta.json")
    if not os.path.exists(output_path):
        pending_files.append(os.path.join(INPUT_DIR, filename))

print(f"Total .npz files found: {len(all_npz_files)}")
print(f"Files needing processing: {len(pending_files)}\n")

# --- Batched Execution Loop ---
for i in tqdm(range(0, len(pending_files), BATCH_SIZE), desc="Batched Generation"):
    batch_paths = pending_files[i : i + BATCH_SIZE]
    
    batch_results = process_batch(batch_paths)
    
    for res in batch_results:
        output_path = os.path.join(OUTPUT_DIR, f"{res['sample_id']}_meta.json")
        with open(output_path, 'w') as f:
            json.dump(res, f, indent=4)