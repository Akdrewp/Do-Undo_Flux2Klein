import torch
import numpy as np
import json
import os
import random
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# --- Settings ---
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEVICE = "cuda"
ADAPTER_PATH = "./checkpoints_7static_1024rank/doundo_flux_epoch_2/adapter_model.safetensors"
NUM_TESTS = 20
DATA_DIR = "static_test_tuples_slim"
META_DIR = "static_test_final_dataset"

def calculate_metrics(img1, img2):
  # Convert PIL to numpy grayscale
  im1 = np.array(img1.convert('L').resize((512, 512)))
  im2 = np.array(img2.convert('L').resize((512, 512)))
  
  # L1 difference
  l1_dist = np.mean(np.abs(im1 - im2))
  
  # SSIM: Structural similarity (0 to 1)
  score, _ = ssim(im1, im2, full=True)
  return l1_dist, score

# 1. Setup Models
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

# 2. Get Test Samples
all_samples = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
test_samples = random.sample(all_samples, min(NUM_TESTS, len(all_samples)))

results = {"base": {"l1": [], "ssim": []}, "lora": {"l1": [], "ssim": []}}

for i, sid in enumerate(test_samples):
  print(f"🧪 Testing {sid} ({i+1}/{NUM_TESTS})...")
  
  # Load Data
  data = np.load(f"{DATA_DIR}/{sid}.npz")
  with open(f"{META_DIR}/{sid}_meta.json", 'r') as f:
    meta = json.load(f)
  
  io_gt = Image.fromarray(data['Io']).convert("RGB") # The Target
  if_img = Image.fromarray(data['If']).convert("RGB") # The Input for Undo
  undo_prompt = meta['Pr']

  # --- Run Base Model ---
  with torch.no_grad():
    base_undo = pipe(prompt=undo_prompt, image=if_img, num_inference_steps=20).images[0]
  b_l1, b_ssim = calculate_metrics(io_gt, base_undo)
  results["base"]["l1"].append(b_l1)
  results["base"]["ssim"].append(b_ssim)

  # --- Load LoRA (Only on first run) ---
  if i == 0:
    state_dict = load_file(ADAPTER_PATH)
    pipe.load_lora_into_transformer(state_dict, transformer=pipe.transformer)

  # --- Run LoRA Model ---
  with torch.no_grad():
    lora_undo = pipe(prompt=undo_prompt, image=if_img, num_inference_steps=20).images[0]
  l_l1, l_ssim = calculate_metrics(io_gt, lora_undo)
  results["lora"]["l1"].append(l_l1)
  results["lora"]["ssim"].append(l_ssim)

# --- Final Report ---
print("Results")
print("="*30)
print(f"BASE MODEL | Avg L1 Distance: {np.mean(results['base']['l1']):.2f} | Avg SSIM: {np.mean(results['base']['ssim']):.4f}")
print(f"LORA MODEL | Avg L1 Distance: {np.mean(results['lora']['l1']):.2f} | Avg SSIM: {np.mean(results['lora']['ssim']):.4f}")

improvement = (np.mean(results['base']['l1']) - np.mean(results['lora']['l1'])) / np.mean(results['base']['l1']) * 100
print(f"\n LORA IS {improvement:.1f}% MORE ACCURATE THAN BASE")