import torch
import numpy as np
import json
import os
import random
from diffusers import DiffusionPipeline
from peft import PeftModel  # <-- NEW: Required for your loading method
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# --- Settings ---
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEVICE = "cuda"
ADAPTER_PATH = "./checkpoints5/epoch_2/"
NUM_TESTS = 20
DATA_DIR = "static_test_tuples_slim"
META_DIR = "static_test_final_dataset"

def calculate_metrics(img1, img2):
    # Convert PIL to numpy grayscale for math
    im1 = np.array(img1.convert('L').resize((512, 512)))
    im2 = np.array(img2.convert('L').resize((512, 512)))
    
    # L1: Mean Absolute Difference (Linear)
    l1_dist = np.mean(np.abs(im1 - im2))
    
    # SSIM: Structural similarity (0 to 1)
    score, _ = ssim(im1, im2, data_range=255, full=True) 
    return l1_dist, score

# 1. Setup Models
print("Loading Base Pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

# 2. Get Test Samples
all_samples = [f.split('.')[0] for f in os.listdir(DATA_DIR) if f.endswith('.npz')]
test_samples = random.sample(all_samples, min(NUM_TESTS, len(all_samples)))

results = {"base": {"l1": [], "ssim": []}, "lora": {"l1": [], "ssim": []}}

# --- PRELOAD DATA TO ENSURE EXACT MATCHES ---
test_data = []
for sid in test_samples:
    data = np.load(f"{DATA_DIR}/{sid}.npz")
    with open(f"{META_DIR}/{sid}_meta.json", 'r') as f:
        meta = json.load(f)
    
    test_data.append({
        "sid": sid,
        "io_gt": Image.fromarray(data['Io']).convert("RGB"), # Target
        "if_gt": Image.fromarray(data['If']).convert("RGB"), # Input
        "undo_prompt": meta['Pr']
    })

# ==========================================
# PHASE 1: PURE BASE MODEL TESTING
# ==========================================
print("\n--- Running Base Model Tests ---")
for i, item in enumerate(test_data):
    print(f"🧪 [Base] Testing {item['sid']} ({i+1}/{NUM_TESTS})...")
    
    with torch.no_grad():
        base_undo = pipe(
            prompt=item['undo_prompt'], 
            image=item['if_gt'], 
            num_inference_steps=25
        ).images[0]
        
    b_l1, b_ssim = calculate_metrics(item['io_gt'], base_undo)
    results["base"]["l1"].append(b_l1)
    results["base"]["ssim"].append(b_ssim)

# ==========================================
# PHASE 2: LORA MODEL TESTING
# ==========================================
print("\n--- Injecting LoRA Weights ---")
# YOUR EXACT LOADING METHOD:
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer, 
    ADAPTER_PATH, 
    adapter_name="default"
)
print("LoRA loaded successfully!")

print("\n--- Running LoRA Model Tests ---")
for i, item in enumerate(test_data):
    print(f"🧪 [LoRA] Testing {item['sid']} ({i+1}/{NUM_TESTS})...")
    
    with torch.no_grad():
        lora_undo = pipe(
            prompt=item['undo_prompt'], 
            image=item['if_gt'], 
            num_inference_steps=25
        ).images[0]
        
    l_l1, l_ssim = calculate_metrics(item['io_gt'], lora_undo)
    results["lora"]["l1"].append(l_l1)
    results["lora"]["ssim"].append(l_ssim)

# ==========================================
# FINAL REPORT
# ==========================================
print("\n" + "="*40)
print("📈 FINAL THESIS BENCHMARK (25 STEPS)")
print("="*40)
base_l1_avg = np.mean(results['base']['l1'])
base_ssim_avg = np.mean(results['base']['ssim'])
lora_l1_avg = np.mean(results['lora']['l1'])
lora_ssim_avg = np.mean(results['lora']['ssim'])

print(f"BASE MODEL | Avg L1 (Lower is better):  {base_l1_avg:.2f} | Avg SSIM (Higher is better): {base_ssim_avg:.4f}")
print(f"LORA MODEL | Avg L1 (Lower is better):  {lora_l1_avg:.2f} | Avg SSIM (Higher is better): {lora_ssim_avg:.4f}")

improvement_l1 = ((base_l1_avg - lora_l1_avg) / base_l1_avg) * 100
improvement_ssim = ((lora_ssim_avg - base_ssim_avg) / base_ssim_avg) * 100

print(f"\n🚀 RESULTS:")
print(f"- L1 Error decreased by: {improvement_l1:.1f}%")
print(f"- Structural Similarity improved by: {improvement_ssim:.1f}%")