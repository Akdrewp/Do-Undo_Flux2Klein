import torch
import numpy as np
import json
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from PIL import Image
import torch.nn.functional as F

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEVICE = "cuda"
SAMPLE_ID = "sample_00122"

# --- 1. Load Data ---
json_path = f"static_test_final_dataset/{SAMPLE_ID}_meta.json"
npz_path = f"static_test_tuples_slim/{SAMPLE_ID}.npz"

with open(json_path, 'r') as f:
    meta = json.load(f)
do_prompt, undo_prompt = meta['Pf'], meta['Pr']

print(do_prompt)
print(undo_prompt)

data = np.load(npz_path)
io_img = Image.fromarray(data['Io']).convert("RGB") # The 'Undo' Target
if_gt = Image.fromarray(data['If']).convert("RGB")  # The 'Do' Target

def get_consistency_loss(gen_img, target_img):
    """Calculates the MSE between two PIL images (Consistency Loss)."""
    # Convert to Tensors [-1, 1] to match training logic
    t1 = torch.from_numpy(np.array(gen_img).transpose(2, 0, 1)).float().to(DEVICE) / 127.5 - 1.0
    t2 = torch.from_numpy(np.array(target_img).transpose(2, 0, 1)).float().to(DEVICE) / 127.5 - 1.0
    return F.mse_loss(t1, t2).item()

# --- 2. Initialize Base Pipeline ---
pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)

# --- 3. Run BASE MODEL ---
print("Running base model")
base_do = pipe(prompt=do_prompt, image=io_img, num_inference_steps=25).images[0]
base_undo = pipe(prompt=undo_prompt, image=if_gt, num_inference_steps=25).images[0]

base_loss = get_consistency_loss(base_undo, io_img)

# --- 4. Load LoRA & Run TRAINED MODEL ---
print("Injecting LoRA")
state_dict = load_file("./checkpoints_7static_1024rank/doundo_flux_epoch_2/adapter_model.safetensors")
pipe.load_lora_into_transformer(state_dict, transformer=pipe.transformer)

lora_do = pipe(prompt=do_prompt, image=io_img, num_inference_steps=25).images[0]
lora_undo = pipe(prompt=undo_prompt, image=if_gt, num_inference_steps=25).images[0]

lora_loss = get_consistency_loss(lora_undo, io_img)

diff = np.abs(np.array(io_img).astype(float) - np.array(lora_undo).astype(float))
print(f"Mean Pixel Change: {np.mean(diff):.4f}")

# --- 5. Print Metrics ---
print(f"CONSISTENCY LOSS REPORT ({SAMPLE_ID})")
print(f"Base Model Undo Loss: {base_loss:.6f}")
print(f"LoRA Model Undo Loss: {lora_loss:.6f}")
print(f"Improvement: {((base_loss - lora_loss) / base_loss * 100):.2f}%")

# --- 6. Save Grid (as before) ---
grid = Image.new('RGB', (1536, 1024))
def p(img, col, row): grid.paste(img.resize((512, 512)), (col*512, row*512))
p(if_gt, 0, 0); p(base_do, 1, 0); p(lora_do, 2, 0)
p(io_img, 0, 1); p(base_undo, 1, 1); p(lora_undo, 2, 1)
grid.save(f"final_results_{SAMPLE_ID}.png")