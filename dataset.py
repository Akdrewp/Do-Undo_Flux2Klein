import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DoUndoDataset(Dataset):
  def __init__(self, npz_dir, json_dir, transform=None):
    self.npz_dir = npz_dir
    self.json_dir = json_dir
    
    # 1. Look for NPZs (the images)
    npz_files = {f.replace(".npz", "") for f in os.listdir(npz_dir) if f.endswith(".npz")}
    
    # 2. Look for JSONs (the prompts) - stripping the '_meta' suffix for matching
    json_files = {f.replace("_meta.json", "") for f in os.listdir(json_dir) if f.endswith(".json")}
    
    # 3. Intersection ensures we only train on samples that have BOTH images and prompts
    self.sample_ids = sorted(list(npz_files.intersection(json_files)))
    
    if len(self.sample_ids) == 0:
        raise ValueError(f"No matching files found! Check {npz_dir} and {json_dir}")

    self.transform = transform or transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])
    
    print(f"Verified {len(self.sample_ids)} (Io, Pf, If, Pr) tuples ready for training.")

  def __len__(self):
    return len(self.sample_ids)

  def __getitem__(self, idx):
    sample_id = self.sample_ids[idx]
    
    # Load images
    npz_path = os.path.join(self.npz_dir, f"{sample_id}.npz")
    frames_data = np.load(npz_path)
  
    io_img = Image.fromarray(frames_data['Io']).convert("RGB")
    if_img = Image.fromarray(frames_data['If']).convert("RGB")
    
    # Load Prompts
    json_path = os.path.join(self.json_dir, f"{sample_id}_meta.json")
    with open(json_path, 'r') as f:
      prompt_data = json.load(f)
    
    pf_text = prompt_data['Pf']
    pr_text = prompt_data['Pr']
    
    # Apply transforms resizing to 512x512 and normalizing
    io_tensor = self.transform(io_img)
    if_tensor = self.transform(if_img)
    
    return {
      "sample_id": sample_id,
      "Io": io_tensor,
      "If": if_tensor,
      "Pf": pf_text,
      "Pr": pr_text
    }