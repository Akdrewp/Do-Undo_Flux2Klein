import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from dataset import DoUndoDataset
from modelComponents import getLoraPipeline, getOptimizer, getLearningScheduler

# Configuration
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEVICE = "cuda"
BATCH_SIZE = 4 
EPOCHS = 10
LEARNING_RATE = 1e-4
LAMBDA_C = 0.5  # Best as in DoUndo
VAE_SCALE = 0.3611 # Standard FLUX scaling factor (not sure what this is)

def calculate_rectified_flow_loss(transformer, target_latent, cond_latent, prompt_embeds, pooled_embeds):
  """Calculates the MSE with noisy latent between predicted and target velocity
  
  1. Generates noise for batch
  2. Calculate image vector
  3. Predict flow and get MSE loss

  returns:
    torch.Tensor: MSE loss between noisy predicted flow and actual flow
  """

  # 1. Generate noise for vector
  batch_size = target_latent.shape[0]
  t = torch.rand((batch_size,), device=DEVICE, dtype=torch.bfloat16)
  t_expanded = t.view(batch_size, 1, 1, 1)
  
  # 2. Get predicted vector
  # Straight-line interpolation: z_t = t*z1 + (1-t)*z0
  z_0 = torch.randn_like(target_latent)
  z_t = (t_expanded * target_latent) + ((1.0 - t_expanded) * z_0)
  target_flow = target_latent - z_0
  
  # 3. Get flow and calculate MSE loss
  model_input = torch.cat([z_t, cond_latent], dim=1)
  pred_flow = transformer(img=model_input, txt=prompt_embeds, y=pooled_embeds, timesteps=t)
  
  return F.mse_loss(pred_flow, target_flow)

def calculate_consistency_loss(pipe, z_If, Io, prompt_r, VAE_SCALE, DEVICE):
  """Calculates the L1 consistency loss

  1. Generate noise for reverse generation
  2. Predict the undone z_Ir_hat.
  3. Decode latent and compate with original image

  Returns:
    loss_c (torch.Tensor): The L1 distance between the original (Io) pixels and the undone (Ir_hat) pixels.
  """

  # 1. Get random image noise
  batch_size = z_If.shape[0]
  z_noise = torch.randn_like(z_If)
  t_full = torch.ones((batch_size,), device=DEVICE, dtype=torch.bfloat16)

  # 2. Predict undone image
  pred_flow_c = pipe.transformer(
      img=torch.cat([z_noise, z_If], dim=1), 
      txt=prompt_r[0], 
      y=prompt_r[1], 
      timesteps=t_full
  )
  
  z_Ir_hat = z_noise + pred_flow_c
  
  # 3. Decode and calculate loss
  Ir_hat = pipe.vae.decode(z_Ir_hat / VAE_SCALE).sample
  return F.l1_loss(Ir_hat, Io)

def getTotalLoss(pipe, z_Io, z_If, Io, prompt_f, prompt_r, LAMBDA_C, VAE_SCALE, DEVICE):
  """Calculates the total loss for a pass using specified loss function

  1. Calculates Forward MSE
  2. Calculates Reverse MSE
  3. Calculates Consistency

  Returns:
    total_loss (torch.Tensor): The weighted sum of Forward, Reverse, and Consistency losses.
  """
  loss_fwd = calculate_rectified_flow_loss(
      pipe.transformer, z_If, z_Io, prompt_f[0], prompt_f[1]
  )
  loss_rev = calculate_rectified_flow_loss(
      pipe.transformer, z_Io, z_If, prompt_r[0], prompt_r[1]
  )
  loss_c = calculate_consistency_loss(
      pipe, z_If, Io, prompt_r, VAE_SCALE, DEVICE
  )
  total_loss = loss_fwd + loss_rev + (LAMBDA_C * loss_c)
  
  return total_loss, loss_fwd, loss_rev, loss_c

def train():
  """ Finetunes the model for DoUndo with Lora

  1. Loads dataset
  2. Sets up components
  3. Loop for specified epochs
    4. Encode images and prompts
    5. Calulate loss
    6. Backpropagate
  
  """
  # 1. Dataset & Loader
  dataset = DoUndoDataset(npz_dir="processed_tuples_slim", json_dir="final_dataset")
  dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
  
  total_steps = len(dataloader) * EPOCHS

  # 2. Gets Lora compenents
  pipe = getLoraPipeline(MODEL_ID, DEVICE)
  optimizer = getOptimizer(pipe, LEARNING_RATE)
  scheduler = getLearningScheduler(optimizer, total_steps)

  # 3. Training Loop
  print(f"🎬 Starting training for {EPOCHS} epochs...")
  for epoch in range(EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
      optimizer.zero_grad()
      
      # 4. Encode images and prompts with pipeline
      Io = batch["Io"].to(DEVICE, dtype=torch.bfloat16)
      If = batch["If"].to(DEVICE, dtype=torch.bfloat16)
      with torch.no_grad():
        z_Io = pipe.vae.encode(Io).latent_dist.sample() * VAE_SCALE
        z_If = pipe.vae.encode(If).latent_dist.sample() * VAE_SCALE
        
        # Note: You'll need a helper to get embeddings from pipe.text_encoder/2
        # For brevity, assuming text_f/r and pool_f/r are already extracted
        # (You can use pipe.encode_prompt internal method here)
        prompt_f = pipe.encode_prompt(batch["Pf"])
        prompt_r = pipe.encode_prompt(batch["Pr"])

      # 5. Calculate loss
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        (total_loss, 
          loss_fwd, 
          loss_rev, 
          loss_c) = getTotalLoss(
            pipe=pipe,
            z_Io=z_Io,
            z_If=z_If,
            Io=Io,
            prompt_f=prompt_f,
            prompt_r=prompt_r,
            LAMBDA_C=LAMBDA_C,
            VAE_SCALE=VAE_SCALE,
            DEVICE=DEVICE
        )

      # 6. Backpropogate & step
      total_loss.backward()
      torch.nn.utils.clip_grad_norm_(pipe.transformer.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

      # Stats for logging
      epoch_loss += total_loss.item()
      progress_bar.set_postfix({
        "fwd": f"{loss_fwd.item():.3f}",
        "rev": f"{loss_rev.item():.3f}",
        "con": f"{loss_c.item():.3f}"
      })

    # Save Checkpoint
    avg_loss = epoch_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")
    pipe.transformer.save_pretrained(f"checkpoints/doundo_flux_epoch_{epoch+1}")

if __name__ == "__main__":
  train()