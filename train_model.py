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
BATCH_SIZE = 2 
EPOCHS = 10
LEARNING_RATE = 1e-5
LAMBDA_C = 25  # 0.5 is reccomended in DoUndo but with training set too much hallucinations happened
VAE_SCALE = 0.3611 # Standard FLUX scaling factor (not sure what this is)


compressor = torch.nn.Linear(256, 128).to(DEVICE, dtype=torch.bfloat16)

def pack_latents(latents):
  """Packs a 4D latent tensor into a 3D sequence for the FLUX transformer."""
  batch_size, num_channels, height, width = latents.shape
  # Chop into 2x2 patches
  latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
  latents = latents.permute(0, 2, 4, 1, 3, 5)
  # Flatten into sequence: (Batch, Sequence_Length, Patch_Features)
  latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
  return latents

def unpack_latents(latents, height, width, num_channels):
  """Unpacks a 3D sequence back into a 4D latent image for the VAE."""
  batch_size = latents.shape[0]
  latents = latents.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
  latents = latents.permute(0, 3, 1, 4, 2, 5)
  latents = latents.reshape(batch_size, num_channels, height, width)
  return latents

def prepare_latent_ids(batch_size, height, width, device):
    # Flux uses 2x2 patching, so dimensions are halved
    latent_h, latent_w = height // 2, width // 2
    
    # Create a grid of coordinates - FLUX.2 needs 4 channels here!
    ids = torch.zeros(latent_h, latent_w, 4)
    ids[..., 1] = torch.arange(latent_h)[:, None]
    ids[..., 2] = torch.arange(latent_w)
    # ids[..., 3] is often used for multi-image or video; 
    # setting to 0 works for static images.
    
    ids = ids.reshape(1, -1, 4).repeat(batch_size, 1, 1)
    return ids.to(device, dtype=torch.bfloat16)

def prepare_text_ids(batch_size, seq_len, device):
    # Text IDs also need to match the 4-channel requirement
    return torch.zeros(batch_size, seq_len, 4).to(device, dtype=torch.bfloat16)

def calculate_rectified_flow_loss(transformer, target_latent, cond_latent, prompt_embeds):
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
  packed_input = pack_latents(model_input) 
  packed_target_flow = pack_latents(target_flow)
  compressed_input = compressor(packed_input) 

  img_ids = prepare_latent_ids(batch_size, 64, 64, DEVICE) 
  txt_ids = prepare_text_ids(batch_size, prompt_embeds.shape[1], DEVICE)

  pred_flow = transformer(
      hidden_states=compressed_input, 
      encoder_hidden_states=prompt_embeds, 
      timestep=t,
      img_ids=img_ids,
      txt_ids=txt_ids,
  )[0]

  return F.mse_loss(pred_flow, packed_target_flow)

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

  model_input = torch.cat([z_noise, z_If], dim=1)
  packed_input = pack_latents(model_input)
  compressed_input = compressor(packed_input)

  img_ids = prepare_latent_ids(batch_size, 64, 64, DEVICE)
  txt_ids = prepare_text_ids(batch_size, prompt_r[0].shape[1], DEVICE)
  guidance = torch.full((batch_size,), 3.5, device=DEVICE, dtype=torch.bfloat16)

  # 2. Predict undone image
  pred_flow_c = pipe.transformer(
    hidden_states=compressed_input, 
    encoder_hidden_states=prompt_r[0], 
    timestep=t_full,
    img_ids=img_ids,
    txt_ids=txt_ids,
    guidance=guidance,
    return_dict=False
  )[0]
  
  # 4. Apply flow and UNPACK
  # z_noise is [B, 32, 64, 64], so we must pack it to add to the flow, then unpack
  packed_z_noise = pack_latents(z_noise)
  z_Ir_hat_packed = packed_z_noise + pred_flow_c
  
  # Return to [B, 32, 64, 64] for the VAE
  z_Ir_hat = unpack_latents(z_Ir_hat_packed, 64, 64, 32)
  
  # 5. Decode and calculate loss
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
    pipe.transformer, z_If, z_Io, prompt_f[0]
  )
  loss_rev = calculate_rectified_flow_loss(
    pipe.transformer, z_Io, z_If, prompt_r[0]
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
  dataset = DoUndoDataset(npz_dir="static_tuples_slim", json_dir="static_final_dataset")
  dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,       
    pin_memory=True,     
    prefetch_factor=2    
  )
  
  total_steps = len(dataloader) * EPOCHS

  # 2. Gets Lora compenents
  pipe = getLoraPipeline(MODEL_ID, DEVICE)
  optimizer = getOptimizer(pipe, LEARNING_RATE)
  scheduler = getLearningScheduler(optimizer, total_steps)

  # Start from latest epoch
  pipe.transformer.load_adapter("./checkpoints_7static_1024rank/doundo_flux_epoch_2/", adapter_name="default")

  # 3. Training Loop
  print(f"🎬 Starting training for {EPOCHS} epochs...")
  for epoch in range(2, EPOCHS):
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

      if step % 10 == 0:
        print(f"Epoch {epoch+1} | Step {step} \
        | Total Loss: {total_loss.item():.4f} \
        | Fwd: {loss_fwd.item():.4f} \
        | Rev: {loss_rev.item():.4f} \
        | Con: {loss_c.item():.4f}")
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
    pipe.transformer.save_pretrained(f"checkpoints_7static_1024rank/doundo_flux_epoch_{epoch+1}")

if __name__ == "__main__":
  train()