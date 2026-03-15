import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torchvision import transforms
import torchvision.transforms.functional as Ft
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

# Import your custom modules
from dataset import DoUndoDataset
from modelComponents import getLoraPipeline, getOptimizer, getLearningScheduler

# Configuration
MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
DEVICE = "cuda"
BATCH_SIZE = 2 
EPOCHS = 10
LEARNING_RATE = 8e-6 # Smaller so I don't have to change it as the epochs go on
LAMBDA_C = 0.8  # 0.5 is reccomended in DoUndo but with training set too much hallucinations happened

def calculate_rectified_flow_loss(pipe, target_latent, input_latent, prompt_embeds, text_ids):
  """Calculates the MSE loss for FLUX.2 native Rectified Flow with Image Conditioning.
    
  1. Sample timesteps for focus on middle of generation.
  2. Patchify and normalize image latents
  3. Generate spatial coordinate
  4. Generate noise and interpolate the target latent
  5. Pack the latents concatenate the noisy target with the clean condition
  6. Predict the flow velocity vector using the transformer.
  7. Remove input tokens from the output and unpack the prediction back to a 2D spatial grid
  8. Calculate MSE loss.
  9. Convert the predicted velocity vector back into an estimated image using Euler integration.

  Returns:
      tuple: (MSE Loss, Predicted Target Image Latent, Clean Patched Target Latent)
    """
  batch_size = target_latent.shape[0]
  
  # 1. Sample logit-normal timesteps
  # The beginning and end aren't as important as the middle
  # Sigmoid gives a larger distribution within the middle
  # of the generation
  u = torch.randn((batch_size,), device=DEVICE, dtype=torch.bfloat16)
  t = torch.sigmoid(u)
  t_expanded = t.view(batch_size, 1, 1, 1)

  # 2. Patchify and normalize image latents
  target_latent_patched = pipe._patchify_latents(target_latent)
  input_latent_patched = pipe._patchify_latents(input_latent)

  latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(DEVICE)
  latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + 1e-5).to(DEVICE)
  target_latent_patched = (target_latent_patched - latents_bn_mean) / latents_bn_std
  input_latent_patched = (input_latent_patched - latents_bn_mean) / latents_bn_std

  # 3. Generate spatial coordinate IDs
  model_img_ids = pipe._prepare_latent_ids(target_latent_patched).to(device=DEVICE)
  cond_model_input_list = [input_latent_patched[i].unsqueeze(0) for i in range(input_latent_patched.shape[0])]
  cond_model_input_ids = pipe._prepare_image_ids(cond_model_input_list).to(DEVICE)
  cond_model_input_ids = cond_model_input_ids.view(
      input_latent_patched.shape[0], -1, model_img_ids.shape[-1]
  )

  # 4. Generate noise and interpolate
  # To create the noisy state for rectified flow
  z_0 = torch.randn_like(target_latent_patched)
  noisy_model_input = (1.0 - t_expanded) * target_latent_patched + t_expanded * z_0

  # 5. Pack into 1D sequences and concatenate
  packed_noisy_model_input = pipe._pack_latents(noisy_model_input)
  packed_cond_model_input = pipe._pack_latents(input_latent_patched)
  
  orig_input_shape = packed_noisy_model_input.shape
  orig_input_ids_shape = model_img_ids.shape

  packed_noisy_model_input = torch.cat([packed_noisy_model_input, packed_cond_model_input], dim=1)
  model_input_ids = torch.cat([model_img_ids, cond_model_input_ids], dim=1)
  
  guidance = torch.full((batch_size,), 3.5, device=DEVICE, dtype=torch.bfloat16)

  # 6. Predict the flow velocity vector
  pred_flow = pipe.transformer(
      hidden_states=packed_noisy_model_input, 
      encoder_hidden_states=prompt_embeds,
      timestep=t,
      img_ids=model_input_ids,
      txt_ids=text_ids,
      guidance=guidance,
      return_dict=False
  )[0]

  # 7. Prune condition tokens and unpack to 2D
  model_pred = pred_flow[:, : orig_input_shape[1], :]
  model_input_ids_pruned = model_input_ids[:, : orig_input_ids_shape[1], :]

  model_pred = pipe._unpack_latents_with_ids(model_pred, model_input_ids_pruned)

  # 8. Calculate ground truth flow and MSE Loss 
  # between predicted and original image
  target_flow = z_0 - target_latent_patched
  loss_mse = F.mse_loss(model_pred.float(), target_flow.float(), reduction="mean")

  # 9. Convert velocity vector to estimated image
  predicted_target = noisy_model_input - (t_expanded * model_pred)
  return loss_mse, predicted_target, target_latent_patched


def getTotalLoss(pipe, z_Io, z_If, prompt_f, prompt_r, LAMBDA_C, DEVICE):
  """ Calculates the total loss for a pass using specified loss function.
  """
  # 1. Forward Pass (Generate If_hat)
  loss_fwd, pred_If, If_original = calculate_rectified_flow_loss(
      pipe=pipe, 
      target_latent=z_If,
      input_latent = z_Io,
      prompt_embeds=prompt_f[0], 
      text_ids=prompt_f[1]
  )
  # 2. Reverse Pass (Generate Io_hat)
  loss_rev, pred_Io, Io_original = calculate_rectified_flow_loss(
      pipe=pipe, 
      target_latent=z_Io,
      input_latent = z_If,
      prompt_embeds=prompt_r[0], 
      text_ids=prompt_r[1]
  )
  
  # 3. Consistency Loss
  # L1 difference of Io_hat and Io
  loss_c = loss_c = F.l1_loss(pred_Io.float(), Io_original.float(), reduction="mean")

  # 4. Total Loss Calculation
  total_loss = loss_fwd + loss_rev + (LAMBDA_C * loss_c)
  
  return total_loss, loss_fwd, loss_rev, loss_c


def normalize_image_tensor(image_tensor, target_size=(512, 512)):
  """Resizes a tensor to target dimensions normalizes pixel values to a [-1, 1] range.
  
  returns: 
    A resized and normalized image tensor
  """
  image = Ft.resize(
      image_tensor, 
      size=target_size, 
      interpolation=Ft.InterpolationMode.BILINEAR,
      antialias=True # Recommended for downscaling tensors
  )
  image = Ft.normalize(image, [0.5], [0.5])
  
  return image

def train(resume_epoch=None):
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
  
  total_steps = (len(dataloader) * EPOCHS)

  # 2. Gets Lora compenents
  pipe = getLoraPipeline(MODEL_ID, DEVICE)
  optimizer = getOptimizer(pipe, LEARNING_RATE)
  scheduler = getLearningScheduler(optimizer, total_steps)

  start_epoch = 0
  if resume_epoch is not None:
    ckpt_dir = f"checkpoints/epoch_{resume_epoch}"
    state_path = os.path.join(ckpt_dir, "training_state.pt")
    
    if os.path.exists(state_path):
      print(f"Resuming from Epoch {resume_epoch}")
      
      # Load the transformer weights
      pipe.transformer.from_pretrained(ckpt_dir)
      
      # Load the optimizer and scheduler memory
      checkpoint = torch.load(state_path, map_location=DEVICE)
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      
      start_epoch = checkpoint['epoch']
      print("Successfully restored optimizer and scheduler states")
    else:
      print(f"Checkpoint {ckpt_dir} not found. Starting from scratch.")

  steps_per_epoch = len(dataloader)
  print("total steps = ", steps_per_epoch)
  # 3. Training Loop
  print(f"Starting training for {EPOCHS} epochs...")
  for epoch in range(0, EPOCHS):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for step, batch in enumerate(progress_bar):
      if step >= steps_per_epoch:
        break
      optimizer.zero_grad()
      
      # 4. Encode images and prompts with pipeline
      Io = batch["Io"].to(DEVICE, dtype=torch.bfloat16)
      If = batch["If"].to(DEVICE, dtype=torch.bfloat16)
      Io = normalize_image_tensor(Io).to(DEVICE, dtype=torch.bfloat16)
      If = normalize_image_tensor(If).to(DEVICE, dtype=torch.bfloat16)
      with torch.no_grad():
        z_Io = pipe.vae.encode(Io).latent_dist.mode() 
        z_If = pipe.vae.encode(If).latent_dist.mode()
        
        prompt_f = pipe.encode_prompt(batch["Pf"], max_sequence_length=300)
        prompt_r = pipe.encode_prompt(batch["Pr"], max_sequence_length=300)

      # 5. Calculate loss
      with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        (total_loss, 
          loss_fwd, 
          loss_rev, 
          loss_c) = getTotalLoss(
            pipe=pipe,
            z_Io=z_Io,
            z_If=z_If,
            prompt_f=prompt_f,
            prompt_r=prompt_r,
            LAMBDA_C=LAMBDA_C,
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

    avg_loss = epoch_loss / steps_per_epoch
    print(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")

    save_epoch(pipe, optimizer, scheduler, epoch, avg_loss)

def save_epoch(pipe, optimizer, scheduler, epoch, avg_loss, base_dir="checkpoints5"):
  """Saves model weights and training state for the current epoch at specified directory
  
  Saves the data in a folder name epoch_{epoch}
  with files from transformer.save_pretrained
  
  And a state dict
  {
   epoch: The epoch number
   optimizer_state_dict: Optimizer state
   scheduler_state_dict: Learning rate scheduler state
   loss: Average loss of epoch
  }
  """
  save_dir = os.path.join(base_dir, f"epoch_{epoch+1}")
  os.makedirs(save_dir, exist_ok=True)
  
  # Save the actual LoRA weights
  pipe.transformer.save_pretrained(save_dir)
  
  # Save the PyTorch training state
  state_dict = {
      'epoch': epoch + 1,
      'optimizer_state_dict': optimizer.state_dict(),
      'scheduler_state_dict': scheduler.state_dict(),
      'loss': avg_loss,
  }
  torch.save(state_dict, os.path.join(save_dir, "training_state.pt"))
  print(f"Saved weights and optimizer state to {save_dir}")

if __name__ == "__main__":
  train()