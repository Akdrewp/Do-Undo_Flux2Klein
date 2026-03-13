import torch
import torch.nn.functional as F

def calculate_rectified_flow_loss(model, target_latent, cond_latent, text_embeds, pooled_text_embeds):
  """
  Calculates the MSE vector-field loss for a Rectified Flow model (FLUX).
  
  target_latent: The latent of the image we want to generate (z_1)
  cond_latent: The latent of the clean input image we condition on
  """
  batch_size = target_latent.shape[0]
  device = target_latent.device
  dtype = target_latent.dtype
  
  # FLUX uses continuous timesteps rather than discrete 1-1000 steps
  t = torch.rand((batch_size,), device=device, dtype=dtype)
  
  # Reshape t for broadcasting against the latent tensor [B, C, H, W]
  t_expanded = t.view(batch_size, 1, 1, 1)
  
  # 2. Sample pure Gaussian noise (z_0)
  z_0 = torch.randn_like(target_latent)
  
  # 3. Create the noisy latent z_t via straight-line interpolation
  # Formula: z_t = t * z_1 + (1 - t) * z_0
  z_1 = target_latent
  z_t = (t_expanded * z_1) + ((1.0 - t_expanded) * z_0)
  
  # 4. The target vector field (velocity) is the straight line from noise to data
  target_flow = z_1 - z_0
  
  # 5. Combine the Noisy Target and the Clean Condition
  # For Image-to-Image tasks in FLUX, you typically concatenate the noisy 
  # latent and the condition latent along the channel dimension.
  # (Assuming 16 channels each -> 32 channel input to the Transformer)
  model_input = torch.cat([z_t, cond_latent], dim=1)
  
  # 6. Predict the flow using the FLUX transformer
  # (Note: parameter names might vary slightly depending on your specific FLUX wrapper)
  predicted_flow = model(
    img=model_input,
    txt=text_embeds,
    y=pooled_text_embeds,
    timesteps=t 
  )
  
  # 7. Calculate the Mean Squared Error between predicted and target velocity
  loss = F.mse_loss(predicted_flow, target_flow)
  
  return loss