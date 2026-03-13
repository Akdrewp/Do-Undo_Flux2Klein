import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

def getLoraPipeline(model_id: str, device: str):
  """ Sets up the FLUX pipeline with LoRA injected for physics-aware training.

  1. Loads the pre-trained DiffusionPipeline.
  2. Freezes VAE, Text Encoders, and base Transformer weights.
  3. Configures and injects LoRA layers into the Transformer attention blocks.
  4. Sets the sub-modules to the correct training (Transformer) or eval (VAE/Encoders) modes.

  Args:
    model_id (str): The identifier of the pre-trained model to load from Hugging Face.
    device (str): The hardware device to run the model on (e.g., "cuda").

  Returns:
    pipe (DiffusionPipeline): The complete FLUX pipeline with a trainable LoRA-augmented transformer.
  """
  
  
  # 1. Load model
  pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

  # 2. Freeze components
  pipe.vae.requires_grad_(False)
  pipe.text_encoder.requires_grad_(False)
  pipe.transformer.requires_grad_(False)
  
  # pipe.transformer.enable_gradient_checkpointing()

  # 3. Get Lora PEFT transformer
  lora_config = LoraConfig(
    r=1024,
    lora_alpha=512,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "add_k_proj", "add_q_proj", "add_v_proj"],
    init_lora_weights="gaussian"
  )
  pipe.transformer = get_peft_model(pipe.transformer, lora_config)
  
  # 4. Set modes to training
  pipe.transformer.train()
  pipe.vae.eval()
  pipe.text_encoder.eval()
  
  pipe.transformer.print_trainable_parameters()
  return pipe

def getOptimizer(pipe, lr: float):
  """ Initializes the AdamW optimizer for the trainable LoRA parameters.

  1. Filters the Transformer parameters to identify those requiring gradients.
  2. Initializes the AdamW optimizer with weight decay to prevent over-fitting.

  Args:
    pipe (DiffusionPipeline): The pipeline containing the LoRA-augmented transformer.
    lr (float): The learning rate for weight updates.

  Returns:
    optimizer (torch.optim.AdamW): The optimizer instance managing the LoRA parameter updates.
  """
  optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, pipe.transformer.parameters()),
    lr=lr,
    weight_decay=1e-4
  )
  return optimizer

def getLearningScheduler(optimizer, num_training_steps: int):
  """ Sets up a cosine learning rate scheduler with a linear warmup phase.

  1. Calculates the warmup period (10% of total steps for now).
  2. Returns a scheduler that ramps up and then decays the learning rate according to a cosine curve.

  Args:
    optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
    num_training_steps (int): Total number of optimization steps in the training run.

  Returns:
    lr_scheduler (LRScheduler): The Hugging Face scheduler object for dynamic LR management.
  """
  warmup_steps = int(num_training_steps * 0.1)
  lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
  )
  return lr_scheduler