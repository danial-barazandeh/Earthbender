#!/usr/bin/env python
"""
```
# You might need: pip install "numpy<2.0" scikit-image opencv-python-headless bitsandbytes
accelerate launch train_controlnet_ablation.py \
  --data_root              ./my_dataset \
  --output_dir             ./out_cn_ablation_original_loss \
  --batch_size             4 \
  --max_steps              50000 \
  --validation_every       1000 \
  ...
```
"""
from __future__ import annotations
import argparse, os, math, re
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage import io as skimage_io
import random

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

# ---------------- Advanced Dataset Class (with Validation Split) -----------------
class AdvancedImageDataset(Dataset):
    def __init__(self, root: str | Path, split: str = 'train', val_split_ratio: float = 0.1, res=512):
        root = Path(root)
        self.res = res
        self.split = split

        input_files = {p.stem: p for p in (root / "input").glob("*.png")}
        output_files = {p.stem: p for p in list((root / "output").glob("*.tif")) + list((root / "output").glob("*.tiff"))}
        
        all_stems = sorted(list(set(input_files.keys()) & set(output_files.keys())))
        
        fix_stems = [s for s in all_stems if s.startswith("Fix")]
        normal_stems = [s for s in all_stems if not s.startswith("Fix")]
        
        random.Random(42).shuffle(normal_stems)
        
        split_idx = int(len(normal_stems) * (1 - val_split_ratio))
        train_normal_stems = normal_stems[:split_idx]
        val_stems = normal_stems[split_idx:]
        
        train_stems = train_normal_stems + fix_stems
        
        if self.split == 'train':
            self.stems = train_stems
            print(f"Found {len(self.stems)} training image pairs (including {len(fix_stems)} 'Fix' files).")
        elif self.split == 'val':
            self.stems = val_stems
            print(f"Found {len(self.stems)} validation image pairs.")
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train' or 'val'.")

        self.cond_files = [input_files[stem] for stem in self.stems]
        self.target_files = [output_files[stem] for stem in self.stems]

        if not self.cond_files:
            raise FileNotFoundError(f"No matching image pairs found for split '{self.split}' in {root}.")

        self.color_ranges = {
            'mountains': {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
            'rivers':    {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
            'lakes':     {'lower': np.array([35, 100, 100]), 'upper': np.array([85, 255, 255])}
        }
        
        self.tf_to_tensor = transforms.ToTensor()
        self.tf_resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((res, res), interpolation=transforms.InterpolationMode.BICUBIC),
        ])

    def extract_feature_maps(self, rgb_image):
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        masks = {}
        for feature, color_range in self.color_ranges.items():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            masks[feature] = mask.astype(np.float32) / 255.0
        
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        
        return np.stack([masks['mountains'], masks['rivers'], masks['lakes'], edges], axis=0)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, i):
        try:
            target_img_gray = skimage_io.imread(str(self.target_files[i]))
            if target_img_gray.ndim != 2:
                target_img_gray = cv2.cvtColor(target_img_gray, cv2.COLOR_BGR2GRAY)
            
            target_img_rgb = np.stack([target_img_gray] * 3, axis=-1)
            target_img_resized = self.tf_resize(target_img_rgb)
            target_tensor = self.tf_to_tensor(target_img_resized)
            target_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(target_tensor)

            img_loaded = cv2.imread(str(self.cond_files[i]), cv2.IMREAD_UNCHANGED)
            if img_loaded.ndim == 3 and img_loaded.shape[2] == 4:
                alpha = img_loaded[:, :, 3, np.newaxis] / 255.0
                bgr = img_loaded[:, :, :3]
                background = np.zeros_like(bgr, dtype=bgr.dtype)
                composited_bgr = (bgr * alpha + background * (1 - alpha)).astype(bgr.dtype)
                final_rgb = cv2.cvtColor(composited_bgr, cv2.COLOR_BGR2RGB)
            else:
                final_rgb = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)

            cond_img_resized = self.tf_resize(final_rgb)
            cond_np = np.array(cond_img_resized)
            
            control_features = self.extract_feature_maps(cond_np)
            control_tensor = torch.from_numpy(control_features)

            # Note: lake_mask is not used by the loss in this script, but kept for data consistency
            hsv = cv2.cvtColor(cond_np, cv2.COLOR_RGB2HSV)
            lake_mask_np = cv2.inRange(hsv, self.color_ranges['lakes']['lower'], self.color_ranges['lakes']['upper'])
            lake_mask_tensor = torch.from_numpy(lake_mask_np.astype(np.float32) / 255.0).unsqueeze(0)

            return {"rgb": target_tensor, "control_image": control_tensor, "lake_mask": lake_mask_tensor}
        except Exception as e:
            print(f"Error loading item {i} ({self.cond_files[i]}): {e}")
            raise e

# ------------- Helpers -------------
@torch.no_grad()
def encode_latents(vae, img_batch):
    return vae.encode(img_batch.to(dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor

def plot_loss(filename, title="Loss Plot", **series):
    plt.figure(figsize=(12, 8))
    for label, (steps, values) in series.items():
        if not values: continue
        marker = 'o' if 'Validation' in label else None
        linestyle = '--' if 'Validation' in label else '-'
        plt.plot(steps, values, label=label, marker=marker, linestyle=linestyle, markersize=4)
    plt.xlabel("Training Steps"); plt.ylabel("Loss"); plt.title(title)
    plt.legend(); plt.grid(True, alpha=0.4); plt.savefig(filename); plt.close()

# ------------- Definitive Training Function -------------
def train(cfg):
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, log_with="tensorboard", project_config=ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=Path(cfg.output_dir, "logs")))
    device = accelerator.device
    
    unet = UNet2DConditionModel.from_pretrained(cfg.base_model, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(cfg.base_model, subfolder="vae")

    if cfg.load_controlnet_from:
        controlnet = ControlNetModel.from_pretrained(cfg.load_controlnet_from)
    else:
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4)

    vae.requires_grad_(False); unet.requires_grad_(False)
    if cfg.use_gradient_checkpointing: controlnet.enable_gradient_checkpointing()

    optimizer_cls = bnb.optim.AdamW8bit if cfg.use_8bit_adam and bnb else torch.optim.AdamW
    optimizer = optimizer_cls(controlnet.parameters(), lr=cfg.lr, weight_decay=cfg.adam_weight_decay)
    
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")
    lr_scheduler = get_scheduler(cfg.lr_scheduler, optimizer=optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps)

    train_dataset = AdvancedImageDataset(cfg.data_root, split='train')
    val_dataset = AdvancedImageDataset(cfg.data_root, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    controlnet, optimizer, train_dataloader, val_dataloader, lr_scheduler, unet, vae = accelerator.prepare(
        controlnet, optimizer, train_dataloader, val_dataloader, lr_scheduler, unet, vae)
    ema_controlnet = EMAModel(accelerator.unwrap_model(controlnet).parameters(), decay=cfg.ema_decay) if cfg.use_ema else None

    global_step, ema_loss = 0, None
    loss_history, step_history = [], []
    val_loss_history, val_step_history = [], []
    last_checked_loss = float('inf')
    early_stop_triggered = False
    
    if cfg.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        accelerator.load_state(cfg.resume_from_checkpoint)
        try:
            global_step = int(re.findall(r"\d+", Path(cfg.resume_from_checkpoint).name)[-1])
        except (IndexError, ValueError): pass
        
        history_path = Path(cfg.output_dir) / "loss_history.pt"
        if os.path.exists(history_path):
            history_data = torch.load(history_path)
            loss_history, step_history = history_data.get("train_loss", []), history_data.get("train_steps", [])
            val_loss_history, val_step_history = history_data.get("val_loss", []), history_data.get("val_steps", [])
            last_checked_loss = val_loss_history[-1] if val_loss_history else float('inf')

    progress_bar = tqdm(range(global_step, cfg.max_steps), initial=global_step, total=cfg.max_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(math.ceil(cfg.max_steps / len(train_dataloader))):
        if global_step >= cfg.max_steps: break
        for batch in train_dataloader:
            controlnet.train()
            with accelerator.accumulate(controlnet):
                target_rgb, control_image, _ = batch["rgb"], batch["control_image"], batch["lake_mask"]

                latents = encode_latents(vae, target_rgb)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                prompt_embeds = torch.zeros((latents.shape[0], 77, unet.config.cross_attention_dim), device=device, dtype=unet.dtype)

                down_samples, mid_sample = controlnet(noisy_latents, timesteps, prompt_embeds, control_image, return_dict=False)
                model_pred = unet(noisy_latents, timesteps, prompt_embeds, down_block_additional_residuals=down_samples, mid_block_additional_residual=mid_sample).sample

                # --- Original ControlNet Loss ---
                # The loss is simply the MSE between the predicted noise and the actual noise.
                loss = F.mse_loss(model_pred.float(), noise.float())
                # --- End of Loss Calculation ---

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), 1.0)
                    optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    if ema_controlnet: ema_controlnet.step(controlnet.parameters())
                    
                    global_step += 1; progress_bar.update(1)
                    
                    if accelerator.is_main_process:
                        ema_loss = loss.item() if ema_loss is None else cfg.ema_loss_decay * ema_loss + (1-cfg.ema_loss_decay) * loss.item()
                        loss_history.append(ema_loss); step_history.append(global_step)
                        
                        log_dict = {"train_loss": ema_loss}
                        postfix_dict = {"loss": f"{ema_loss:.4f}"}
                        progress_bar.set_postfix(postfix_dict)
                        
                        if global_step % cfg.validation_every == 0:
                            controlnet.eval()
                            val_losses = []
                            with torch.no_grad():
                                for val_batch in val_dataloader:
                                    val_target_rgb, val_control_image, _ = val_batch["rgb"], val_batch["control_image"], val_batch["lake_mask"]
                                    val_latents = encode_latents(vae, val_target_rgb)
                                    val_noise = torch.randn_like(val_latents)
                                    val_timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (val_latents.shape[0],), device=device).long()
                                    val_noisy_latents = noise_scheduler.add_noise(val_latents, val_noise, val_timesteps)
                                    val_prompt_embeds = torch.zeros((val_latents.shape[0], 77, unet.config.cross_attention_dim), device=device, dtype=unet.dtype)
                                    down, mid = controlnet(val_noisy_latents, val_timesteps, val_prompt_embeds, val_control_image, return_dict=False)
                                    pred = unet(val_noisy_latents, val_timesteps, val_prompt_embeds, down_block_additional_residuals=down, mid_block_additional_residual=mid).sample
                                    val_loss = F.mse_loss(pred.float(), val_noise.float())
                                    val_losses.append(val_loss.item())
                            
                            avg_val_loss = np.mean(val_losses)
                            val_loss_history.append(avg_val_loss)
                            val_step_history.append(global_step)
                            log_dict["val_loss"] = avg_val_loss
                            postfix_dict["val_loss"] = f"{avg_val_loss:.4f}"
                            progress_bar.set_postfix(postfix_dict)
                            
                            plot_loss(
                                Path(cfg.output_dir) / f"loss_plot_step_{global_step}.png",
                                title="Training & Validation Loss (Original)",
                                **{
                                    "Training Loss": (step_history, loss_history),
                                    "Validation Loss": (val_step_history, val_loss_history)
                                }
                            )
                        
                        accelerator.log(log_dict, step=global_step)
                        
                        if global_step > 0 and global_step % cfg.save_checkpoint_every == 0:
                            save_path = Path(cfg.output_dir) / f"checkpoint_step_{global_step}"
                            accelerator.save_state(save_path)
                            controlnet_to_save = accelerator.unwrap_model(controlnet)
                            if ema_controlnet: ema_controlnet.copy_to(controlnet_to_save.parameters())
                            controlnet_to_save.save_pretrained(save_path / "controlnet")
                            torch.save({
                                "train_loss": loss_history, "train_steps": step_history,
                                "val_loss": val_loss_history, "val_steps": val_step_history
                            }, Path(cfg.output_dir) / "loss_history.pt")
                            print(f"\nSaved full checkpoint and inference model to {save_path}")

                        if global_step > 0 and global_step % cfg.check_loss_every == 0 and val_loss_history:
                            current_val_loss = val_loss_history[-1]
                            if current_val_loss > last_checked_loss - (last_checked_loss * cfg.early_stop_threshold):
                                print(f"\nEarly stopping triggered at step {global_step}.")
                                early_stop_triggered = True
                            last_checked_loss = current_val_loss

            if global_step >= cfg.max_steps or early_stop_triggered: break
        if global_step >= cfg.max_steps or early_stop_triggered: break

    if accelerator.is_main_process:
        save_name = "final_model" if not early_stop_triggered else f"model_step_{global_step}_earlystop"
        save_path = Path(cfg.output_dir) / save_name
        model_to_save = accelerator.unwrap_model(controlnet)
        if ema_controlnet: ema_controlnet.copy_to(model_to_save.parameters())
        model_to_save.save_pretrained(save_path)
        print(f"Saved final model to {save_path}")
        plot_loss(
            Path(cfg.output_dir) / "final_loss_plot.png",
            title="Final Training & Validation Loss (Original)",
            **{"Training Loss": (step_history, loss_history), "Validation Loss": (val_step_history, val_loss_history)}
        )
    accelerator.end_training()

# ------------- CLI Argument Parser -------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ablation study script for ControlNet with original loss.")
    # Kept arguments identical for easy comparison, but some are unused by the loss function
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--load_controlnet_from", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--ema_loss_decay", type=float, default=0.95)
    p.add_argument("--save_checkpoint_every", type=int, default=5000)
    p.add_argument("--validation_every", type=int, default=1000, help="Run validation every N steps.")
    p.add_argument("--check_loss_every", type=int, default=10000, help="Check for early stopping every N steps.")
    p.add_argument("--early_stop_threshold", type=float, default=0.001, help="Threshold for early stopping (e.g., 0.001 for 0.1%).")
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--use_gradient_checkpointing", action="store_true")
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    
    # --- The following arguments are ignored by this script but kept for interface consistency ---
    p.add_argument("--lambda_l1", type=float, default=1.0, help="[Ignored in this script]")
    p.add_argument("--lambda_smooth", type=float, default=2.5, help="[Ignored in this script]")
    p.add_argument("--lake_loss_weight", type=float, default=10.0, help="[Ignored in this script]")
    
    cfg = p.parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    train(cfg)
