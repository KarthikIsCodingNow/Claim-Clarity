"""
ClaimClarity AI - Deep Model Training Pipeline
Trains ClaimDCE-Net on real vehicle damage and low-light claim datasets.
Uses Metal Performance Shaders (MPS) on Apple Silicon.
"""

import os
import glob
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np

from model import ClaimDCE_Net, L_spatial, L_exposure, L_color, L_TV


class ClaimDamageDataset(Dataset):
    """
    Dataset loader for vehicle damage & low-light claim photos.
    Applies diverse photometric augmentations (underexposure, sensor noise, shadow falloff)
    to teach the model how to naturally illuminate obscured physical details.
    """
    def __init__(self, image_paths, patch_size=256, is_train=True):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Fallback black image if corrupted
            img = Image.new('RGB', (self.patch_size, self.patch_size), (20, 20, 20))

        # Photometric augmentation: simulate realistic dark garage / shadow falloff
        if self.is_train:
            # Random darkening factor between 0.15 and 0.55
            darken_factor = np.random.uniform(0.18, 0.55)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(darken_factor)

            # Random contrast adjustment
            contrast_enhancer = ImageEnhance.Contrast(img)
            img = contrast_enhancer.enhance(np.random.uniform(0.75, 1.1))

        tensor = self.transform(img)

        # Subtle Poisson-Gaussian noise simulation (high-ISO sensor noise in dark scenes)
        if self.is_train:
            noise = torch.randn_like(tensor) * np.random.uniform(0.01, 0.035)
            tensor = torch.clamp(tensor + noise, 0.0, 1.0)

        return tensor


def train_claim_model(
    epochs=15,
    batch_size=8,
    lr=1e-4,
    save_path="models/claim_dce_model.pth",
    meta_path="models/model_metadata.json"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Gather all dataset images
    train_patterns = [
        "datasets/car_damage/car_damage/train/*.jpg",
        "datasets/car_damage/car_damage/test/*.jpg",
        "static/samples/*.jpg"
    ]
    all_images = []
    for p in train_patterns:
        all_images.extend(glob.glob(p))

    print(f"📦 Total dataset images located: {len(all_images)}")
    if len(all_images) == 0:
        raise ValueError("No images found in dataset directories. Please check datasets/ path.")

    # 2. Setup Device (MPS for Apple Silicon, fallback to CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("⚡ Using Apple Silicon Metal Performance Shaders (MPS) Acceleration!")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU")

    # 3. Model & Loss Functions
    model = ClaimDCE_Net().to(device)
    loss_spa = L_spatial().to(device)
    loss_exp = L_exposure(patch_size=16, target_exp=0.55).to(device)
    loss_col = L_color().to(device)
    loss_tv = L_TV().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # 4. DataLoader
    dataset = ClaimDamageDataset(all_images, patch_size=256, is_train=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"🚀 Starting Deep Curve Training for {epochs} epochs on {len(dataset)} damage images...")
    start_time = time.time()
    history = []

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_spa = 0.0
        epoch_exp = 0.0
        epoch_col = 0.0
        epoch_tv = 0.0
        batches = 0

        for batch_idx, img_batch in enumerate(loader):
            img_batch = img_batch.to(device)

            optimizer.zero_grad()
            enhanced, A = model(img_batch)

            # Compute specialized loss weights
            l_spa = loss_spa(img_batch, enhanced)
            l_exp = 10.0 * loss_exp(enhanced)
            l_col = 5.0 * loss_col(enhanced)
            l_tv = 200.0 * loss_tv(A)

            total_loss = l_spa + l_exp + l_col + l_tv
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_spa += l_spa.item()
            epoch_exp += l_exp.item()
            epoch_col += l_col.item()
            epoch_tv += l_tv.item()
            batches += 1

        avg_loss = round(epoch_loss / batches, 4)
        avg_spa = round(epoch_spa / batches, 4)
        avg_exp = round(epoch_exp / batches, 4)
        avg_tv = round(epoch_tv / batches, 4)

        history.append({
            "epoch": epoch,
            "total_loss": avg_loss,
            "spatial_loss": avg_spa,
            "exposure_loss": avg_exp,
            "tv_smoothness_loss": avg_tv
        })

        if epoch % 2 == 0 or epoch == epochs or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} (Spa: {avg_spa:.4f}, Exp: {avg_exp:.4f}, TV: {avg_tv:.4f})")

    total_time = round(time.time() - start_time, 2)
    print(f"✅ Training completed in {total_time}s!")

    # 5. Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model checkpoint saved to: {save_path}")

    # 6. Save metadata
    metadata = {
        "model_architecture": "ClaimDCE-Net (Deep Curve Estimation Network)",
        "num_iterations": 8,
        "parameter_channels": 24,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trained_epochs": epochs,
        "training_time_seconds": total_time,
        "dataset_samples_count": len(all_images),
        "dataset_name": "Car Damage Vision Benchmark (HF/chittaranjankhatua/car_damage_pub + Synthetic Damaged Pairs)",
        "accelerator": "Apple Silicon Metal Performance Shaders (MPS)" if torch.backends.mps.is_available() else "CPU",
        "loss_weights": {
            "spatial_consistency": 1.0,
            "exposure_control": 10.0,
            "color_constancy": 5.0,
            "illumination_smoothness_tv": 200.0
        },
        "final_loss": history[-1]["total_loss"],
        "history": history
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata


if __name__ == "__main__":
    train_claim_model(epochs=12, batch_size=8, lr=1e-4)
