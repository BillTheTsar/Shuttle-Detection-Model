import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision.transforms.functional as F


# =========================
# Helper Functions
# =========================

def apply_jitter(frames, color_jitter, jitter_enabled=True):
    """
    Apply SAME color jitter to all frames by sampling one set of params.
    """
    if jitter_enabled:
        brightness = random.uniform(*color_jitter.brightness) if color_jitter.brightness else 1
        contrast = random.uniform(*color_jitter.contrast) if color_jitter.contrast else 1
        saturation = random.uniform(*color_jitter.saturation) if color_jitter.saturation else 1
        hue = random.uniform(*color_jitter.hue) if color_jitter.hue else 0

        jittered = []
        for img in frames:
            img = F.adjust_brightness(img, brightness)
            img = F.adjust_contrast(img, contrast)
            img = F.adjust_saturation(img, saturation)
            img = F.adjust_hue(img, hue)
            jittered.append(img)
        return jittered
    else:
        return frames

def apply_noise(frames, noise_std=0.01):
    """
    Apply different Gaussian noise per frame.
    Takes in a list of images and returns a list of tensors.
    """
    noisy_frames = []
    for img in frames:
        img_tensor = transforms.ToTensor()(img)
        noise = torch.randn_like(img_tensor) * noise_std
        img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        noisy_frames.append(img_tensor)
    return noisy_frames

def generate_crop_box(gt_pos, gt_vis, img_w, img_h, crop_size, train_mode=True):
    """
    Compute crop box for Stage 2.
    train_mode=True: uses GT + randomness.
    train_mode=False: uses inferred_target directly (passed as gt_pos here).
    """
    if train_mode:
        if gt_vis == 1:
            if random.random() < 0.8:  # 80% center around GT
                cx, cy = gt_pos
                offset_x = random.uniform(-0.35 * crop_size, 0.35 * crop_size)
                offset_y = random.uniform(-0.35 * crop_size, 0.35 * crop_size)
                center_x = cx * img_w + offset_x
                center_y = cy * img_h + offset_y
            else:
                center_x = random.uniform(0, img_w)
                center_y = random.uniform(0, img_h)
        else:  # gt_vis == 0 → random crop
            center_x = random.uniform(0, img_w)
            center_y = random.uniform(0, img_h)
    else:
        center_x, center_y = gt_pos[0] * img_w, gt_pos[1] * img_h

    x_min = int(center_x - crop_size / 2)
    y_min = int(center_y - crop_size / 2)
    x_max = x_min + crop_size
    y_max = y_min + crop_size
    return x_min, y_min, x_max, y_max

def crop_with_padding(img, x_min, y_min, x_max, y_max, crop_size):
    """
    Crop with black padding if crop exceeds image bounds.
    Returns a square crop of size crop_size.
    """
    img_w, img_h = img.size
    crop = Image.new("RGB", (crop_size, crop_size), (0, 0, 0))

    # Calculate overlap
    overlap_x_min = max(0, x_min)
    overlap_y_min = max(0, y_min)
    overlap_x_max = min(img_w, x_max)
    overlap_y_max = min(img_h, y_max)

    if overlap_x_min < overlap_x_max and overlap_y_min < overlap_y_max:
        img_crop = img.crop((overlap_x_min, overlap_y_min, overlap_x_max, overlap_y_max))
        paste_x = overlap_x_min - x_min
        paste_y = overlap_y_min - y_min
        crop.paste(img_crop, (paste_x, paste_y))
    return crop

def adjust_positions_for_crop(positions, x_min, y_min, crop_size, img_w, img_h):
    """
    Normalize positions relative to crop.
    If point is outside crop, visibility = 0.
    The convention is that we pass in positions in the order of -30, -29, ... , -2, -1.
    Thus, we do not need to reverse the order again
    """
    threshold = 0.3
    adjusted = []
    for (x, y, vis) in positions:
        if vis < threshold: # If the shuttle is not visible enough
            adjusted.append((0, 0, 0))
            continue
        abs_x, abs_y = x * img_w, y * img_h
        if x_min <= abs_x < x_min + crop_size and y_min <= abs_y < y_min + crop_size: # Visible and in crop
            rel_x = (abs_x - x_min) / crop_size
            rel_y = (abs_y - y_min) / crop_size
            adjusted.append((rel_x, rel_y, vis))
        else:
            adjusted.append((0, 0, 0)) # Visible but not in crop
    return adjusted


# =========================
# Stage 1 Dataset
# =========================

class Stage1Dataset(Dataset):
    def __init__(self, root_dir, index_file, split="train", train_mode = True, apply_flip=True, jitter_enabled=True, noise_std=0.01):
        self.root_dir = Path(root_dir)
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.sample_dirs = index_data[split]
        self.train_mode = train_mode
        self.apply_flip = apply_flip if train_mode else False # Do not flip unless training
        self.color_jitter = transforms.ColorJitter(0.3, 0.2, 0.2, 0.05)
        self.noise_std = noise_std if train_mode else 0.0 # Disable noise in inference
        self.jitter_enabled = jitter_enabled if train_mode else False  # Disable jitter in inference
        self.resize = transforms.Resize((300, 300))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.root_dir / self.sample_dirs[idx]
        current_img = Image.open(sample_path / "frame.jpg").convert("RGB")
        past_imgs = [Image.open(sample_path / f"frame-{i}.jpg").convert("RGB") for i in range(3, 0, -1)]

        jittered_imgs = apply_jitter([current_img] + past_imgs, self.color_jitter, self.jitter_enabled)

        resized_current = self.resize(jittered_imgs[0])
        resized_past = [self.resize(img) for img in jittered_imgs[1:]]
        current_tensor = apply_noise([resized_current], self.noise_std)[0]
        past_tensor = torch.stack(apply_noise(resized_past, self.noise_std))

        positions = pd.read_csv(sample_path / "positions.csv").values[::-1].copy()
        positions_tensor = torch.tensor(positions, dtype=torch.float32)


        target = pd.read_csv(sample_path / "target.csv").iloc[0]
        target_tensor = torch.tensor([target.shuttle_x, target.shuttle_y, target.shuttle_visibility], dtype=torch.float32)

        # Horizontal flip
        if self.apply_flip and random.random() < 0.5:
            current_tensor = torch.flip(current_tensor, dims=[2])
            past_tensor = torch.flip(past_tensor, dims=[3])
            positions_tensor[:, 0] = 1.0 - positions_tensor[:, 0]
            target_tensor[0] = 1.0 - target_tensor[0]

        return {
            "current_img": current_tensor,
            "past_imgs": past_tensor,
            "positions": positions_tensor,
            "target": target_tensor
        }


# ==============================
# Stage 2 Dataset
# ==============================

class Stage2Dataset(Dataset):
    def __init__(self, root_dir, index_file, split="train", train_mode=True, inferred_target=None,
                 apply_flip=True, jitter_enabled=True, noise_std=0.01):
        self.root_dir = Path(root_dir)
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        self.sample_dirs = index_data[split]

        self.train_mode = train_mode
        self.inferred_target = inferred_target
        self.apply_flip = apply_flip if train_mode else False
        self.color_jitter = transforms.ColorJitter(0.3, 0.2, 0.2, 0.05)
        self.noise_std = noise_std if train_mode else 0.0 # Disable noise in inference
        self.jitter_enabled = jitter_enabled if train_mode else False  # Disable jitter in inference
        self.final_resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.root_dir / self.sample_dirs[idx]
        current_img = Image.open(sample_path / "frame.jpg").convert("RGB")
        past_imgs = [Image.open(sample_path / f"frame-{i}.jpg").convert("RGB") for i in range(3, 0, -1)]

        positions = pd.read_csv(sample_path / "positions.csv").values[::-1].copy()
        if self.train_mode:
            target = pd.read_csv(sample_path / "target.csv").iloc[0]
            gt_x, gt_y, gt_vis = target.shuttle_x, target.shuttle_y, target.shuttle_visibility
        else:
            if self.inferred_target is None:
                raise ValueError("inferred_target must be provided for inference mode")
            gt_x, gt_y, gt_vis = self.inferred_target


        jittered_imgs = apply_jitter([current_img] + past_imgs, self.color_jitter, self.jitter_enabled)
        current_img, past_imgs = jittered_imgs[0], jittered_imgs[1:]

        img_w, img_h = current_img.size
        crop_size = random.choice([224, 300, 400]) if self.train_mode else 224

        x_min, y_min, x_max, y_max = generate_crop_box((gt_x, gt_y), gt_vis, img_w, img_h, crop_size, train_mode=self.train_mode)
        cropped_current = crop_with_padding(current_img, x_min, y_min, x_max, y_max, crop_size)
        cropped_past = [crop_with_padding(img, x_min, y_min, x_max, y_max, crop_size) for img in past_imgs]

        resized_current = self.final_resize(cropped_current)
        resized_past = [self.final_resize(img) for img in cropped_past]

        current_tensor = apply_noise([resized_current], self.noise_std)[0]
        past_tensor = torch.stack(apply_noise(resized_past, self.noise_std))

        adjusted_positions = adjust_positions_for_crop(positions, x_min, y_min, crop_size, img_w, img_h)
        positions_tensor = torch.tensor(adjusted_positions, dtype=torch.float32)
        adjusted_target = adjust_positions_for_crop([[gt_x, gt_y, gt_vis]], x_min, y_min, crop_size, img_w, img_h)[0]
        target_tensor = torch.tensor(adjusted_target, dtype=torch.float32)

        # Horizontal flip
        if self.apply_flip and random.random() < 0.5:
            current_tensor = torch.flip(current_tensor, dims=[2])
            past_tensor = torch.flip(past_tensor, dims=[3])
            positions_tensor[:, 0] = 1.0 - positions_tensor[:, 0]
            target_tensor[0] = 1.0 - target_tensor[0]

        return {
            "current_crop": current_tensor,
            "past_crops": past_tensor,
            "positions": positions_tensor,
            "target": target_tensor
        }


# class ShuttleDataset(Dataset):
#     def __init__(self, root_dir, index_file, split="train", apply_flip=True):
#         self.root_dir = Path(root_dir)
#         with open(index_file, 'r') as f:
#             index_data = json.load(f)
#         self.sample_dirs = index_data[split]
#
#         self.apply_flip = apply_flip
#
#         # Resize transforms
#         self.current_frame_transform = transforms.ToTensor()  # Keep original resolution
#         self.past_frame_transform = transforms.Compose([
#             transforms.Resize((270, 480)),  # Downscale while preserving 16:9 ratio
#             transforms.ToTensor()
#         ])
#
#         # Define color jitter parameters (but apply consistently per sample)
#         self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2,
#                                                   saturation=0.2, hue=0.05)
#
#         self.gaussian_noise_std = 0.01  # For normalized [0,1] pixel values
#
#     def __len__(self):
#         return len(self.sample_dirs)
#
#     def __getitem__(self, idx):
#         sample_path = self.root_dir / Path(self.sample_dirs[idx])
#         seed = random.randint(0, 9999)
#         random.seed(seed)
#
#         # Load and transform current frame
#         current_img = Image.open(sample_path / "frame.jpg").convert("RGB")
#         current_img = self.color_jitter(current_img)  # Apply same jitter
#         current_tensor = self.current_frame_transform(current_img)
#
#         # Add Gaussian noise (independent for this frame)
#         current_tensor += torch.randn_like(current_tensor) * self.gaussian_noise_std
#         current_tensor = torch.clamp(current_tensor, 0, 1)
#
#         # Load past frames (3 → 1)
#         past_frames = []
#         for i in range(3, 0, -1):  # oldest last
#             random.seed(seed) # Resets the seed
#             img = Image.open(sample_path / f"frame-{i}.jpg").convert("RGB")
#             img = self.color_jitter(img)  # Same jitter for consistency
#             img_tensor = self.past_frame_transform(img)
#             img_tensor += torch.randn_like(img_tensor) * self.gaussian_noise_std  # Independent noise
#             img_tensor = torch.clamp(img_tensor, 0, 1)
#             past_frames.append(img_tensor)
#         past_frames_tensor = torch.stack(past_frames)  # Shape: [3, 3, H, W]
#
#         # Load positions (reverse for chronological order)
#         positions = pd.read_csv(sample_path / "positions.csv").values[::-1].copy()
#         positions_tensor = torch.tensor(positions, dtype=torch.float32)
#
#         # Load target
#         target = pd.read_csv(sample_path / "target.csv").iloc[0]
#         target_tensor = torch.tensor([target.shuttle_x, target.shuttle_y, target.shuttle_visibility],
#                                      dtype=torch.float32)
#
#         # Horizontal flip (50% chance if enabled)
#         if self.apply_flip and random.random() < 0.5:
#             current_tensor = torch.flip(current_tensor, dims=[2])
#             past_frames_tensor = torch.flip(past_frames_tensor, dims=[3])
#             positions_tensor[:, 0] = 1.0 - positions_tensor[:, 0]
#             target_tensor[0] = 1.0 - target_tensor[0]
#
#         return {
#             "current_img": current_tensor,        # [3, 1080, 1920]
#             "past_frames": past_frames_tensor,    # [3, 3, 270, 480]
#             "positions": positions_tensor,        # [30, 3]
#             "target": target_tensor               # [3]
#         }
