import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as TF

class LandscapeDataset(Dataset):
    def __init__(self, dataroot, size=(128, 128)):
        self.dataroot = dataroot
        self.size = size
        self.images = os.listdir(f"{self.dataroot}/color")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # -------------------------
        # Load COLOR image (RGB)
        # -------------------------
        color_img = read_image(f"{self.dataroot}/color/{img_name}").float() / 255.0
        # (3, H, W)

        # -------------------------
        # Load GRAYSCALE image
        # -------------------------
        gray_img = read_image(f"{self.dataroot}/gray/{img_name}").float() / 255.0

        # Ensure grayscale is (1, H, W)
        if gray_img.shape[0] == 3:
            gray_img = gray_img.mean(dim=0, keepdim=True)

        # -------------------------
        # Resize (IMPORTANT)
        # -------------------------
        color_img = TF.resize(color_img, self.size)
        gray_img = TF.resize(gray_img, self.size)

        # -------------------------
        # Normalize to [-1, 1]
        # -------------------------
        color_img = (color_img - 0.5) / 0.5
        gray_img = (gray_img - 0.5) / 0.5

        return gray_img, color_img