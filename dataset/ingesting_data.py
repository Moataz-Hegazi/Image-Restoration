import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

class LandscapeDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        self.dataroot = dataroot
        self.transform = transform
        self.images = os.listdir(f"{self.dataroot}/color")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        color_img = read_image(f"{self.dataroot}/color/{img_path}").float()/255

        gray_img = read_image(f"{self.dataroot}/gray/{img_path}").float()/255
        gray_img = gray_img.mean(dim=0, keepdim=True)

        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)

        return color_img, gray_img