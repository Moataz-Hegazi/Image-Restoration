import os
import torch
import torchvision
from torchvision.io import read_image
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import  torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

MANUAL_SEED = 42
BATCH_SIZE = 32
SHUFFLE = True

class LandscapeDataset(Dataset):
    def __init__(self, transform=None):
        self.dataroot = './Data'
        self.transform = transform
        self.images = os.listdir(f'{self.dataroot}/color')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        color_img = read_image(f'{self.dataroot}/color/{img_path}') /255
        gray_img = read_image(f'{self.dataroot}/gray/{img_path}').float() / 255
        gray_img = gray_img.mean(dim=0, keepdim=True)
        
        if self.transform:
            color_img = self.transform(color_img)
            gray_img = self.transform(gray_img)
        
        return color_img, gray_img
    
transform = transforms.Compose([transforms.Resize((128, 128), antialias=False)])

dataset = LandscapeDataset(transform=transform)

train_set, test_set = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(MANUAL_SEED))

trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

def show_images(color_imgs, gray_imgs):
    fig, axes = plt.subplots(5, 2, figsize=(15, 15))

    axes[0, 0].set_title('Grayscale Image')
    axes[0, 1].set_title('Color Image')
    for i in range(5):
        axes[i, 0].imshow(gray_imgs[i].squeeze(), cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(color_imgs[i].squeeze().permute(1, 2, 0))
        axes[i, 1].axis('off')
    plt.show()

color, gray = next(iter(trainloader))
show_images(color, gray)

EPOCHS = 10
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ColorAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.up1 = nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(512, 128, 3, 2, 1, output_padding=1)
        self.up3 = nn.ConvTranspose2d(256, 64, 3, 2, 1, output_padding=1)
        self.up4 = nn.ConvTranspose2d(128, 3, 3, 2, 1, output_padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        d3 = self.relu(self.down3(d2))
        d4 = self.relu(self.down4(d3))

        u1 = self.relu(self.up1(d4))
        u2 = self.relu(self.up2(torch.cat((u1, d3), dim=1)))
        u3 = self.relu(self.up3(torch.cat((u2, d2), dim=1)))
        u4 = self.sigmoid(self.up4(torch.cat((u3, d1), dim=1)))

        return u4
    
model = ColorAutoEncoder().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {total_params}')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    running_loss = 0.0
    for idx, (color_imgs, gray_imgs) in tqdm(enumerate(trainloader), total = len(trainloader)):
        color_imgs = color_imgs.to(DEVICE) 
        gray_imgs = gray_imgs.to(DEVICE)

        predictions = model(gray_imgs)
        optimizer.zero_grad()

        loss = criterion(color_imgs, predictions)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch: {epoch+1}, loss: {running_loss:.6f}')

total_loss = 0.0
with torch.no_grad():
    for idx, (color_imgs, gray_imgs) in tqdm(enumerate(testloader), total = len(testloader)):
        color_imgs = color_imgs.to(DEVICE) 
        gray_imgs = gray_imgs.to(DEVICE)

        predictions = model(gray_imgs)
        loss = criterion(color_imgs, predictions)
        total_loss += loss.item()
print(f'Test Loss: {total_loss/len(testloader):.6f}')

def show_predictions(color, gray, predictions):
    fig, axes = plt.subplots(5, 3, figsize=(15, 15))

    axes[0, 0].set_title('Grayscale Image')
    axes[0, 1].set_title('Color Image')
    axes[0, 2].set_title('Predicted Color Image')
    for i in range(5):
        axes[i, 0].imshow(gray[i].squeeze(), cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(color[i].squeeze().permute(1, 2, 0))
        axes[i, 1].axis('off')
        axes[i, 2].imshow(predictions[i].squeeze().permute(1, 2, 0))
        axes[i, 2].axis('off')
    plt.show()
show_predictions(color_imgs.detach().cpu(), gray_imgs.detach().cpu(), predictions.detach().cpu())