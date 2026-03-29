import torch
from tqdm import tqdm

def evaluate(model, dataloader, criterion, device):

    total_loss = 0.0

    with torch.no_grad():

        for color_imgs, gray_imgs in tqdm(dataloader):

            color_imgs = color_imgs.to(device)
            gray_imgs = gray_imgs.to(device)

            predictions = model(gray_imgs)

            loss = criterion(predictions, color_imgs)

            total_loss += loss.item()

    return total_loss/len(dataloader)