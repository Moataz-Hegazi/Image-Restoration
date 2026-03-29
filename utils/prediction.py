import torch

def predict_batch(model, dataloader, device):

    model.eval()

    color, gray = next(iter(dataloader))

    color = color.to(device)
    gray = gray.to(device)

    with torch.no_grad():
        predictions = model(gray)

    return color.cpu(), gray.cpu(), predictions.cpu()