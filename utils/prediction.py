import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter


def predict_batch(model, dataloader, device, smooth=True, kernel_size=3):

    model.eval()

    gray, color = next(iter(dataloader))

    color = color.to(device)
    gray = gray.to(device)

    with torch.no_grad():
        predictions = model(gray)

    # Move to CPU for optional smoothing/post-processing
    preds_cpu = predictions.cpu().clone()

    if smooth:
        # Convert to [0,1] for PIL image operations
        preds_vis = torch.clamp((preds_cpu + 1) / 2, 0, 1)
        smoothed = []
        for i in range(preds_vis.size(0)):
            pil = TF.to_pil_image(preds_vis[i])
            # Median filter helps remove speckle/noise while preserving edges
            pil_sm = pil.filter(ImageFilter.MedianFilter(size=kernel_size))
            t = TF.to_tensor(pil_sm)  # returns 0-1
            t = t * 2 - 1  # convert back to [-1,1]
            smoothed.append(t)

        smoothed_preds = torch.stack(smoothed, dim=0)
    else:
        smoothed_preds = preds_cpu

    return color.cpu(), gray.cpu(), smoothed_preds