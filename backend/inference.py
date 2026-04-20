import torch
from models.generator import Generator
import torchvision.transforms.functional as TF
from PIL import Image

device = torch.device("cpu")

model = Generator().to(device)
model.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
model.eval()

def colorize_image(image_path):
    img = Image.open(image_path).convert("L")  # grayscale

    img = TF.resize(img, (128, 128))
    img = TF.to_tensor(img)

    # Normalize [-1,1]
    img = (img - 0.5) / 0.5
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    # Back to [0,1]
    output = (output + 1) / 2
    output = output.squeeze(0)

    return TF.to_pil_image(output)