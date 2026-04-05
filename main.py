import os
import torch
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset.ingesting_data import LandscapeDataset
from utils.visualization import show_predictions

# =============================
# CHOOSE MODEL
# =============================
USE_GAN = False   # Set False for Autoencoder

# =============================
# Imports based on choice
# =============================
if USE_GAN:
    from models.generator import Generator
    from models.discriminator import Discriminator
else:
    from models.autoencoder import ColorAutoEncoder
    from training.train import train
    from training.evaluate import evaluate
    from utils.prediction import predict_batch

# =============================
# Configuration
# =============================
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

if USE_GAN:
    EPOCHS = 20
else:
    EPOCHS = 10

AUTOENCODER_PATH = "checkpoints/autoencoder.pth"
GEN_PATH = "checkpoints/generator.pth"
DISC_PATH = "checkpoints/discriminator.pth"

# =============================
# Setup
# =============================
os.makedirs("checkpoints", exist_ok=True)

# ❌ NO TRANSFORMS ANYMORE
# Fix path: dataset folder is lowercase `data`
dataset = LandscapeDataset("./data")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =============================
# ===== AUTOENCODER MODE ======
# =============================
if not USE_GAN:

    print("\n=== AUTOENCODER MODE ===")

    model = ColorAutoEncoder().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if os.path.exists(AUTOENCODER_PATH):
        print("Loading Autoencoder...")
        model.load_state_dict(torch.load(AUTOENCODER_PATH, map_location=device))
    else:
        print("Training Autoencoder...")
        train(model, trainloader, criterion, optimizer, device, EPOCHS)
        torch.save(model.state_dict(), AUTOENCODER_PATH)
        print("Model saved!")

    model.eval()

    test_loss = evaluate(model, testloader, criterion, device)
    print("Test Loss:", test_loss)

    color, gray, preds = predict_batch(model, testloader, device)

    # Convert for visualization
    gray = (gray + 1) / 2
    color = (color + 1) / 2
    preds = (preds + 1) / 2

    show_predictions(color, gray, preds)

# =============================
# ========= GAN MODE ==========
# =============================
else:

    print("\n=== GAN MODE ===")

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Total Variation (TV) loss to encourage spatial smoothness and reduce speckle
    def tv_loss(x):
        # x: (B, C, H, W)
        h_tv = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        v_tv = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return h_tv + v_tv

    tv_weight = 0.1

    # Color-consistency loss: compare normalized chroma (color ratios) to
    # encourage stable colors independent of brightness.
    def color_consistency_loss(fake, real, eps=1e-6):
        # Inputs are in [-1,1]; convert to [0,1]
        f = (fake + 1) / 2
        r = (real + 1) / 2

        # normalize across channels to get color ratios (sum to 1)
        f_sum = torch.sum(f, dim=1, keepdim=True)
        r_sum = torch.sum(r, dim=1, keepdim=True)

        f_norm = f / (f_sum + eps)
        r_norm = r / (r_sum + eps)

        return torch.mean(torch.abs(f_norm - r_norm))

    cc_weight = 10.0

    # -----------------------------
    # Load or Train
    # -----------------------------
    if os.path.exists(GEN_PATH) and os.path.exists(DISC_PATH):

        print("Loading GAN...")
        generator.load_state_dict(torch.load(GEN_PATH, map_location=device))
        discriminator.load_state_dict(torch.load(DISC_PATH, map_location=device))

    else:

        print("Training GAN...")

        for epoch in range(EPOCHS):

            for gray, real_color in trainloader:

                gray = gray.to(device)
                real_color = real_color.to(device)

                # -----------------
                # Train Generator
                # -----------------
                g_optimizer.zero_grad()

                fake_color = generator(gray)
                pred_fake = discriminator(gray, fake_color)

                valid = torch.ones_like(pred_fake)
                fake = torch.zeros_like(pred_fake)

                cc = color_consistency_loss(fake_color, real_color)
                g_loss = (
                    adversarial_loss(pred_fake, valid)
                    + 100 * l1_loss(fake_color, real_color)
                    + tv_weight * tv_loss(fake_color)
                    + cc_weight * cc
                )

                g_loss.backward()
                g_optimizer.step()

                # -----------------
                # Train Discriminator
                # -----------------
                d_optimizer.zero_grad()

                pred_real = discriminator(gray, real_color)
                loss_real = adversarial_loss(pred_real, valid)

                pred_fake = discriminator(gray, fake_color.detach())
                loss_fake = adversarial_loss(pred_fake, fake)

                d_loss = (loss_real + loss_fake) / 2

                d_loss.backward()
                d_optimizer.step()

            print(f"Epoch [{epoch+1}/{EPOCHS}] | G Loss: {g_loss.item():.4f} | D Loss: {d_loss.item():.4f}")

        torch.save(generator.state_dict(), GEN_PATH)
        torch.save(discriminator.state_dict(), DISC_PATH)

        print("Models saved!")

    # -----------------------------
    # 🔥 Visualization (GAN)
    # -----------------------------
    generator.eval()

    gray, real_color = next(iter(testloader))

    gray = gray.to(device)

    with torch.no_grad():
        fake_color = generator(gray)

    # Move to CPU
    gray = gray.cpu()
    real_color = real_color.cpu()
    fake_color = fake_color.cpu()

    # ✅ Convert from [-1,1] → [0,1]
    gray = (gray + 1) / 2
    real_color = (real_color + 1) / 2
    fake_color = (fake_color + 1) / 2

    # Clamp
    gray = torch.clamp(gray, 0, 1)
    real_color = torch.clamp(real_color, 0, 1)
    fake_color = torch.clamp(fake_color, 0, 1)

    show_predictions(real_color, gray, fake_color)

    print("GAN visualization complete!")