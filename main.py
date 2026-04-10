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
USE_GAN = True   # Set False for Autoencoder

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
EPOCHS = 25

AUTOENCODER_PATH = "checkpoints/autoencoder.pth"
GEN_PATH = "checkpoints/generator.pth"
DISC_PATH = "checkpoints/discriminator.pth"

# =============================
# Setup
# =============================
os.makedirs("checkpoints", exist_ok=True)

dataset = LandscapeDataset("./Data")

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

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

            for step, (gray, real_color) in enumerate(trainloader):

                gray = gray.to(device)
                real_color = real_color.to(device)

                # -----------------
                # Train Generator
                # -----------------
                g_optimizer.zero_grad()

                fake_color = generator(gray)

                # Optional blur stabilization
                fake_for_disc = TF.gaussian_blur(fake_color, kernel_size=3)

                pred_fake = discriminator(gray, fake_for_disc)

                valid = torch.ones_like(pred_fake)

                g_loss = 0.5 * adversarial_loss(pred_fake, valid) + 200 * l1_loss(fake_color, real_color)

                g_loss.backward()
                g_optimizer.step()

                # -----------------
                # Train Discriminator (slower)
                # -----------------
                if step % 2 == 0:
                    d_optimizer.zero_grad()

                    fake_detached = fake_color.detach()
                    fake_detached = TF.gaussian_blur(fake_detached, kernel_size=3)

                    pred_real = discriminator(gray, real_color)
                    pred_fake = discriminator(gray, fake_detached)

                    valid = torch.ones_like(pred_real)
                    fake = torch.zeros_like(pred_fake)

                    loss_real = adversarial_loss(pred_real, valid)
                    loss_fake = adversarial_loss(pred_fake, fake)

                    d_loss = (loss_real + loss_fake) / 2

                    d_loss.backward()
                    d_optimizer.step()

            print(f"Epoch [{epoch+1}/{EPOCHS}] | G Loss: {g_loss.item():.4f}")

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

    # Convert from [-1,1] → [0,1]
    gray = (gray + 1) / 2
    real_color = (real_color + 1) / 2
    fake_color = (fake_color + 1) / 2

    # Clamp
    gray = torch.clamp(gray, 0, 1)
    real_color = torch.clamp(real_color, 0, 1)
    fake_color = torch.clamp(fake_color, 0, 1)

    # Crop borders (final polish)
    gray = gray[:, :, 2:-2, 2:-2]
    real_color = real_color[:, :, 2:-2, 2:-2]
    fake_color = fake_color[:, :, 2:-2, 2:-2]

    show_predictions(real_color, gray, fake_color)

    print("GAN visualization complete!")