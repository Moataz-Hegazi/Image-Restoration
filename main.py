import os
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset.ingesting_data import LandscapeDataset
from models.autoencoder import ColorAutoEncoder
from training.train import train
from training.evaluate import evaluate
from utils.visualization import show_predictions
from utils.prediction import predict_batch

# -----------------------------
# Configuration
# -----------------------------

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = "checkpoints/color_autoencoder.pth"

# -----------------------------
# Create checkpoint folder
# -----------------------------

os.makedirs("checkpoints", exist_ok=True)

# -----------------------------
# Transformations
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((128, 128))
])

# -----------------------------
# Dataset
# -----------------------------

dataset = LandscapeDataset("./Data", transform)

train_set, test_set = random_split(dataset,[0.8, 0.2])

trainloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Device
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Model
# -----------------------------

model = ColorAutoEncoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Load or Train model
# -----------------------------

if os.path.exists(MODEL_PATH):

    print("Loading trained model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

else:

    print("Training model...")
    train(model, trainloader, criterion, optimizer, device, EPOCHS)

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

# Set model to evaluation mode
model.eval()

# -----------------------------
# Evaluate model
# -----------------------------

test_loss = evaluate(model, testloader, criterion, device)

print("Test Loss:", test_loss)

# -----------------------------
# Show predictions
# -----------------------------

color, gray, preds = predict_batch(model, testloader, device)

show_predictions(color, gray, preds)