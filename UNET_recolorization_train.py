""" Training script for a U-NET5 based image colorization model. 
The script converts RGB images to Lab color space and quantizes the ab channels into 256 bins.
Includes weights for class imbalance and early stopping based on validation loss"""
#Importation of libraries needed for training, preprocessing and visualization. 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from torchvision import datasets
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path


# Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 96
NUM_EPOCHS = 25
LR = 3e-4
PATIENCE = 6
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

# Path to data and model checkpoint path (best validation model is saved)
DATA_ROOT = Path("imagenet_full_256")
MODEL_PATH = "colorization_best_val.pth"

# Quantization parameters for ab color space (bin number 256, grid 16x16)
NUM_BINS = 256
GAMMA = 0.5
a_edges = torch.linspace(-110, 110, 17)
b_edges = torch.linspace(-110, 110, 17)
a_centers = (a_edges[:-1] + a_edges[1:]) / 2
b_centers = (b_edges[:-1] + b_edges[1:]) / 2

# Each pixel is mapped to nearest ab bin
def encode_ab(ab_np):
    a_idx = np.digitize(ab_np[..., 0], a_edges.numpy()[1:-1])
    b_idx = np.digitize(ab_np[..., 1], b_edges.numpy()[1:-1])
    return (a_idx * 16 + b_idx).astype(np.int32)

# Class that resizes and tensorizes input images, converts RGB to Lab, splits L and ab.
# *** This Class was both debugged and made in collaboration with XAI GROK 3/CHAT GPT*** 
class GrayColorTransform:
    def __init__(self, size=128):
        self.size = size
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BICUBIC)
        tensor = self.to_tensor(img)
        rgb_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        lab = rgb2lab(rgb_np)

        L_np  = lab[..., :1] / 100.0
        ab_np = lab[..., 1:]

        class_map = encode_ab(ab_np)
        class_t   = torch.from_numpy(class_map).long()

        counts = np.bincount(class_map.flatten(), minlength=NUM_BINS)
        counts = np.maximum(counts, 1)
        w = 1.0 / (counts ** GAMMA)
        w = w / w.sum() * NUM_BINS
        weight_t = torch.from_numpy(w[class_map]).float()

        L_input = torch.from_numpy(L_np).permute(2, 0, 1).float()

        return (L_input, L_input, class_t, weight_t) 

transform = GrayColorTransform()

# Dataset split, train 70%, val 10%, test 20%, seed for reproducibility
full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size   = int(0.1 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

train_ds, val_ds, _ = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
# Stack grayscale, L-channel, target and weight tensors
# *** This section was both debugged and made in collaboration with XAI GROK 3/CHAT GPT***
def collate_fn(batch):
    batch = [item[0] for item in batch]  
    L_input = torch.stack([x[0] for x in batch])
    L_true  = torch.stack([x[1] for x in batch])
    target  = torch.stack([x[2] for x in batch]).long()
    weight  = torch.stack([x[3] for x in batch])
    return L_input, L_true, target, weight

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=10, pin_memory=True,
                          prefetch_factor=2, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=8,  pin_memory=True,
                          persistent_workers=True)

# 2 x convolution block with BatchNorm and ReLU used in U-Net
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

# 5-level U-Net architecture
class UNet5(nn.Module):
    def __init__(self, n_classes=256):
        super().__init__()
        ch = [64, 128, 256, 512, 1024]

        self.enc = nn.ModuleList([ConvBlock(1 if i==0 else ch[i-1], ch[i]) for i in range(5)])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(1024, 2048)

        self.up5  = nn.ConvTranspose2d(2048, 1024, 2, stride=2); self.dec5 = ConvBlock(2048, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512,  2, stride=2); self.dec4 = ConvBlock(1024, 512)
        self.up3  = nn.ConvTranspose2d(512,  256,  2, stride=2); self.dec3 = ConvBlock(512,  256)
        self.up2  = nn.ConvTranspose2d(256,  128,  2, stride=2); self.dec2 = ConvBlock(256,  128)
        self.up1  = nn.ConvTranspose2d(128,   64,  2, stride=2); self.dec1 = ConvBlock(128,   64)
        self.out  = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        x = self.up5(x); x = torch.cat([x, skips[4]], dim=1); x = self.dec5(x)
        x = self.up4(x); x = torch.cat([x, skips[3]], dim=1); x = self.dec4(x)
        x = self.up3(x); x = torch.cat([x, skips[2]], dim=1); x = self.dec3(x)
        x = self.up2(x); x = torch.cat([x, skips[1]], dim=1); x = self.dec2(x)
        x = self.up1(x); x = torch.cat([x, skips[0]], dim=1); x = self.dec1(x)

        return self.out(x)

# Model, loss, optimizer
model = UNet5().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scaler = torch.amp.GradScaler('cuda')

# Training loop
# *** This section was both debugged and made in collaboration with XAI GROK 3/CHAT GPT***
best_val = float('inf')
patience_cnt = 0
train_losses, val_losses = [], []

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    running_loss = 0.0
    for L_input, L_true, target, weight in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        L_input = L_input.to(DEVICE, non_blocking=True)
        target  = target.to(DEVICE, non_blocking=True)
        weight  = weight.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(L_input)
            loss = (criterion(logits, target) * weight).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for L_input, L_true, target, weight in val_loader:
            L_input, target, weight = L_input.to(DEVICE), target.to(DEVICE), weight.to(DEVICE)
            with torch.amp.autocast('cuda'):
                logits = model(L_input)
                loss = (criterion(logits, target) * weight).mean()
            val_loss += loss.item()
    val_loss /= len(val_loader)
    train_loss = running_loss / len(train_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch:02d} | Train {train_loss:.5f} | Val {val_loss:.5f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"   BEST MODEL SAVED ({best_val:.5f})")
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping")
            break

# Plot
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Val")
plt.legend(); plt.grid(alpha=0.3)
plt.title("Colorization Training")
plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nBest val loss = {best_val:.5f}")
print(f"Model saved → {MODEL_PATH}")