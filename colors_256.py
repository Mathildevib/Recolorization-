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

# -------------------------- L4 SETTINGS --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_EPOCHS = 4
LEARNING_RATE = 1e-4
IMG_SIZE = 128
SEED = 42
torch.manual_seed(SEED)

DATA_ROOT = "/ceph/home/student.aau.dk/ak68le/deeplmini/imagenet_full_256"
MODEL_SAVE_PATH = 'colorization_l4_final_4.pth'
LOSS_PLOT_PATH = 'loss_plot_final.png'

print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory // 1e9} GB")

# -------------------------- 256 BINS --------------------------
Q = 256
GAMMA = 0.5
a_edges = torch.linspace(-110, 110, 17)
b_edges = torch.linspace(-110, 110, 17)
a_centers = (a_edges[:-1] + a_edges[1:]) / 2
b_centers = (b_edges[:-1] + b_edges[1:]) / 2

def encode_ab(ab_np):
    a = ab_np[:, :, 0]
    b = ab_np[:, :, 1]
    a_idx = np.digitize(a, a_edges.numpy()[1:-1])
    b_idx = np.digitize(b, b_edges.numpy()[1:-1])
    return (a_idx * 16 + b_idx).astype(np.int32)

# -------------------------- TRANSFORM --------------------------
class GrayColorTransform:
    def __init__(self, size):
        self.resize = T.Resize((size, size))
        self.to_tensor = T.ToTensor()
        self.to_gray = T.Grayscale(num_output_channels=1)

    def __call__(self, img):
        img = self.resize(img)
        color_tensor = self.to_tensor(img)
        rgb_np = (color_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lab = rgb2lab(rgb_np)
        L_np = lab[:, :, 0:1] / 100.0
        ab_np = lab[:, :, 1:3]

        class_map = encode_ab(ab_np)
        class_map_t = torch.from_numpy(class_map).long()

        counts = np.bincount(class_map.flatten(), minlength=Q)
        counts = np.maximum(counts, 1)
        weights_np = 1.0 / (counts ** GAMMA)
        weights_np = weights_np / weights_np.sum() * Q
        weight_map = weights_np[class_map]

        gray = self.to_tensor(self.to_gray(img))
        L = torch.from_numpy(L_np).permute(2, 0, 1)
        weights_t = torch.from_numpy(weight_map)

        return (gray, L, class_map_t, weights_t)  # ← SINGLE TUPLE

transform = GrayColorTransform(IMG_SIZE)

# -------------------------- DATASET --------------------------
full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
print(f"Found {len(full_dataset)} images across {len(full_dataset.classes)} classes")

train_size = int(0.7 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

# -------------------------- COLLATE --------------------------
def collate_fn(batch):
    # batch = [((gray, L, class, weights), label), ...]
    grays = torch.stack([item[0][0] for item in batch])
    Ls = torch.stack([item[0][1] for item in batch])
    classes = torch.stack([item[0][2] for item in batch])
    weights = torch.stack([item[0][3] for item in batch])
    return grays, Ls, classes, weights

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, num_workers=8, pin_memory=True, prefetch_factor=2
)

# -------------------------- MODEL --------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet5(nn.Module):
    def __init__(self, num_classes=Q):
        super().__init__()
        self.enc1 = ConvBlock(1, 64);   self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256);self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(1024, 2048)
        self.up5 = nn.ConvTranspose2d(2048, 1024, 2, stride=2); self.dec5 = ConvBlock(2048, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512,  2, stride=2); self.dec4 = ConvBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512,  256,  2, stride=2); self.dec3 = ConvBlock(512,  256)
        self.up2 = nn.ConvTranspose2d(256,  128,  2, stride=2); self.dec2 = ConvBlock(256,  128)
        self.up1 = nn.ConvTranspose2d(128,   64,  2, stride=2); self.dec1 = ConvBlock(128,   64)
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4)); b = self.bottleneck(self.pool(e5))
        d5 = self.dec5(torch.cat([self.up5(b), e5], 1))
        d4 = self.dec4(torch.cat([self.up4(d5), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

# -------------------------- TRAIN --------------------------
model = UNet5().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting L4 training...")
losses = []
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for gray, L, target_class, weights in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        gray = gray.to(DEVICE)
        target_class = target_class.to(DEVICE)
        weights = weights.to(DEVICE)

        logits = model(gray)
        loss = criterion(logits, target_class)
        loss = (loss * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  → Best model saved! Loss: {best_loss:.6f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Training complete! Final model: {MODEL_SAVE_PATH}")

plt.figure(figsize=(8, 5))
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.title('Training Loss Over Epochs (ImageNet-256 → 128×128)')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("report_loss_curve.png", dpi=300, bbox_inches='tight')
print("REPORT: Loss curve saved → report_loss_curve.png")
