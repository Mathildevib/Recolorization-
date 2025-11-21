"""Test script for a U-NET5 based image colorization model.
Converts test images in the same manner as in training.
Loads the trained model, evaluates PSNR and SSIM on the full test set,
and saves 100 colorized images alongside their grayscale inputs and ground truth."""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
MODEL_PATH = "colorization_best_val_2.pth"
DATA_ROOT = Path("imagenet_full_256")

# Quantization parameters
NUM_BINS = 256
a_edges = torch.linspace(-110, 110, 17)
b_edges = torch.linspace(-110, 110, 17)
a_centers = (a_edges[:-1] + a_edges[1:]) / 2
b_centers = (b_edges[:-1] + b_edges[1:]) / 2

grid_a, grid_b = torch.meshgrid(a_centers, b_centers, indexing='ij')
ab_lookup = torch.stack([grid_a.flatten(), grid_b.flatten()], dim=1).to(DEVICE)

def encode_ab(ab_np):
    a_idx = np.digitize(ab_np[..., 0], a_edges.numpy()[1:-1])
    b_idx = np.digitize(ab_np[..., 1], b_edges.numpy()[1:-1])
    return (a_idx * 16 + b_idx).astype(np.int32)

# Class that resizes and tensorizes input images, converts RGB to Lab, splits L and ab. 
# Same as in training. # *** This section was both debugged and made in collaboration with XAI GROK 3/CHAT GPT***
class GrayColorTransform:
    def __init__(self, size=128):
        self.size = size
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BICUBIC)
        tensor = self.to_tensor(img)
        rgb_np = (tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
        lab = rgb2lab(rgb_np)

        L_np  = lab[..., :1] / 100.0
        ab_np = lab[..., 1:]

        class_map = encode_ab(ab_np)
        class_t   = torch.from_numpy(class_map).long()

        counts = np.bincount(class_map.flatten(), minlength=NUM_BINS)
        counts = np.maximum(counts, 1)
        w = 1.0 / (counts ** 0.5)
        w = w / w.sum() * NUM_BINS
        weight_t = torch.from_numpy(w[class_map]).float()

        L_input = torch.from_numpy(L_np).permute(2,0,1).float()

        return (L_input, L_input, class_t, weight_t)

transform = GrayColorTransform()

# Getting test dataset
full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
train_size = int(0.7 * len(full_dataset))
val_size   = int(0.1 * len(full_dataset))
test_size  = len(full_dataset) - train_size - val_size

_, _, test_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
# *** This section was both debugged and made in collaboration with XAI GROK 3/CHAT GPT***
def collate_test(batch):
    batch = [item[0] for item in batch]
    L_input = torch.stack([item[0] for item in batch])      
    L_true  = torch.stack([item[1] for item in batch])      
    target  = torch.stack([item[2] for item in batch]).long()  
    weight  = torch.stack([item[3] for item in batch])
    return L_input, L_true, target, weight

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_test, num_workers=4, pin_memory=True)

# 2 x convolution block with BatchNorm and ReLU used in U-Net
# Same as in training.
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

# 5-level U-Net architecture
# Same as in training. 
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

# Get model
model = UNet5().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Loaded {MODEL_PATH}")

# Test metric accumulators
psnr_total = 0.0
ssim_total = 0.0
metric_count = 0

# Creating save directory
os.makedirs("test_results", exist_ok=True)

# *** This section was both debugged and made in collaboration with XAI GROK 3/CHAT GPT***
with torch.no_grad():
    for batch_idx, (L_input, L_true, target_class, _) in enumerate(tqdm(test_loader, desc="Colorizing")):
        L_input = L_input.to(DEVICE)                                 

        logits = model(L_input)
        pred_class = logits.softmax(1).argmax(1)        

        pred_ab = ab_lookup[pred_class]                  
        true_ab = ab_lookup[target_class.to(DEVICE).long()]

        for i in range(L_input.size(0)):
            idx = batch_idx * BATCH_SIZE + i
            if idx >= 100:                               
                continue

            L_np = L_true[i].cpu().numpy().squeeze(0) * 100              
            pred_ab_np = pred_ab[i].cpu().numpy()        
            true_ab_np = true_ab[i].cpu().numpy()

            pred_lab = np.stack([L_np, pred_ab_np[..., 0], pred_ab_np[..., 1]], axis=-1)
            true_lab = np.stack([L_np, true_ab_np[..., 0], true_ab_np[..., 1]], axis=-1)

            pred_rgb = lab2rgb(pred_lab)
            true_rgb = lab2rgb(true_lab)

            # PSNR/SSIM between prediction and ground truth (RGB)
            try:
                psnr_total += psnr(true_rgb, pred_rgb, data_range=1.0)
                ssim_total += ssim(true_rgb, pred_rgb, data_range=1.0, channel_axis=-1)
                metric_count += 1
            except:
                pass 

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title("Input (True L*)")
            plt.imshow(L_np, cmap='gray', vmin=0, vmax=100)
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.title("Colorization")
            plt.imshow(pred_rgb)
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.title("Ground Truth")
            plt.imshow(true_rgb)
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(f"test_results/{idx:04d}.jpg", dpi=200, bbox_inches='tight')
            plt.close()

# Print average PSNR/SSIM
if metric_count > 0:
    print(f"\nPSNR average over {metric_count} images: {psnr_total / metric_count:.3f} dB")
    print(f"SSIM average over {metric_count} images: {ssim_total / metric_count:.4f}")

print("\nSaved 100 images in ./test_results/")

