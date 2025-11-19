# test_colorization.py   ← FINAL VERSION – WILL WORK 100%
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.color import lab2rgb, rgb2lab
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 128
BATCH_SIZE = 64
MODEL_PATH = 'colorization_l4_final_4.pth'
DATA_ROOT = "/ceph/home/student.aau.dk/ak68le/deeplmini/imagenet_full_256"
Q = 256

# ab centers (exact same as training)
a_edges = torch.linspace(-110, 110, 17)
b_edges = torch.linspace(-110, 110, 17)
a_centers = (a_edges[:-1] + a_edges[1:]) / 2
b_centers = (b_edges[:-1] + b_edges[1:]) / 2
grid_a, grid_b = torch.meshgrid(a_centers, b_centers, indexing='ij')
ab_centers = torch.stack([grid_a, grid_b], dim=-1).reshape(-1, 2).to(DEVICE)

def decode_ab(class_map):
    return ab_centers[class_map]

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1,bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o,o,3,padding=1,bias=False),
            nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

# MODEL – exact naming as training
class UNet5(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()
        c = ConvBlock
        self.enc1 = c(1,64);   self.enc2 = c(64,128)
        self.enc3 = c(128,256);self.enc4 = c(256,512)
        self.enc5 = c(512,1024)
        self.pool = nn.MaxPool2d(2,2)
        self.bottleneck = c(1024,2048)               # ← MUST be "bottleneck"
        self.up5 = nn.ConvTranspose2d(2048,1024,2,stride=2); self.dec5 = c(2048,1024)
        self.up4 = nn.ConvTranspose2d(1024,512,2,stride=2);  self.dec4 = c(1024,512)
        self.up3 = nn.ConvTranspose2d(512,256,2,stride=2);   self.dec3 = c(512,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,stride=2);   self.dec2 = c(256,128)
        self.up1 = nn.ConvTranspose2d(128,64,2,stride=2);    self.dec1 = c(128,64)
        self.out = nn.Conv2d(64,num_classes,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1))
        e3=self.enc3(self.pool(e2)); e4=self.enc4(self.pool(e3))
        e5=self.enc5(self.pool(e4))
        b=self.bottleneck(self.pool(e5))
        d5=self.dec5(torch.cat([self.up5(b),e5],1))
        d4=self.dec4(torch.cat([self.up4(d5),e4],1))
        d3=self.dec3(torch.cat([self.up3(d4),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.out(d1)

# Transform – now 100% correct L channel
class TestTransform:
    def __init__(self, size=128):
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()
        self.gray = transforms.Grayscale(1)
    def __call__(self, img):
        img = self.resize(img)
        gray = self.to_tensor(self.gray(img))
        rgb  = self.to_tensor(img)
        rgb_np = (rgb.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        lab = rgb2lab(rgb_np)
        L = torch.from_numpy(lab[:,:,0:1] / 100.0).permute(2,0,1).float()  # (1,128,128)
        return (gray, L, rgb)

# Dataset + collate that discards ImageFolder label
dataset = datasets.ImageFolder(DATA_ROOT, transform=TestTransform())
train_n = int(0.7 * len(dataset))
val_n   = int(0.1 * len(dataset))
test_n  = len(dataset) - train_n - val_n
_, _, test_ds = torch.utils.data.random_split(
    dataset, [train_n, val_n, test_n],
    generator=torch.Generator().manual_seed(42)
)

def collate_fn(batch):
    grays = torch.stack([x[0][0] for x in batch])
    Ls    = torch.stack([x[0][1] for x in batch])
    rgbs  = torch.stack([x[0][2] for x in batch])
    return grays, Ls, rgbs

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_fn, num_workers=8, pin_memory=True)

# Load model
model = UNet5(num_classes=Q).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("Model loaded – starting inference!")

softmax = nn.Softmax(dim=1)
os.makedirs("test_results", exist_ok=True)

# FIXED: torch.no_grad() not torch.no.no_grad()
with torch.no_grad():
    for batch_idx, (gray, L, rgb_gt) in enumerate(tqdm(test_loader, desc="Testing")):
        gray = gray.to(DEVICE)
        logits = model(gray)
        pred_class = torch.argmax(softmax(logits), dim=1)
        ab_pred = decode_ab(pred_class)

        for i in range(gray.size(0)):
            idx = batch_idx * BATCH_SIZE + i
            if idx >= 60:           # save first 60 beautiful results
                break

            L_np = L[i].cpu().numpy() * 100
            lab_full = np.concatenate([L_np.transpose(1,2,0), ab_pred[i].cpu().numpy()], axis=-1)
            colorized = lab2rgb(lab_full)

            plt.figure(figsize=(15,5))
            plt.subplot(131); plt.imshow(gray[i].cpu().squeeze(), cmap='gray'); plt.title('Grayscale Input'); plt.axis('off')
            plt.subplot(132); plt.imshow(colorized); plt.title('Colorized (Our Model)'); plt.axis('off')
            plt.subplot(133); plt.imshow(rgb_gt[i].permute(1,2,0).cpu().numpy()); plt.title('Ground Truth'); plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"test_results/sample_{idx:03d}.png", dpi=200, bbox_inches='tight')
            plt.close()

print("\nALL DONE! Go check ./test_results/ – your colorized images are ready!")
