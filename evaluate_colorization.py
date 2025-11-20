import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 128
BATCH_SIZE = 32
MODEL_PATH = 'colorization_best_val.pth'
DATA_ROOT = "/ceph/home/student.aau.dk/ak68le/deeplmini/imagenet_full_256"
Q = 256

# ab bins – exactly the same as training
a_edges = torch.linspace(-110, 110, 17)
b_edges = torch.linspace(-110, 110, 17)
a_centers = (a_edges[:-1] + a_edges[1:]) / 2
b_centers = (b_edges[:-1] + b_edges[1:]) / 2
grid_a, grid_b = torch.meshgrid(a_centers, b_centers, indexing='ij')
ab_centers = torch.stack([grid_a, grid_b], dim=-1).reshape(-1, 2).to(DEVICE)

def decode_ab(class_map):
    return ab_centers[class_map]

def rgb_to_ab_class(rgb_tensor):
    rgb_np = (rgb_tensor.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
    lab = rgb2lab(rgb_np)
    a_idx = np.digitize(lab[...,1], a_edges[1:-1].cpu().numpy())
    b_idx = np.digitize(lab[...,2], b_edges[1:-1].cpu().numpy())
    return torch.from_numpy(a_idx * 16 + b_idx).long().to(DEVICE)

# Transform that returns ONLY what we need
class EvalTransform:
    def __init__(self, size=128):
        self.resize = transforms.Resize((size, size))
        self.to_tensor = transforms.ToTensor()
    def __call__(self, img):
        img = self.resize(img)
        rgb = self.to_tensor(img)                                   # (3,128,128)
        lab = rgb2lab((rgb.permute(1,2,0).numpy()*255).astype(np.uint8))
        L = torch.from_numpy(lab[:,:,0:1] / 100.0).permute(2,0,1).float()  # (1,128,128)
        return (L, rgb)                     # ← tuple, ImageFolder will add label → ((L,rgb), label)

dataset = datasets.ImageFolder(DATA_ROOT, transform=EvalTransform())

# Same split as before
train_n = int(0.7 * len(dataset))
val_n   = int(0.1 * len(dataset))
test_n  = len(dataset) - train_n - val_n
_, _, test_ds = torch.utils.data.random_split(
    dataset, [train_n, val_n, test_n],
    generator=torch.Generator().manual_seed(42)
)

# This collate function discards the useless ImageFolder class label
def collate_eval(batch):
    # batch = [ ((L, rgb), label), ... ]
    Ls   = torch.stack([item[0][0] for item in batch])
    rgbs = torch.stack([item[0][1] for item in batch])
    return Ls, rgbs

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=collate_eval, num_workers=8, pin_memory=True)

# Model – exact same as training
class ConvBlock(nn.Module):
    def __init__(self,i,o): super().__init__(); self.block=nn.Sequential(
        nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
        nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNet5(nn.Module):
    def __init__(self,num_classes=256):
        super().__init__()
        c=ConvBlock
        self.enc1=c(1,64);   self.enc2=c(64,128)
        self.enc3=c(128,256);self.enc4=c(256,512)
        self.enc5=c(512,1024)
        self.pool=nn.MaxPool2d(2,2)
        self.bottleneck=c(1024,2048)
        self.up5=nn.ConvTranspose2d(2048,1024,2,stride=2); self.dec5=c(2048,1024)
        self.up4=nn.ConvTranspose2d(1024,512,2,stride=2);  self.dec4=c(1024,512)
        self.up3=nn.ConvTranspose2d(512,256,2,stride=2);   self.dec3=c(512,256)
        self.up2=nn.ConvTranspose2d(256,128,2,stride=2);   self.dec2=c(256,128)
        self.up1=nn.ConvTranspose2d(128,64,2,stride=2);    self.dec1=c(128,64)
        self.out=nn.Conv2d(64,num_classes,1)
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

print("Loading model...")
model = UNet5(num_classes=Q).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("Model loaded – starting evaluation")

top1 = top5 = pixels = 0
psnr_sum = ssim_sum = n_imgs = 0

with torch.no_grad():
    for L, rgb_gt in tqdm(test_loader, desc="Evaluating"):
        gray = torch.mean(rgb_gt, dim=1, keepdim=True).to(DEVICE)   # (B,1,128,128)
        logits = model(gray)

        # Classification metrics
        target = rgb_to_ab_class(rgb_gt)
        pred   = logits.argmax(1)
        pixels += pred.numel()
        top1   += (pred == target).sum().item()
        top5   += logits.topk(5, dim=1)[1].eq(target.unsqueeze(1)).any(1).sum().item()

        # PSNR/SSIM on first 1000 images
        if n_imgs < 1000:
            pred_ab = decode_ab(pred).cpu().numpy()
            L_np = L.numpy() * 100
            for i in range(L.size(0)):
                if n_imgs >= 1000: break
                lab_pred = np.concatenate([L_np[i].transpose(1,2,0), pred_ab[i]], axis=-1)
                lab_gt   = rgb2lab((rgb_gt[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                psnr_sum += psnr(lab_gt, lab_pred, data_range=100)
                ssim_sum += ssim(lab_gt, lab_pred, multichannel=True, data_range=100, channel_axis=-1)
                n_imgs += 1

print("\n" + "="*60)
print("                 FINAL QUANTITATIVE RESULTS")
print("="*60)
print(f"Top-1 ab accuracy : {100*top1/pixels:5.2f}%")
print(f"Top-5 ab accuracy : {100*top5/pixels:5.2f}%")
print(f"PSNR  (LAB)       : {psnr_sum/n_imgs:5.2f} dB")
print(f"SSIM  (LAB)       : {ssim_sum/n_imgs:.4f}")
print(f"Evaluated on {n_imgs} images and {pixels:,} pixels")
print("="*60)
