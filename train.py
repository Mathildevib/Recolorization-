import os
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from recolor_data import RecolorDataset
from models import UNetGenerator, Discriminator
from perceptual import PerceptualLoss

device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
bw_path    = "recolor_data/train_black/"
color_path = "recolor_data/train_color/"

dataset = RecolorDataset(bw_path, color_path)
loader  = DataLoader(dataset, batch_size=4, shuffle=True)

# Models
G = UNetGenerator(3, 3).to(device)
D = Discriminator().to(device)

# Loss functions
criterion_gan  = nn.MSELoss()
criterion_l1   = nn.L1Loss()
criterion_perc = PerceptualLoss( weight=0.1 ).to(device)

# Optimizers
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

lambda_l1   = 100
lambda_perc = 10
epochs      = 50

os.makedirs("samples", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    for bw, color in loader:
        bw    = bw.to(device)
        color = color.to(device)

        # Train D
       
        G.eval()
        D.train()

        with torch.no_grad():
            fake = G(bw)

        pred_real = D(bw, color)
        loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))

        pred_fake = D(bw, fake)
        loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))

        loss_D = 0.5 * (loss_real + loss_fake)

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train G
      
        G.train()
        D.eval()

        fake = G(bw)
        pred = D(bw, fake)

        loss_G_gan  = criterion_gan(pred, torch.ones_like(pred))
        loss_G_l1   = criterion_l1(fake, color)
        loss_G_perc = criterion_perc(fake, color)

        loss_G = loss_G_gan + lambda_l1 * loss_G_l1 + lambda_perc * loss_G_perc

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"[{epoch+1}/{epochs}] D={loss_D.item():.4f} G={loss_G.item():.4f}")

    # Save output image
    save_image((fake[0] * 0.5 + 0.5), f"samples/epoch_{epoch+1}.png")

    # Save checkpoints
    torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch+1}.pth")
    torch.save(D.state_dict(), f"checkpoints/D_epoch_{epoch+1}.pth")

