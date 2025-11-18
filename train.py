# train_pix2pix.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from recolor_data import RecolorDataset
from models import UNet, Discriminator
from perceptual import PerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bw_path    = "recolor_data/train_black/"
color_path = "recolor_data/train_color/"

dataset = RecolorDataset(bw_path, color_path)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

#Generator U-Net and Discriminator
G = UNet().to(device)           
D = Discriminator().to(device)  

# Loss functions
criterion_gan = nn.BCEWithLogitsLoss()
criterion_l1 = nn.L1Loss()
criterion_perc = PerceptualLoss(weight=0.1).to(device)  

# Optimizers
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

lambda_l1 = 100.0   
lambda_perc = 10.0  
num_epochs = 20

for epoch in range(num_epochs):
    for bw, color in loader:
        bw = bw.to(device)
        color = color.to(device)

        # Training of the discriminator
        G.eval()
        D.train()

        with torch.no_grad():
            fake_color = G(bw)

        # Real color images
        pred_real = D(bw, color)
        real_labels = torch.ones_like(pred_real, device=device)
        loss_D_real = criterion_gan(pred_real, real_labels)

        # Fake color images
        pred_fake = D(bw, fake_color)
        fake_labels = torch.zeros_like(pred_fake, device=device)
        loss_D_fake = criterion_gan(pred_fake, fake_labels)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Training of the generator
        G.train()
        D.eval()

        fake_color = G(bw)
        pred_fake_for_G = D(bw, fake_color)

        # GAN-loss: 
        target_real_for_G = torch.ones_like(pred_fake_for_G, device=device)
        loss_G_gan = criterion_gan(pred_fake_for_G, target_real_for_G)

        # L1 loss mellem fake og ground truth
        loss_G_l1 = criterion_l1(fake_color, color)

        # Perceptual loss
        loss_G_perc = criterion_perc(fake_color, color)

        # Joint generator-loss
        loss_G = loss_G_gan + lambda_l1 * loss_G_l1 + lambda_perc * loss_G_perc

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"D: {loss_D.item():.4f} | "
          f"G_gan: {loss_G_gan.item():.4f} | "
          f"G_l1: {loss_G_l1.item():.4f} | "
          f"G_perc: {loss_G_perc.item():.4f}")
