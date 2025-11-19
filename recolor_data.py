import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RecolorDataset(Dataset):
    def __init__(self, grey_dir, color_dir):
        self.grey_paths  = sorted(os.listdir(grey_dir))
        self.grey_dir    = grey_dir
        self.color_dir   = color_dir

        self.transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.grey_paths)

    def __getitem__(self, idx):
        name = self.grey_paths[idx]

        grey  = Image.open(os.path.join(self.grey_dir, name)).convert("RGB")
        color = Image.open(os.path.join(self.color_dir, name)).convert("RGB")

        return self.transform(grey), self.transform(color)

