#Recolorization dataset handling. 

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os
class RecolorDataset(Dataset):
    def __init__(self, grey_dir, color_dir):
        self.grey_dir = grey_dir
        self.color_dir = color_dir

        # Find joint filenames
        self.filenames = sorted(os.listdir(grey_dir))

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        grey_path = os.path.join(self.grey_dir, name)
        color_path = os.path.join(self.color_dir, name)

        grey = Image.open(grey_path).convert("L")
        color = Image.open(color_path).convert("RGB")

        grey = self.to_tensor(grey)
        color = self.to_tensor(color)

        return grey, color
