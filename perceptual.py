
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], weight=1.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Vi klipper de første N lag, men vi skal bruge output fra flere checkpoints
        self.slices = nn.ModuleList()
        prev_idx = 0
        for lid in layer_ids:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev_idx, lid)]))
            prev_idx = lid
        for p in self.parameters():
            p.requires_grad = False

        # Normalization til VGG16
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        pred_n = self.normalize(pred)
        target_n = self.normalize(target)

        loss = 0.0
        x_p = pred_n
        x_t = target_n
        for slice in self.slices:
            x_p = slice(x_p)
            x_t = slice(x_t)
            loss = loss + self.l1(x_p, x_t)

        return self.weight * loss
