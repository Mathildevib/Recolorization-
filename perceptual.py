import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], weight=1.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList()

        prev = 0
        for lid in layer_ids:
            self.slices.append(nn.Sequential(*[vgg[i] for i in range(prev, lid)]))
            prev = lid

        for p in self.parameters():
            p.requires_grad = False

        self.normalize = T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
        self.weight = weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        pred_n   = self.normalize(pred)
        target_n = self.normalize(target)

        loss = 0.0
        x_p, x_t = pred_n, target_n

        for s in self.slices:
            x_p = s(x_p)
            x_t = s(x_t)
            loss += self.l1(x_p, x_t)

        return loss * self.weight

