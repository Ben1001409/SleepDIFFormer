import torch
import torch.nn as nn
import torch.nn.functional as F


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, x, recon):

        recon_loss = F.mse_loss(recon, x, reduction="mean")
        return recon_loss