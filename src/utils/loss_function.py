import torch
import torch.nn as nn
#import lpips
from pytorch_msssim import SSIM, MS_SSIM  # You can also use 'torchmetrics.functional.structural_similarity_index_measure'

class ReconstructionLoss_L1_Ssim(nn.Module):
    def __init__(self, device='cpu', lambda_l1=1.0, lambda_ssim=0.5, lambda_lpips=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips

        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # For RGB
        self.l1 = nn.L1Loss()
        #self.lpips_loss = lpips.LPIPS(net='vgg').to(device)  # LPIPS uses [-1,1] range

    def forward(self, recon, target):
        # Ensure images are in [0,1] for SSIM
        _, _, C, H, W = target.shape
        target_flat = target.view(-1, C, H, W)
        recon_flat = recon.view(-1, C, H, W)

        # Compute SSIM loss
        ssim_loss = 1 - self.ssim(recon_flat, target_flat)

        # L1 loss
        l1_loss = self.l1(recon, target)

        # LPIPS expects [-1,1]
        #lpips_loss = self.lpips_loss(recon_flat, target_flat).mean()

        # Weighted sum
        loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss #+ self.lambda_lpips * lpips_loss
        return loss