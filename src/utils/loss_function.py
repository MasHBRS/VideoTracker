import torch
import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM  # You can also use 'torchmetrics.functional.structural_similarity_index_measure'
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class ReconstructionLoss_L1_Ssim(nn.Module):
    def __init__(self, device='cpu', lambda_l1=0.5, lambda_ssim=0.5):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.device=device
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # For RGB
        self.l1 = nn.L1Loss()

    def forward(self, recon, target):
        """
        recon, target: (B, T, C, H, W)
        """
        B, T, C, H, W = target.shape

        total_loss = 0.0
        for t in range(T):
            recon_t = recon[:, t, :, :, :]
            target_t = target[:, t, :, :, :]

            # SSIM loss per frame
            ssim_loss = 1 - self.ssim(recon_t, target_t)

            # L1 loss per frame
            l1_loss = self.l1(recon_t, target_t)

            total_loss += self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss

        # Average over frames
        return total_loss / T
    
class ReconstructionLoss_PSNR_SSIM(nn.Module):
    def __init__(self, lambda_psnr=1.0, lambda_ssim=0.5):
        super().__init__()
        self.lambda_psnr = lambda_psnr
        self.lambda_ssim = lambda_ssim
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # For RGB

    def forward(self, recon, target):
        """
        recon, target: (B, T, C, H, W)
        """
        B, T, C, H, W = target.shape
        total_loss = 0.0
        for t in range(T):
            recon_t = recon[:, t, :, :, :]
            target_t = target[:, t, :, :, :]
            # SSIM loss
            ssim_loss = 1 - self.ssim(recon_t, target_t)
            # PSNR loss (negative because higher PSNR = better)
            psnr_loss = -self.psnr(recon_t, target_t, max_val=1.0)
            total_loss += self.lambda_ssim * ssim_loss + self.lambda_psnr * psnr_loss
        return total_loss / T

    def psnr(self, img1, img2, max_val=1.0):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return torch.tensor(float('inf'), device=img1.device)
        psnr_value = 20 * torch.log10(torch.tensor(max_val, device=img1.device)) - 10 * torch.log10(mse)
        return psnr_value