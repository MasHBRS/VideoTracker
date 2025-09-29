import torch
import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM  # You can also use 'torchmetrics.functional.structural_similarity_index_measure'
#from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import models


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
    
class ReconstructionLoss_MSE_SSIM(nn.Module):
    def __init__(self, device='cpu', lambda_mse=0.5, lambda_ssim=0.5):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.device=device
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # For RGB
        self.mse = nn.MSELoss()

    def forward(self, recon, target):
        """
        recon, target: (B, T, C, H, W)
        """
        B, T, C, H, W = target.shape

        total_loss = 0.0
        for t in range(T):
            recon_t = recon[:, t, :, :, :]
            target_t = target[:, t, :, :, :]

            ssim_loss = 1 - self.ssim(recon_t, target_t)
            mse_loss = self.mse(recon_t, target_t)
            
            total_loss += self.lambda_mse * mse_loss + self.lambda_ssim * ssim_loss

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

class ImprovedLoss5D(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg = models.vgg16(pretrained=True).features[:16]
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def perceptual_loss(self, pred, target):
        # pred/target shape: [B, F, C, H, W]
        batch_size, num_frames = pred.shape[0], pred.shape[1]
        
        # Reshape to process all frames at once: [B*F, C, H, W]
        pred_flat = pred.reshape(-1, *pred.shape[2:])
        target_flat = target.reshape(-1, *target.shape[2:])
        self.vgg=self.vgg.to(pred.device)
        # Get features for all frames
        pred_feats = self.vgg(pred_flat)
        target_feats = self.vgg(target_flat)
        
        # Return to batch dimension and average over frames
        pred_feats = pred_feats.reshape(batch_size, num_frames, -1)
        target_feats = target_feats.reshape(batch_size, num_frames, -1)
        
        return torch.mean((pred_feats - target_feats) ** 2)
    
    def forward(self, pred, target):
        l1 = self.l1(pred, target)  # L1 automatically handles 5D
        
        # Optional: You might want to average over frames for L1 too
        # l1 = self.l1(pred.reshape(-1, *pred.shape[2:]), 
        #              target.reshape(-1, *target.shape[2:]))
        
        percep = self.perceptual_loss(pred, target)
        return l1 + 0.05 * percep