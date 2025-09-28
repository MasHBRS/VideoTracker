import torch
import torch.nn as nn
from pytorch_msssim import SSIM, MS_SSIM  # You can also use 'torchmetrics.functional.structural_similarity_index_measure'
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class ReconstructionLoss_L1_Ssim_Lpips(nn.Module):
    def __init__(self, device='cpu', lambda_l1=1.0, lambda_ssim=0.5, lambda_lpips=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.device=device
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)  # For RGB
        self.l1 = nn.L1Loss()
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)  # LPIPS uses [-1,1] range

    def forward(self, recon, target):
        # Ensure images are in [0,1] for SSIM
        _, _, C, H, W = target.shape
        target_flat = target.view(-1, C, H, W)
        recon_flat = recon.view(-1, C, H, W)

        # Compute SSIM loss
        ssim_loss = 1 - self.ssim(recon_flat, target_flat)
        recon_flat=target_flat=None
        # L1 loss
        l1_loss = self.l1(recon, target)

        # LPIPS expects [-1,1]
        lpips_loss = self.compute_loss_metric(recon, target,self.lpips_loss)

        # Weighted sum
        loss = self.lambda_l1 * l1_loss + self.lambda_ssim * ssim_loss + self.lambda_lpips * lpips_loss
        return loss
    
    def compute_loss_metric(self,tensor1, tensor2,metric):
        batch_size, T, C, H, W = tensor1.shape
        losses = []
        for i in range(0, batch_size, 1):  # One batch at a time
            for t in range(T): # One time step at a time
                loss = metric(tensor1[i:i+1, t, :, :, :], tensor2[i:i+1, t, :, :, :])
                losses.append(loss.item())
        torch.cuda.empty_cache()  # Clear memory
        return sum(losses)/len(losses)
