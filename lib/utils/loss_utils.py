#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2, mask=None):
    '''
    img1, img2: (C, H, W)
    mask: (1, H, W)
    '''

    img1 = img1.permute(1, 2, 0)
    img2 = img2.permute(1, 2, 0)

    if mask is not None:
        mask = mask.squeeze(0)
        img1 = img1[mask]
        img2 = img2[mask]

    # mse = ((img1 - img2) ** 2).view(-1, img1.shape[-1]).mean(dim=0, keepdim=True)
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


def _to_grayscale(img):
    if img.shape[0] == 3:
        return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    return img[0]


def _fftshift2d(x):
    shift_y = x.shape[-2] // 2
    shift_x = x.shape[-1] // 2
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(-2, -1))


def _phase_only_correlation(fft1, fft2, eps):
    phase1 = fft1 / fft1.abs().clamp_min(eps)
    phase2 = fft2 / fft2.abs().clamp_min(eps)
    return torch.real(phase1 * torch.conj(phase2)).mean()


def _build_log_polar_grid(height, width, radial_bins, angular_bins, device, dtype):
    center_x = 0.5 * (width - 1)
    center_y = 0.5 * (height - 1)
    max_radius = max(min(center_x, center_y), 1.0)

    radial = torch.linspace(0.0, 1.0, radial_bins, device=device, dtype=dtype)
    angular = torch.linspace(-torch.pi, torch.pi, angular_bins, device=device, dtype=dtype)
    rr, tt = torch.meshgrid(radial, angular, indexing="ij")
    radius = torch.exp(rr * torch.log(torch.tensor(max_radius + 1.0, device=device, dtype=dtype))) - 1.0

    x = center_x + radius * torch.cos(tt)
    y = center_y + radius * torch.sin(tt)
    grid_x = 2.0 * x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * y / max(height - 1, 1) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def _log_polar_sample(image):
    height, width = image.shape[-2], image.shape[-1]
    grid = _build_log_polar_grid(
        height, width, height, width, image.device, image.dtype
    )
    sampled = F.grid_sample(
        image.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, 0]


def phase_loss(img1, img2, eps=1.0e-8):
    """
    Fourier-Mellin style scale-aware phase loss.

    img1, img2: torch.tensor (C, H, W) in [0, 1]
    """
    gray1 = _to_grayscale(img1)
    gray2 = _to_grayscale(img2)

    gray1 = gray1 - gray1.mean()
    gray2 = gray2 - gray2.mean()

    fft1 = torch.fft.fft2(gray1, norm="ortho")
    fft2 = torch.fft.fft2(gray2, norm="ortho")
    translation_phase_loss = 1.0 - _phase_only_correlation(fft1, fft2, eps)

    magnitude1 = torch.log1p(_fftshift2d(fft1.abs()))
    magnitude2 = torch.log1p(_fftshift2d(fft2.abs()))
    log_polar1 = _log_polar_sample(magnitude1)
    log_polar2 = _log_polar_sample(magnitude2)
    log_polar1 = log_polar1 - log_polar1.mean()
    log_polar2 = log_polar2 - log_polar2.mean()

    fm_fft1 = torch.fft.fft2(log_polar1, norm="ortho")
    fm_fft2 = torch.fft.fft2(log_polar2, norm="ortho")
    scale_rotation_phase_loss = 1.0 - _phase_only_correlation(fm_fft1, fm_fft2, eps)
    return 0.5 * (translation_phase_loss + scale_rotation_phase_loss)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    img1: torch.tensor (C, H, W)
    img2: torch.tensor (C, H, W)
    """
    channel = img1.shape[0]
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)



class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, labels, logits = None, preds = None):
        if logits is not None:
            preds = torch.nn.functional.softmax(logits, dim=1)
        elif preds is not None:
            pass
        labels = labels.float()
        eps = 1e-7
        loss_y1 = -1 * (1 - self.alpha) * torch.pow(1 - preds, self.gamma) * torch.log(preds + eps) * labels
        loss_y0 = -1 * self.alpha * torch.pow(1 - (1 - preds), self.gamma) * torch.log((1 - preds) + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return loss.mean()

class BinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bce_logit = torch.nn.BCEWithLogitsLoss()
        self.bce = torch.nn.BCELoss()

    def forward(self, labels, logits = None, preds = None):
        if logits is not None:
            onehot = torch.nn.functional.one_hot(labels, num_classes=2)
            loss =  - torch.log_softmax(logits, dim=1) * onehot
            loss = loss.mean()
        elif preds is not None:
            loss = self.bce(preds, labels.float())
        return loss


def lovasz_hinge(logits, labels):
    
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    losses = torch.dot(errors_sorted, gt_sorted.float())
    return losses
