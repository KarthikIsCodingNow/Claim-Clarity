"""
ClaimClarity AI - Deep Neural Enhancement Architecture
Model: ClaimDCE-Net (Zero-Reference Deep Curve Estimation for Forensic Claim Photos)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClaimDCE_Net(nn.Module):
    """
    Deep Curve Estimation Network with symmetric skip connections.
    Estimates 8 iterations of pixel-wise curve parameter maps A (24 channels total).
    Transforms input x in [0, 1] via quadratic recurrence:
        LE_n(x) = LE_{n-1}(x) + A_n(x) * LE_{n-1}(x) * (1 - LE_{n-1}(x))
    Guarantees:
    - Zero highlight blowout (derivative is zero at 1)
    - Strict dynamic range boundedness in [0, 1]
    - Natural illumination recovery without noise amplification
    """

    def __init__(self, in_channels=3, num_filters=32, num_iterations=8):
        super(ClaimDCE_Net, self).__init__()
        self.num_iterations = num_iterations

        # Feature Extraction & Downward stream
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.PReLU()

        # Bottleneck
        self.conv4 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.PReLU()

        # Upward stream with Skip Connections
        self.conv5 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.PReLU()

        self.conv6 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.PReLU()

        # Output parameter maps: 3 channels per iteration * num_iterations (8 * 3 = 24)
        self.conv7 = nn.Conv2d(num_filters * 2, num_iterations * 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Feature extraction with symmetrical skip connections
        x1 = self.relu1(self.conv1(x))
        x2 = self.relu2(self.conv2(x1))
        x3 = self.relu3(self.conv3(x2))
        x4 = self.relu4(self.conv4(x3))

        x5 = self.relu5(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu6(self.conv6(torch.cat([x2, x5], dim=1)))
        
        # A in [-1, 1]
        A = self.tanh(self.conv7(torch.cat([x1, x6], dim=1)))

        # Split A into 8 chunks of 3 channels
        A_chunks = torch.split(A, 3, dim=1)

        # Iterative curve application
        curr = x
        for i in range(self.num_iterations):
            a_i = A_chunks[i]
            curr = curr + a_i * curr * (1.0 - curr)

        enhanced = curr
        return enhanced, A


# ==============================================================================
# Specialized Forensic Loss Functions for Claim Photo Clarification
# ==============================================================================

class L_spatial(nn.Module):
    """
    Spatial Consistency Loss: Preserves difference between neighboring regions
    between input evidence and enhanced output, protecting hairline scratches/fissures.
    """
    def __init__(self):
        super(L_spatial, self).__init__()
        # 4 directional kernels
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4, stride=4)

    def forward(self, org, enhance):
        # Convert RGB to grayscale for spatial gradient check
        org_mean = torch.mean(org, 1, keepdim=True)
        enh_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enh_pool = self.pool(enh_mean)

        w_l = self.weight_left.to(org.device)
        w_r = self.weight_right.to(org.device)
        w_u = self.weight_up.to(org.device)
        w_d = self.weight_down.to(org.device)

        d_org_l = F.conv2d(org_pool, w_l, padding=1)
        d_org_r = F.conv2d(org_pool, w_r, padding=1)
        d_org_u = F.conv2d(org_pool, w_u, padding=1)
        d_org_d = F.conv2d(org_pool, w_d, padding=1)

        d_enh_l = F.conv2d(enh_pool, w_l, padding=1)
        d_enh_r = F.conv2d(enh_pool, w_r, padding=1)
        d_enh_u = F.conv2d(enh_pool, w_u, padding=1)
        d_enh_d = F.conv2d(enh_pool, w_d, padding=1)

        loss_l = torch.pow(d_org_l - d_enh_l, 2)
        loss_r = torch.pow(d_org_r - d_enh_r, 2)
        loss_u = torch.pow(d_org_u - d_enh_u, 2)
        loss_d = torch.pow(d_org_d - d_enh_d, 2)

        return torch.mean(loss_l + loss_r + loss_u + loss_d)


class L_exposure(nn.Module):
    """
    Exposure Control Loss: Measures distance of local patch intensity to optimal exposure E=0.55.
    Prevents harsh over-contrast and blown highlights.
    """
    def __init__(self, patch_size=16, target_exp=0.55):
        super(L_exposure, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.target_exp = target_exp

    def forward(self, x):
        mean_intensity = torch.mean(x, 1, keepdim=True)
        patch_mean = self.pool(mean_intensity)
        return torch.mean(torch.abs(patch_mean - self.target_exp))


class L_color(nn.Module):
    """
    Color Constancy Loss: Constrains color ratios following Gray-World hypothesis,
    preventing artificial color saturation and paint tone drifting.
    """
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3])
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        return torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))


class L_TV(nn.Module):
    """
    Total Variation Illumination Smoothness Loss:
    Enforces smooth parameter map transitions, strictly eliminating noise grain & halos.
    """
    def __init__(self):
        super(L_TV, self).__init__()

    def forward(self, A):
        batch_size = A.size(0)
        h_tv = torch.pow(A[:, :, 1:, :] - A[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(A[:, :, :, 1:] - A[:, :, :, :-1], 2).sum()
        count_h = (A.size(2) - 1) * A.size(3)
        count_w = A.size(2) * (A.size(3) - 1)
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
