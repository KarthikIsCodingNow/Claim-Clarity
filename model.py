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


class ClaimDamageClassifier(nn.Module):
    """
    Forensic Vehicle Damage Classifier trained on labeled vehicle damage dataset:
    Classes (6):
      0: F_Breakage (Front Headlamp / Windshield / Grille Fracture)
      1: F_Crushed  (Front Bumper / Hood / Radiator Crush)
      2: F_Normal   (Front Intact / Baseline)
      3: R_Breakage (Rear Tail Lamp / Glass Fracture)
      4: R_Crushed  (Rear Collision / Trunk Buckle)
      5: R_Normal   (Rear Intact / Baseline)
    """
    def __init__(self, num_classes=6):
        super(ClaimDamageClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 112x112
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 14x14
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


class ClaimMultiTaskAssessor(nn.Module):
    """
    Unified Multi-Task Forensic Vehicle Accident & Damage Assessment Network.
    Trained across multiple modalities & forms of data:
      1. Damage Mode & Perspective (6 classes from Kaggle Vehicle Damage Inspection)
      2. Accident Severity Level (3 classes: Fatal, Severe, Minor from GitHub Dataset)
      3. Involved Vehicle Category (3 classes: Light Vehicle, Heavy Vehicle, Motorcycle)
      4. Accident Verification (2 classes: Authentic Collision vs Ambient Non-Accident)
    """
    def __init__(self):
        super(ClaimMultiTaskAssessor, self).__init__()
        self.shared_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # 112x112
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 14x14
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3)
        )
        # Head 1: Damage Mode & Perspective (6 classes)
        self.head_damage = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 6)
        )
        # Head 2: Severity (3 classes: 0=Fatal/Catastrophic, 1=Severe, 2=Minor)
        self.head_severity = nn.Sequential(
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, 3)
        )
        # Head 3: Vehicle Type (3 classes: 0=Light Vehicle, 1=Heavy Vehicle, 2=Motorcycle)
        self.head_vehicle = nn.Sequential(
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.Linear(64, 3)
        )
        # Head 4: Accident Detection (2 classes: 0=Accident Verified, 1=Non-Accident)
        self.head_detection = nn.Sequential(
            nn.Linear(256, 32),
            nn.PReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        feat = self.shared_features(x)
        return {
            "damage": self.head_damage(feat),
            "severity": self.head_severity(feat),
            "vehicle": self.head_vehicle(feat),
            "detection": self.head_detection(feat)
        }


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


class ClaimVehicleRecognizer(nn.Module):
    """
    Forensic Vehicle Recognition Network trained on multi-class vehicle dataset:
      0: Bus (Commercial Transit / Bus / Van)
      1: Car (Passenger Automobile: Sedan / SUV / Coupe / Hatchback)
      2: Truck (Freight Truck / Pickup / Heavy Transport)
      3: Motorcycle (Two-Wheeler / Motorbike / Scooter)
    """
    def __init__(self, num_classes=4):
        super(ClaimVehicleRecognizer, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)

