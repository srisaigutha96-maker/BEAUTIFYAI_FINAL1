import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================
# Feature Pyramid Network (Fixed)
# ================================
class PreEncoderFPN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

        # All pyramid levels unified to 256 channels
        self.lat4 = nn.Conv2d(base_channels * 8, 256, 1)
        self.lat3 = nn.Conv2d(base_channels * 4, 256, 1)
        self.lat2 = nn.Conv2d(base_channels * 2, 256, 1)
        self.lat1 = nn.Conv2d(base_channels, 256, 1)

        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth1 = nn.Conv2d(256, 256, 3, 1, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)

        p4 = self.lat4(c4)

        p4_up = F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.smooth3(self.lat3(c3) + p4_up)

        p3_up = F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.smooth2(self.lat2(c2) + p3_up)

        p2_up = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self.smooth1(self.lat1(c1) + p2_up)

        return p1, p2, p3, p4


# ================================
# StyleGAN2-style Modulated Conv
# ================================
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.modulation = nn.Linear(style_dim, in_channels)
        self.modulation.weight.data.normal_()
        self.modulation.bias.data.fill_(1.0)

    def forward(self, x, style):
        batch, in_c, height, width = x.shape

        style = self.modulation(style).view(batch, 1, in_c, 1, 1)
        weight = self.weight.unsqueeze(0) * style

        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum([2, 3, 4], keepdim=True) + 1e-8)
            weight = weight * d

        x = x.view(1, batch * in_c, height, width)
        weight = weight.view(
            batch * self.out_channels,
            in_c,
            self.kernel_size,
            self.kernel_size
        )

        padding = self.kernel_size // 2
        out = F.conv2d(x, weight, padding=padding, groups=batch)
        out = out.view(batch, self.out_channels, height, width)

        return out


# ================================
# Style Extractor
# ================================
class StyleExtractor(nn.Module):
    def __init__(self, in_channels=3, style_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, style_dim)
        )

    def forward(self, x):
        return self.net(x)


# ================================
# Generator (Fixed)
# ================================
class Generator(nn.Module):
    def __init__(self, style_dim=512):
        super().__init__()

        self.fpn = PreEncoderFPN()
        self.style_ext = StyleExtractor(style_dim=style_dim)

        self.mapping = nn.Sequential(
            nn.Linear(style_dim + 1, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim)
        )

        # All pyramid levels = 256
        self.stem = nn.Parameter(torch.randn(1, 256, 32, 32))

        self.block4 = ModulatedConv2d(256, 256, 3, style_dim)
        self.block3 = ModulatedConv2d(256, 256, 3, style_dim)
        self.block2 = ModulatedConv2d(256, 256, 3, style_dim)
        self.block1 = ModulatedConv2d(256, 64, 3, style_dim)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x, intensity=1.0):
        b = x.size(0)

        p1, p2, p3, p4 = self.fpn(x)

        style = self.style_ext(x)

        if isinstance(intensity, torch.Tensor):
            int_vec = intensity
        else:
            int_vec = torch.full((b, 1), float(intensity), device=x.device)
        style = torch.cat([style, int_vec], dim=1)
        w = self.mapping(style)

        out = self.stem.repeat(b, 1, 1, 1)

        # ---- p4 ----
        p4 = F.interpolate(p4, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = out + p4
        out = F.leaky_relu(self.block4(out, w), 0.2)

        # ---- p3 ----
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = out + p3
        out = F.leaky_relu(self.block3(out, w), 0.2)

        # ---- p2 ----
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = F.interpolate(p2, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = out + p2
        out = F.leaky_relu(self.block2(out, w), 0.2)

        # ---- p1 ----
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = F.interpolate(p1, size=out.shape[2:], mode='bilinear', align_corners=False)
        out = out + p1
        out = F.leaky_relu(self.block1(out, w), 0.2)

        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return self.to_rgb(out)

