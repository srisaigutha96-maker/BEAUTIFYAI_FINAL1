import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    Discriminates patches of the face 512x512 image rather than the whole image.
    Enforces high-frequency detail synthesis such as skin pores, hair strands.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        def conv_block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
            
        self.net = nn.Sequential(
            conv_block(in_channels, base_channels, normalize=False),           # (256, 256)
            conv_block(base_channels, base_channels * 2),                      # (128, 128)
            conv_block(base_channels * 2, base_channels * 4),                  # (64, 64)
            conv_block(base_channels * 4, base_channels * 8, stride=1),        # (63, 63)
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1)                           # (62, 62)
        )
        
    def forward(self, x):
        return self.net(x)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.d1 = PatchGANDiscriminator(in_channels)
        self.d2 = PatchGANDiscriminator(in_channels)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        # Forward pass on original resolution
        out1 = self.d1(x)
        # Forward pass on downsampled resolution
        out2 = self.d2(self.downsample(x))
        return [out1, out2]
