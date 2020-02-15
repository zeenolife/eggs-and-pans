import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ResNet50Encoder


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, residual):

        # Upsample first block
        x = F.interpolate(x, scale_factor=2)

        # Concatenate the residual
        if residual is not None:
            x = torch.cat([x, residual], dim=1)

        # Two conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):

    def __init__(self, classes, pretrained_encoder=True):

        super().__init__()

        self.encoder = ResNet50Encoder(pretrained=pretrained_encoder)

        # Defining channels for both encoder's residual channels and upsample channels
        self.encoder_channels = [0, 64, 256, 512, 1024, 2048]
        self.decoder_channels = [256, 128, 64, 32, 16]

        # Initialize upsamplers
        self.upsamplers = nn.ModuleList()
        for idx, upsampler_out in enumerate(self.decoder_channels):

            # Get ResNet's residual block channel size
            residual_ch = self.encoder_channels[len(self.encoder_channels) - idx - 2]

            # Get upsampled block channel size
            # Case 1: Target block is ResNet block
            if idx == 0:
                upsampler_in = self.encoder_channels[-1]

            # Case 2: Target bloack is upsampled block
            else:
                upsampler_in = self.decoder_channels[idx - 1]

            self.upsamplers.append(Upsampler(residual_ch + upsampler_in, upsampler_out))

        # Initialize segmentation head
        self.head = nn.Conv2d(self.decoder_channels[-1], classes, kernel_size=3, padding=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):

        for name, m in self.named_modules():

            # Skip encoder initialization, since they are already initialized
            if name.startswith('encoder'):
                continue
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Get encoder residuals
        residuals = self.encoder(x)
        residuals = [None] + residuals
        out = residuals[-1]

        # Forward through upsamplers
        for idx, upsampler in enumerate(self.upsamplers):

            # Get residual
            residual = residuals[len(residuals) - idx - 2]
            out = upsampler(out, residual)

        # Forward through segmentation head
        out = self.head(out)

        return out


if __name__ == '__main__':

    model = UNet(2)
    img = torch.rand((1, 3, 1024, 1024))
    out = model(img)
    print(out.shape)
