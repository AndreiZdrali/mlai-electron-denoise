import torch
from torch import nn
from torch.nn import functional as F

class ChannelAttentionModule(nn.Module):
  def __init__(self):
    super(ChannelAttentionModule, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    self.fc = nn.Sequential(
        nn.Conv2d(2, 1, 1, bias=False),
        nn.Sigmoid()
    )

  def forward(self, x):
    avg_out = self.avg_pool(x)
    max_out = self.max_pool(x)
    out = torch.cat([avg_out, max_out], dim=1)
    out = self.fc(out)
    return x * out

class KernelSelectingModule(nn.Module):
  def __init__(self, in_channels, channels):
    super(KernelSelectingModule, self).__init__()
    self.conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.conv(x)
    out = self.softmax(out)
    return out

class PRIDNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(PRIDNet, self).__init__()

    # U-Net
    self.unet = nn.Sequential(
      # Encoder
      nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.MaxPool2d(2, stride=2),

      nn.Conv2d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.MaxPool2d(2, stride=2),

      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.MaxPool2d(2, stride=2),

      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, padding=1),

      # Decoder
      nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=3, padding=1),
      nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    )

    # Channel Attention Module
    self.cam = ChannelAttentionModule()

    # Kernel Selecting Module
    self.ksm = KernelSelectingModule(32, 8)

  def forward(self, x):
    out = self.unet(x)

    # Kernel Selection
    # 1. Extract bottleneck features from the encoder (modify if needed)
    bottleneck = out  # Assuming bottleneck features are at the end of the encoder

    # 2. Apply Kernel Selecting Module
    kernels = self.ksm(bottleneck)  # Get weights for each kernel

    # 3. Expand kernels (assuming kernels are defined elsewhere)
    expanded_kernels = []  # Placeholder for expanded kernels
    for i in range(len(kernels[0])):  # Iterate through kernel weights
      # Assuming you have pre-defined kernels (kernel1, kernel2, ...)
      # Expand the weights from a 1x1 kernel to match the actual kernel size
      expanded_kernels.append(nn.functional.conv2d(kernel[i].unsqueeze(1), weight=torch.ones(1, 1, *kernel_size).cuda()))

    # 4. Apply weighted kernels (replace with your desired operation)
    weighted_features = torch.zeros_like(out)  # Placeholder for weighted features
    for i in range(len(expanded_kernels)):
      weighted_features += kernels[:, i, 0, 0].unsqueeze(1) * nn.functional.conv2d(out, expanded_kernels[i])

    # 5. Combine with original features (replace with your desired operation)
    out = self.cam(weighted_features + out)  # Apply CAM and combine

    # Rest of the decoder (same as before)
    # ... your decoder architecture here ...

    return out