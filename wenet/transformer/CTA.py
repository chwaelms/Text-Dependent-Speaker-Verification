import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================
#    Channel Temporal Attention(CTA)
# ========================================
class ChannelTemporalAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3, middle_channels=8):
        super().__init__()
        
        self.in_channels = in_channels 
        self.middle_channels = middle_channels
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            middle_channels, 
            kernel_size, 
            padding=kernel_size // 2
        )
        
        self.conv2 = nn.Conv2d(
            middle_channels, 
            in_channels, 
            kernel_size, 
            padding=kernel_size // 2
        )
        
        self.bn = nn.BatchNorm2d(middle_channels)
    
    def forward(self, x):
        # print(f"Input Shape: {x.shape}")  # [1, 256, 53, 39] = [B, C, T, F]
        batch, channels, time, freq = x.size()
        
        if channels != self.in_channels:
            raise ValueError( 
                f"Expected {self.in_channels} input channels but got {channels}."
            )
            
        # z = x.mean(dim=3, keepdim=True)  # dim=2 -> (batch, channel, 1, freq)
        z = torch.sqrt(x.var(dim=3, keepdim=True) + 1e-5)
        # print(f"Mean Shape: {mean.shape}, STD Shape: {std.shape}")

        # print(f"After sqrt Shape: {z.shape}")
        z = self.conv1(z)
        # print(f"After conv1 Shape: {z.shape}")
        
        z = self.bn(z)
        z = F.relu(z)
        z = self.conv2(z)
        # print(f"After conv2 Shape: {z.shape}")
        
        z = F.relu(z)
        w = torch.sigmoid(z)  # (batch, in_channels, time, freq)
        # print(f"Final Attention Weights Shape: {w.shape}")
        
        return w