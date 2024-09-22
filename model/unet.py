import torch.nn as nn
from model.attention import Attention, GTAttention
import math
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=3):
        super(ResidualBlock, self).__init__()
#         self.t_embed_dim = t_embed_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        ]
        
        self.first = nn.Sequential(*layers)
        
        # input layer to go from in_channel --> out_channel
        
        for _ in range(n_layers - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.ReLU())
            
        self.residual_block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.first(x) + self.residual_block(x)
        
        return out / 1.414
    
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=3, n_heads=8, dim_head=None):
        super(UNetDown, self).__init__()
        
#         import ipdb; ipdb.set_trace()
        layers = [
            ResidualBlock(in_channels, out_channels, n_layers),
            Attention(out_channels, n_heads, dim_head),
            nn.MaxPool2d(2)
        ]
        
#         in_dim = hidden_dims[0]
#         for hidden_dim in range(hidden_dims[1:]):
#             layers.append(ResidualBlock(in_dim, in_dim, n_layers))
#             layers.append(Attention(in_dim, n_heads, dim_head))
#             layers.append(nn.MaxPool2d(2))
#             in_dim = hidden_dim
            
#         layers.append(ResidualBlock(hidden_dim, out_channels, n_layers))
#         layers.append(Attention(out_channels, n_heads, dim_head))
#         layers.append(nn.MaxPool2d(2))
            
        self.down = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.down(x)
    
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=3, n_heads=8, dim_head=None):
        super(UNetUp, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            Attention(out_channels, n_heads, dim_head),
            ResidualBlock(out_channels, out_channels, n_layers),
            ResidualBlock(out_channels, out_channels, n_layers)
        ]
        
#         in_dim = hidden_dims[0]
#         for hidden_dim in range(hidden_dims[1:]):
#             layers.append(nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2))
#             layers.append(Attention(in_dim, n_heads, dim_head))
#             layers.append(ResidualBlock(in_dim, in_dim, n_layers))
#             layers.append(ResidualBlock(in_dim, in_dim, n_layers))
#             in_dim = hidden_dim
            
        self.up = nn.Sequential(*layers)
            
    def forward(self, x, skip_connection):
        x = torch.cat((x, skip_connection), dim=1)
        return self.up(x)

class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(TimeEmbedding, self).__init__()
        self.lin1 = nn.Linear(1, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t):
        t = t.view(-1, 1)
        t = torch.sin(self.lin1(t))
        t = self.lin2(t)
        return t

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=256, n_layers=3, n_heads=4, dim_head=None):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        
        self.first_block = ResidualBlock(in_channels, n_feat, n_layers)
        
        self.down1 = UNetDown(n_feat, n_feat, n_layers, n_heads, n_feat)
        self.down2 = UNetDown(n_feat, 2 * n_feat, n_layers, n_heads, 2 * n_feat)
        self.down3 = UNetDown(2 * n_feat, 2 * n_feat, n_layers, n_heads, 2 * n_feat)
        
        self.latent = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())
        
        self.time_embedding = TimeEmbedding(2 * n_feat)
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.SiLU(),
        )
        self.up1 = UNetUp(4 * n_feat, 2 * n_feat, n_layers, n_heads, 2 * n_feat) # 4 * n_feat for the skip connection
        self.up2 = UNetUp(4 * n_feat, n_feat, n_layers, n_heads, n_feat)
        self.up3 = UNetUp(2 * n_feat, n_feat, n_layers, n_heads, n_feat)
        self.pred = nn.Conv2d(2 * n_feat, self.out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        x = self.first_block(x)
        
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        
        latent_vector = self.latent(down3)
        time_embed = self.time_embedding(t).view(-1,2 * self.n_feat, 1, 1)
        
        up0 = self.up0(latent_vector + time_embed)
        up1 = self.up1(up0, down3) + time_embed
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        
        pred = self.pred(torch.cat((up3, x), 1))
        
        return pred
        