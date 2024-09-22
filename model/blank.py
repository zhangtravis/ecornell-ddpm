import torch.nn as nn
from model.attention import Attention, AttentionBlock
import math
import torch

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=None):
        super(ResidualBlock, self).__init__()
        self.embed_time = TimeEmbedding(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU()
        
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
            
        self.dropout_val = dropout_val
        if dropout_val != None:
            self.dropout = nn.Dropout(dropout_val)
        
    def forward(self, x, t):
        skip = self.skip(x)
        time_embed = self.embed_time(t)
        x = self.conv1(self.silu(x))
#         import ipdb; ipdb.set_trace()
        x += time_embed[:, :, None, None]
        x = self.silu(self.norm(x))
        if self.dropout_val != None:
            x = self.dropout(x)
        x = self.conv2(x)
        
        return x + skip
    
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=None, use_attn=False, n_heads=8, dim_head=None):
        super(UNetDown, self).__init__()
        
        self.residual_block = ResidualBlock(in_channels, out_channels, dropout_val)
        self.use_attn = use_attn
        if self.use_attn:
            self.attention = Attention(out_channels)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x, t):
        x = self.residual_block(x, t)
        if self.use_attn:
            x = self.attention(x)
        return self.maxpool(x)
    
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=None, use_attn=False, n_heads=8, dim_head=None):
        super(UNetUp, self).__init__()
        
        self.convTranspose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.use_attn = use_attn
        if self.use_attn:
            self.attention = Attention(out_channels)
        self.residual_block = ResidualBlock(out_channels, out_channels, dropout_val)
            
    def forward(self, x, skip_connection, t):
        x = torch.cat((x, skip_connection), dim=1)
        x = self.convTranspose(x)
        if self.use_attn:
            x = self.attention(x)
        return self.residual_block(x, t)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=256, dropout_val=None, n_heads=4, dim_head=None):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        
        self.first_block = ResidualBlock(in_channels, n_feat, dropout_val)
        
        self.down1 = UNetDown(n_feat, n_feat, None, False, n_heads, n_feat)
        self.down2 = UNetDown(n_feat, 2 * n_feat, dropout_val, False, n_heads, 2 * n_feat)
        self.down3 = UNetDown(2 * n_feat, 2 * n_feat, None, True, n_heads, 2 * n_feat)
        
        self.latent = nn.Sequential(nn.AvgPool2d(4), nn.SiLU())
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.SiLU(),
        )
        self.up1 = UNetUp(4 * n_feat, 2 * n_feat, dropout_val, False, n_heads, 2 * n_feat) # 4 * n_feat for the skip connection
        self.up2 = UNetUp(4 * n_feat, n_feat, None, False, n_heads, n_feat)
        self.up3 = UNetUp(2 * n_feat, n_feat, dropout_val, True, n_heads, n_feat)
        self.pred = nn.Conv2d(2 * n_feat, self.out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        x = self.first_block(x, t)
        
        down1 = self.down1(x, t)
        down2 = self.down2(down1, t)
        down3 = self.down3(down2, t)
        
        latent_vector = self.latent(down3)
        
        up0 = self.up0(latent_vector)
        up1 = self.up1(up0, down3, t)
        up2 = self.up2(up1, down2, t)
        up3 = self.up3(up2, down1, t)
        
        pred = self.pred(torch.cat((up3, x), 1))
        
        return pred
        