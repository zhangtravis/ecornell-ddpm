import torch.nn as nn
from model.attention import Attention
import math
import torch

class TimeEmbedding(nn.Module):
    """
    From paper "Attention Is All You Need", section 3.5
    """
    def __init__(self, dimension, max_timesteps=1001):
        super(TimeEmbedding, self).__init__()
        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension
        self.pe_matrix = torch.zeros(max_timesteps, dimension)
        # Gather all the even dimensions across the embedding vector
        even_indices = torch.arange(0, self.dimension, 2)
        # Calculate the term using log transforms for faster calculations
        # (https://stackoverflow.com/questions/17891595/pow-vs-exp-performance)
        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        # Precompute positional encoding matrix based on odd/even timesteps
        timesteps = torch.arange(max_timesteps).unsqueeze(1)
        self.pe_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pe_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep, T):
        # [bs, d_model]
        return self.pe_matrix[(timestep * T).to(torch.int32).cpu()].to(timestep.device)

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
        
    def forward(self, x, t, T):
        skip = self.skip(x)
        time_embed = self.embed_time(t, T)
        x = self.conv1(self.silu(x))
#         import ipdb; ipdb.set_trace()
        if(len(time_embed.shape) == len(x.shape)):
            x += time_embed
        else:
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
#             self.attention = AttentionBlock(out_channels)
            self.attention = Attention(out_channels, n_heads, dim_head)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x, t, T):
        x = self.residual_block(x, t, T)
        if self.use_attn:
            x = self.attention(x)
        return self.maxpool(x)
    
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=None, use_attn=False, n_heads=8, dim_head=None):
        super(UNetUp, self).__init__()
        
        self.convTranspose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.use_attn = use_attn
        if self.use_attn:
#             self.attention = AttentionBlock(out_channels)
            self.attention = Attention(out_channels, n_heads, dim_head)
        self.residual_block = ResidualBlock(out_channels, out_channels, dropout_val)
            
    def forward(self, x, skip_connection, t, T):
#         import ipdb; ipdb.set_trace()
        x = torch.cat((x, skip_connection), dim=1)
        x = self.convTranspose(x)
        if self.use_attn:
            x = self.attention(x)
        return self.residual_block(x, t, T)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=256, dropout_val=None, n_heads=2, dim_head=None):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        
        self.first_block = ResidualBlock(in_channels, n_feat, dropout_val)
        
        self.down1 = UNetDown(n_feat, n_feat, None, False, n_heads, n_feat)
        self.down2 = UNetDown(n_feat, 2 * n_feat, dropout_val, False, n_heads, 2 * n_feat)
        self.down3 = UNetDown(2 * n_feat, 4 * n_feat, None, True, n_heads, 4 * n_feat)
        self.down4 = UNetDown(4 * n_feat, 8 * n_feat, None, False, n_heads, 8 * n_feat)
        self.down5 = UNetDown(8 * n_feat, 8 * n_feat, None, False, n_heads, 8 * n_feat)
        
        self.latent = nn.Sequential(nn.SiLU())
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(8 * n_feat, 8 * n_feat, 2, 2),
            nn.GroupNorm(8, 8 * n_feat),
            nn.SiLU(),
        )
        self.up1 = UNetUp(16 * n_feat, 4 * n_feat, dropout_val, False, n_heads, 8 * n_feat) # 4 * n_feat for the skip connection
        self.up2 = UNetUp(8 * n_feat, 2 * n_feat, None, False, n_heads, 2 * n_feat)
        self.up3 = UNetUp(4 * n_feat, n_feat, dropout_val, True, n_heads, n_feat)
        self.up4 = UNetUp(2 * n_feat, n_feat, dropout_val, False, n_heads, n_feat)
#         self.up5 = UNetUp(4 * n_feat, 2 * n_feat, dropout_val, True, n_heads, 2 * n_feat)
        self.pred = nn.Conv2d(2 * n_feat, self.out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t, T):
        
        x = self.first_block(x, t, T)
        
        down1 = self.down1(x, t, T)
        down2 = self.down2(down1, t, T)
        down3 = self.down3(down2, t, T)
        down4 = self.down4(down3, t, T)
#         import ipdb; ipdb.set_trace()
        down5 = self.down5(down4, t, T)
        
#         import ipdb; ipdb.set_trace()
        
        latent_vector = self.latent(down5)
        
        up0 = self.up0(latent_vector)
        up1 = self.up1(up0, down4, t, T)
        up2 = self.up2(up1, down3, t, T)
#         import ipdb; ipdb.set_trace()
        up3 = self.up3(up2, down2, t, T)
        up4 = self.up4(up3, down1, t, T)
#         up5 = self.up5(up4, down1, t, T)
        
        pred = self.pred(torch.cat((up4, x), 1))
        
        return pred
        