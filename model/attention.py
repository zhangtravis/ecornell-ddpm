import torch.nn as nn
# from torchtune.modules import RMSNorm
import torch.nn.functional as F
import torch

from torch import einsum
from einops import rearrange
import math

class RMSNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1]**0.5)

class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, dim_head=None):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim_head = dim // n_heads if dim_head is None else dim_head
        hidden_dim = self.dim_head * n_heads
        self.dim = dim
        
        self.rms_norm = RMSNorm(dim)
        
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False) # times 3 for q, k, and v
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        
    
    def forward(self, x):
#         import ipdb; ipdb.set_trace()
        bs, channel, height, width = x.shape # batch_size, channel, height, width
        
        qkv = self.to_qkv(self.rms_norm(x)).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(bs, self.n_heads, channel, height * width), qkv)
        
        out = F.scaled_dot_product_attention(q, k, v)
        
        # Permute to bring the flattened spatial dimension to the last position
        out = out.permute(0, 1, 3, 2).contiguous()

        # Reshape to merge the heads and dimensions, and separate the spatial dimensions
        out = out.view(bs, self.n_heads * self.dim_head, height, width)
        
        return self.to_out(out) + x
    
# class AttentionBlock(nn.Module):

#     def __init__(
#             self,
#             in_channels,
#             mid_channels=None,
#             out_channels=None
#     ):
#         super(AttentionBlock, self).__init__()
#         mid_channels = mid_channels or in_channels
#         out_channels = out_channels or in_channels
#         self.norm = nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6)
#         self.project_in = nn.Conv2d(in_channels, 3 * mid_channels, 1)
#         self.project_out = nn.Conv2d(mid_channels, out_channels, 1)
#         self.in_channels = in_channels
#         self.mid_channels = mid_channels
#         self.out_channels = out_channels
#         self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

#     @staticmethod
#     def qkv(q, k, v):
#         B, C, H, W = q.shape
#         w = torch.einsum("bchw, bcHW -> bhwHW", q, k)
#         w = torch.softmax(
#             w.reshape(B, H, W, H * W) / math.sqrt(C), dim=-1
#         ).reshape(B, H, W, H, W)
#         out = torch.einsum("bhwHW, bcHW -> bchw", w, v)  # this will break the contiguity -> impaired performance
#         return out.contiguous()  # force to return a contiguous tensor

#     def forward(self, x, **kwargs):
#         skip = self.skip(x)
#         C = x.shape[1]
#         assert C == self.in_channels
#         q, k, v = self.project_in(self.norm(x)).chunk(3, dim=1)
#         x = self.qkv(q, k, v)
#         x = self.project_out(x)
#         return x + skip

# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         heads = 4,
#         dim_head = 32,
#         num_mem_kv = 4,
#     ):
#         super().__init__()
#         self.heads = heads
#         hidden_dim = dim_head * heads

#         self.norm = RMSNorm(dim)
#         self.attend = Attend(flash = flash)

#         self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)

#     def forward(self, x):
#         b, c, h, w = x.shape

#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim = 1)
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

#         mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
#         k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

#         out = self.attend(q, k, v)

#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
#         return self.to_out(out)

# class AttentionBlock(nn.Module):
#     def __init__(self, in_channels, mid_channels=None, out_channels=None):
#         super(AttentionBlock, self).__init__()
#         mid_channels = mid_channels if mid_channels else in_channels
#         out_channels = out_channels if out_channels else in_channels
        
#         self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        
#         self.to_qkv = nn.Conv2d(in_channels, 3 * mid_channels, 1)
#         self.out = nn.Conv2d(mid_channels, out_channels, 1)
        
#         self.in_channels = in_channels
#         self.mid_channels = mid_channels
#         self.out_channels = out_channels
        
#         self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
    
#     @staticmethod
#     def compute_output(q, k, v):
#         B, C, H, W = q.shape

#         # Reshape q and k for matrix multiplication
#         q_reshaped = q.view(B, C, H * W).transpose(1, 2)  # Shape: (B, H*W, C)
#         k_reshaped = k.view(B, C, H * W)  # Shape: (B, C, H*W)

#         # Compute attention weights
#         w = torch.bmm(q_reshaped, k_reshaped)  # Shape: (B, H*W, H*W)
#         w = w.view(B, H, W, H * W)

#         # Apply softmax
#         w = torch.softmax(w / math.sqrt(C), dim=-1)
#         w = w.view(B, H, W, H, W)

#         # Reshape v for matrix multiplication
#         v_reshaped = v.view(B, C, H * W)

#         # Compute output
#         w_reshaped = w.view(B, H * W, H * W)
#         out = torch.bmm(v_reshaped, w_reshaped.transpose(1, 2))
#         out = out.view(B, C, H, W)

#         return out.contiguous()
        
#     def forward(self, x):
#         skip = self.skip(x)
#         C = x.shape[1]
#         assert C == self.in_channels
#         q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
#         out = self.compute_output(q, k, v)
#         out = self.out(out)
        
#         return out + skip
    
if __name__ == '__main__':
    # Example usage
    b, c, h, w = 2, 32, 8, 8
    x = torch.randn(b, c, h, w)
    attention = Attention(dim=c, n_heads=4, dim_head=None)
    gt_attention = GTAttention(dim=c, heads=4)
#     output = attention(x)
    gt_output = gt_attention(x)
#     print(output.shape)  # Should be (b, c, h, w)
    print(gt_output.shape)