import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 


def scale_dot_product(q, k, v, mask=None):    
    d_k = q.size()[-1] # HeadDim
    
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) #QK^T
    attn_logits = attn_logits/math.sqrt(d_k) # scaling
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask==0, -9e15)
    attention = F.softmax(attn_logits, dim=-1) #softmax
    values = torch.matmul(attention, v) # AV
    
    return values, attention
    

def expand_mask(mask):
    assert mask.ndim >= 2, \
    "Mask must be at least 2-dimensional with seq_len x seq_len"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        # validation 
        assert embed_dim % num_heads == 0, \
              "Embedding dimension must be 0 modulo number of heads"
        # configuration
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Project layers
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, input_dim)

    def forward(self, x, mask=None, return_attention=False):
        # x: [B, L, D]
        batch_size, seq_len, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv_proj = self.qkv_proj(x)

        # split Q, K, V from linear output
        qkv = qkv_proj.reshape(batch_size, seq_len, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [B, H, L, D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scale_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [B, L, H, D]
        values = values.reshape(batch_size, seq_len, self.embed_dim)
        o = self.o_proj(values)
        
        if return_attention:
            return o, attention
        else:
            return o

        