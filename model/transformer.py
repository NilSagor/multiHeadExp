import torch 
import torch.nn as nn

from .attention import MultiheadAttention

class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_forward, dropout):
        """
        Args:
            input_dim: Dimensionality of the input
            num_heads: Number of heads 
            dim_forward: Dimensionality of the hidden layer in the MLP
            dropout: Dropout probability to use in the dropout layers 
        """
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_forward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_forward, input_dim)
        )

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, input_dim, dim_forward, num_heads, dropout):        
        super().__init__()
        block_config = {
            "input_dim": input_dim,
            "num_heads": num_heads,
            "dim_forward": dim_forward,
            "dropout": dropout
        }

        self.layers = nn.ModuleList([EncoderBlock(**block_config) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x
    
    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            # compute attention and hidden states in one pass
            x, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            
            x = l.norm1(x + l.dropout(x))
            x = l.linear_net(x) + x
            x = l.norm2(x)
            attention_maps.append(attn_map)            
        return attention_maps
    

