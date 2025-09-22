import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from GUIEmbeddingModule import GUIEmbeddingModule

def relative_positional_bucket(delta, num_buckets=32, max_distance=128, log_base=2):
    sign = torch.sign(delta)
    n = torch.abs(delta)
    n = torch.clamp(n, min=1e-6, max=max_distance)
    log_index = (torch.log(n) / math.log(log_base)).floor().long()
    log_index = torch.clamp(log_index, min=0, max=num_buckets - 1)
    return log_index * sign.to(torch.long) + num_buckets - 1

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_buckets=32):
        super().__init__()

        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.num_buckets = num_buckets

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.rpe_bias_x = nn.Embedding(num_buckets * 2 - 1, num_heads)
        self.rpe_bias_y = nn.Embedding(num_buckets * 2 - 1, num_heads)

    def forward(self, query, key, value, coords, mask=None):
        B, G_q, D = query.size()
        G_k = key.size(1)

        Q = self.q_proj(query).view(B, G_q, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, G_k, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, G_k, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 2D Relative Positional Encoding
        coords_q = coords[:, :G_q, :]
        coords_k = coords[:, :G_k, :]
        delta = coords_q[:, :, None, :] - coords_k[:, None, :, :]
        delta_x, delta_y = delta[..., 0], delta[..., 1]

        bucket_x = relative_positional_bucket(delta_x, self.num_buckets)
        bucket_y = relative_positional_bucket(delta_y, self.num_buckets)

        bias_x = self.rpe_bias_x(bucket_x)
        bias_y = self.rpe_bias_y(bucket_y)
        rpe_bias = bias_x + bias_y
        rpe_bias = rpe_bias.permute(0, 3, 1, 2)

        scores = scores + rpe_bias

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, G_q, D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.3, num_buckets=32):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, num_buckets)
        self.ffn = FeedForward(embed_dim, d_ff, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, coords, mask=None):
        x_ = self.norm1(x)
        attn_out = self.self_attn(x_, x_, x_, coords, mask)
        x = x + self.dropout(attn_out)

        x_ = self.norm2(x)
        ff_out = self.ffn(x_)
        x = x + self.dropout(ff_out)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, d_ff, dropout=0.3, num_buckets=32):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, d_ff, dropout, num_buckets)
            for _ in range(num_layers)
        ])

    def forward(self, x, coords, mask=None):
        for layer in self.layers:
            x = layer(x, coords, mask)
        return x

class ScreenSBERT(nn.Module):
    def __init__(self, device, embed_dim=768, num_heads=8, num_layers=6, d_ff=768*4, dropout=0.3):
        super().__init__()

        self.device = device

        self.gui_embedding = GUIEmbeddingModule()
        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, d_ff, dropout)

    def encode(self, coords, types, visions, texts, padding_mask):        
        gui_tokens = self.gui_embedding(coords, types, visions, texts)

        center_coords = torch.zeros(coords[:, :, :2].shape).to(self.device)
        center_coords[:, :, 0] = (coords[:, :, 0] + coords[:, :, 2]) / 2
        center_coords[:, :, 1] = (coords[:, :, 1] + coords[:, :, 3]) / 2

        enc = self.transformer_encoder(gui_tokens, center_coords, padding_mask)
        
        # average pooling excluding padding tokens
        mask = padding_mask.unsqueeze(-1)
        masked_enc = enc * mask
        sum_enc = masked_enc.sum(dim=1)
        valid_token_counts = mask.sum(dim=1)
        screen_emb = sum_enc / valid_token_counts.clamp(min=1)

        return screen_emb

    def cosine_similarity(
        self, 
        coords1, types1, visions1, texts1, padding_mask1,
        coords2, types2, visions2, texts2, padding_mask2
    ):
        screen_emb1 = self.encode(coords1, types1, visions1, texts1, padding_mask1)
        screen_emb2 = self.encode(coords2, types2, visions2, texts2, padding_mask2)

        similarities = F.cosine_similarity(screen_emb1, screen_emb2, dim=1)

        return similarities