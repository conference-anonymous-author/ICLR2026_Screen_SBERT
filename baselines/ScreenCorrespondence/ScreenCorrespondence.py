import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def relative_position_bucket(delta, num_buckets=32, max_distance=128, log_base=2):
    sign = torch.sign(delta)
    n = torch.abs(delta)
    n = torch.clamp(n, min=1e-6, max=max_distance)
    log_index = (torch.log(n) / math.log(log_base)).floor().long()
    log_index = torch.clamp(log_index, min=0, max=num_buckets - 1)
    return log_index * sign.to(torch.long) + num_buckets - 1

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_buckets=32):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads
        self.num_buckets = num_buckets

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Not shared → each head has its own bias
        self.rpe_bias_x = nn.Embedding(num_buckets * 2 - 1, num_heads)
        self.rpe_bias_y = nn.Embedding(num_buckets * 2 - 1, num_heads)

    def forward(self, query, key, value, coords, mask=None):
        B, G_q, D = query.size()
        G_k = key.size(1)

        Q = self.q_proj(query).view(B, G_q, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, G_k, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, G_k, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        coords_q = coords[:, :G_q, :]
        coords_k = coords[:, :G_k, :]
        delta = coords_q[:, :, None, :] - coords_k[:, None, :, :]  # (B, G_q, G_k, 2)
        delta_x, delta_y = delta[..., 0], delta[..., 1]

        bucket_x = relative_position_bucket(delta_x, self.num_buckets)  # (B, G_q, G_k)
        bucket_y = relative_position_bucket(delta_y, self.num_buckets)

        bias_x = self.rpe_bias_x(bucket_x)  # (B, G_q, G_k, heads)
        bias_y = self.rpe_bias_y(bucket_y)  # (B, G_q, G_k, heads)
        rpe_bias = bias_x + bias_y  # (B, G_q, G_k, heads)

        rpe_bias = rpe_bias.permute(0, 3, 1, 2)  # (B, heads, G_q, G_k)
        scores = scores + rpe_bias

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, heads, G_q, d_head)
        out = out.transpose(1, 2).contiguous().view(B, G_q, D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout after activation
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
        # PreNorm + Residual + Dropout (Attention)
        x_ = self.norm1(x)
        attn_out = self.self_attn(x_, x_, x_, coords, mask)
        x = x + self.dropout(attn_out)

        # PreNorm + Residual + Dropout (FFN)
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

class ScreenCorrespondence(nn.Module):
    def __init__(self, device, embed_dim=256, num_heads=4, num_layers=4, d_ff=256*4, dropout=0.3):
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim

        self.class_embed = nn.Embedding(28, embed_dim, padding_idx=27)
        self.vision_embed = nn.Linear(512*7*7, embed_dim)
        self.text_embed = nn.Linear(768, embed_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, d_ff)

        self.loss_fn = nn.MSELoss()

    def mask_predict(self, coords, class_idx, visions, texts, padding_mask):    
        E_class = self.class_embed(class_idx)
        class_mask_indices = self.mask_gui(padding_mask)
        class_tokens = torch.where(class_mask_indices.unsqueeze(-1), self.mask_token, E_class)
        
        E_vision = self.vision_embed(visions)
        vision_mask_indices = self.mask_gui(padding_mask)
        vision_tokens = torch.where(vision_mask_indices.unsqueeze(-1), self.mask_token, E_vision)
        
        E_text = self.text_embed(texts)
        no_text = (texts == 0).all(dim=-1)
        E_text = E_text * (~no_text).unsqueeze(-1)
        
        text_mask = padding_mask & (~no_text)
        text_mask_indices = self.mask_gui(text_mask)
        text_tokens = torch.where(text_mask_indices.unsqueeze(-1), self.mask_token, E_text)

        tokens = torch.cat([class_tokens, vision_tokens, text_tokens], dim=1)
        
        center_coords = torch.zeros(coords[:, :, :2].shape).to(self.device)
        center_coords[:, :, 0] = (coords[:, :, 0] + coords[:, :, 2])/2
        center_coords[:, :, 1] = (coords[:, :, 1] + coords[:, :, 3])/2
        center_coords = torch.cat([center_coords, center_coords, center_coords], dim=1)

        padding_mask = torch.cat([padding_mask, padding_mask, text_mask], dim=1)
        
        enc = self.transformer_encoder(tokens, center_coords, padding_mask) # shape: (B, 3G, D)

        G = class_idx.size(1)

        enc_class = enc[:, :G]
        enc_vision = enc[:, G:2*G]
        enc_text = enc[:, 2*G:]

        pred_class = enc_class[class_mask_indices]
        pred_vision = enc_vision[vision_mask_indices]
        pred_text = enc_text[text_mask_indices]
        pred = torch.cat([pred_class, pred_vision, pred_text], dim=0)
        
        true_class = E_class[class_mask_indices]
        true_vision = E_vision[vision_mask_indices]
        true_text = E_text[text_mask_indices]
        true = torch.cat([true_class, true_vision, true_text], dim=0)

        loss = self.loss_fn(pred, true)

        return loss

    def encode(self, coords, class_idx, visions, texts, padding_mask):    
        E_class = self.class_embed(class_idx)
        
        E_vision = self.vision_embed(visions)
        
        E_text = self.text_embed(texts)
        no_text = (texts == 0).all(dim=-1)
        E_text = E_text * (~no_text).unsqueeze(-1)

        tokens = torch.cat([E_class, E_vision, E_text], dim=1)
        
        center_coords = torch.zeros(coords[:, :, :2].shape).to(self.device)
        center_coords[:, :, 0] = (coords[:, :, 0] + coords[:, :, 2])/2
        center_coords[:, :, 1] = (coords[:, :, 1] + coords[:, :, 3])/2
        center_coords = torch.cat([center_coords, center_coords, center_coords], dim=1)

        text_mask = padding_mask & (~no_text)
        padding_mask = torch.cat([padding_mask, padding_mask, text_mask], dim=1)
        
        enc = self.transformer_encoder(tokens, center_coords, padding_mask) # shape: (B, 3G, D)

        G = class_idx.size(1)

        enc_class = enc[:, :G]
        enc_vision = enc[:, G:2*G]
        enc_text = enc[:, 2*G:]
        enc_text = enc_text * (~no_text).unsqueeze(-1)

        pooled = enc_class + enc_vision + enc_text
        pooled[no_text] /= 2
        pooled[~no_text] /= 3

        return pooled

    def sim_cosine(
        self, 
        coords1, class_idx1, visions1, texts1, padding_mask1,
        coords2, class_idx2, visions2, texts2, padding_mask2
    ):
        screen_emb1 = self.encode(coords1, class_idx1, visions1, texts1, padding_mask1)
        mask1 = padding_mask1.unsqueeze(-1)  # (B, G, 1)
        masked_emb1 = screen_emb1 * mask1  # 패딩 위치는 0이 됨
        sum_emb1 = masked_emb1.sum(dim=1)  # (B, D)
        valid_token_counts1 = mask1.sum(dim=1)  # (B, 1)
        mean_emb1 = sum_emb1 / valid_token_counts1.clamp(min=1)
        
        screen_emb2 = self.encode(coords2, class_idx2, visions2, texts2, padding_mask2)
        mask2 = padding_mask2.unsqueeze(-1)  # (B, G, 1)
        masked_emb2 = screen_emb2 * mask2  # 패딩 위치는 0이 됨
        sum_emb2 = masked_emb2.sum(dim=1)  # (B, D)
        valid_token_counts2 = mask2.sum(dim=1)  # (B, 1)
        mean_emb2 = sum_emb2 / valid_token_counts2.clamp(min=1)

        similarity = F.cosine_similarity(mean_emb1, mean_emb2, dim=1)

        return similarity

    def mask_gui(self, padding_mask, mask_ratio=0.15):
        B, G = padding_mask.shape
    
        rand = torch.rand(B, G, device=self.device)
        rand[~padding_mask] = 1.1  # 패딩은 선택 안 되게
    
        valid_counts = padding_mask.sum(dim=1)  # (B,)
        k = (valid_counts.float() * mask_ratio).ceil().long()  # (B,)
    
        # 정렬된 인덱스
        sorted_indices = torch.argsort(rand, dim=1)  # (B, G)
    
        # 마스킹용 인덱스 선택 마스크: (B, G)
        mask_selector = torch.arange(G, device=self.device).expand(B, G) < k.unsqueeze(1)  # (B, G)
    
        # 배치별 scatter
        mask_indices = torch.zeros(B, G, dtype=torch.bool, device=self.device)
        mask_indices.scatter_(1, sorted_indices, mask_selector)
    
        return mask_indices
