import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LayoutEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        # 인코더
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.GroupNorm(1, 8),  # LayerNorm 대체
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),  # → (8, 64, 128)

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(1, 16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),  # → (16, 32, 64)

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(1, 16),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),  # → (16, 16, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2),  # → (32, 8, 16)
        )

        self.fc = nn.Linear(4096, embed_dim)

    def forward(self, x):
        x = self.cnn(x)                # (B, 32, 8, 16)
        x = torch.flatten(x, 1)        # (B, 4096)
        z = self.fc(x)                 # (B, 384)
        return z

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_head = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        B, G_q, D = query.size()
        G_k = key.size(1)

        Q = self.q_proj(query).view(B, G_q, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, G_k, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, G_k, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

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
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.3):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, d_ff, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # PreNorm + Residual + Dropout (Attention)
        x_ = self.norm1(x)
        attn_out = self.self_attn(x_, x_, x_, mask)
        x = x + self.dropout(attn_out)

        # PreNorm + Residual + Dropout (FFN)
        x_ = self.norm2(x)
        ff_out = self.ffn(x_)
        x = x + self.dropout(ff_out)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, d_ff, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class PW2SS(nn.Module):
    def __init__(self, device, embed_dim=256, num_heads=4, num_layers=4, d_ff=256*4, dropout=0.3):
        super().__init__()

        self.device = device
        self.embed_dim = embed_dim
        self.width = 128
        self.height = 256

        self.x0_table = nn.Embedding(self.width+1, embed_dim)
        self.y0_table = nn.Embedding(self.height+1, embed_dim)
        self.x1_table = nn.Embedding(self.width+1, embed_dim)
        self.y1_table = nn.Embedding(self.height+1, embed_dim)
        self.w_table = nn.Embedding(self.width+1, embed_dim)
        self.h_table = nn.Embedding(self.height+1, embed_dim)
        
        self.class_embed = nn.Embedding(28, embed_dim, padding_idx=27)
        self.text_embed = nn.Linear(768, embed_dim)

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.layout_encoder = LayoutEncoder(embed_dim=embed_dim)
        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, d_ff)

        self.loss_fn = nn.MSELoss()

    def mask_predict(self, text_embed, text_coords, text_mask, graphic_class_idx, graphic_coords, graphic_mask, layout):
        coords = torch.cat([text_coords, graphic_coords], dim=1) # (B, T+G, 6)
        
        E_x0 = self.x0_table((coords[:, :, 0] * self.width).long())
        E_y0 = self.y0_table((coords[:, :, 1] * self.height).long())
        E_x1 = self.x1_table((coords[:, :, 2] * self.width).long())
        E_y1 = self.y1_table((coords[:, :, 3] * self.height).long())
        E_w = self.w_table((coords[:, :, 4] * self.width).long())
        E_h = self.h_table((coords[:, :, 5] * self.height).long())
        
        E_text = self.text_embed(text_embed)
        E_class = self.class_embed(graphic_class_idx)
        E_pw = torch.cat([E_text, E_class], dim=1)

        gui_tokens = E_pw + E_x0 + E_y0 + E_x1 + E_y1 + E_w + E_h
        
        layout_emb = self.layout_encoder(layout).unsqueeze(1)

        text_mask_indices = self.mask_gui(text_mask)
        graphic_mask_indices = self.mask_gui(graphic_mask)
        mask_indices = torch.cat([text_mask_indices, graphic_mask_indices], dim=1)
        gui_tokens = torch.where(mask_indices.unsqueeze(-1), self.mask_token, gui_tokens)

        batch_size = text_mask.size(0)
        tokens = torch.cat([layout_emb, gui_tokens], dim=1)
        true_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
        padding_mask = torch.cat([true_mask, text_mask, graphic_mask], dim=1)
        
        enc = self.transformer_encoder(tokens, padding_mask) # shape: (B, G, D)

        T = text_mask.size(1)

        enc_text = enc[:, 1:T+1]
        enc_graphic = enc[:, T+1:]

        masked_enc_text = enc_text[text_mask_indices]
        masked_enc_graphic = enc_graphic[graphic_mask_indices]
        masked_enc = torch.cat([masked_enc_text, masked_enc_graphic], dim=0)
        
        true_text = E_text[text_mask_indices]
        true_graphic = E_class[graphic_mask_indices]
        true_embed = torch.cat([true_text, true_graphic], dim=0)

        loss = self.loss_fn(masked_enc, true_embed)

        return loss

    def encode(self, text_embed, text_coords, text_mask, graphic_class_idx, graphic_coords, graphic_mask, layout):
        coords = torch.cat([text_coords, graphic_coords], dim=1) # (B, T+G, 6)
        
        E_x0 = self.x0_table((coords[:, :, 0] * self.width).long())
        E_y0 = self.y0_table((coords[:, :, 1] * self.height).long())
        E_x1 = self.x1_table((coords[:, :, 2] * self.width).long())
        E_y1 = self.y1_table((coords[:, :, 3] * self.height).long())
        E_w = self.w_table((coords[:, :, 4] * self.width).long())
        E_h = self.h_table((coords[:, :, 5] * self.height).long())
        
        E_text = self.text_embed(text_embed)
        E_class = self.class_embed(graphic_class_idx)
        E_pw = torch.cat([E_text, E_class], dim=1)

        gui_tokens = E_pw + E_x0 + E_y0 + E_x1 + E_y1 + E_w + E_h
        
        layout_emb = self.layout_encoder(layout).unsqueeze(1)

        batch_size = text_mask.size(0)
        tokens = torch.cat([layout_emb, gui_tokens], dim=1)
        true_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
        padding_mask = torch.cat([true_mask, text_mask, graphic_mask], dim=1)
        
        enc = self.transformer_encoder(tokens, padding_mask)

        mask = padding_mask.unsqueeze(-1)  # (B, T+G+1, 1)
        masked_enc = enc.masked_fill(~mask, float('-inf'))          # 패딩 위치는 0으로 만듦
        max_enc = masked_enc.max(dim=1).values
        
        return max_enc

    def sim_cosine(
        self, 
        text_embed1, text_coords1, text_mask1, graphic_class_idx1, graphic_coords1, graphic_mask1, layout1,
        text_embed2, text_coords2, text_mask2, graphic_class_idx2, graphic_coords2, graphic_mask2, layout2
    ):
        screen_emb1 = self.encode(text_embed1, text_coords1, text_mask1, graphic_class_idx1, graphic_coords1, graphic_mask1, layout1)
        screen_emb2 = self.encode(text_embed2, text_coords2, text_mask2, graphic_class_idx2, graphic_coords2, graphic_mask2, layout2)

        similarity = F.cosine_similarity(screen_emb1, screen_emb2, dim=1)

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



        