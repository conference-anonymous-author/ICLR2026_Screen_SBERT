import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class GUIEmbedding(nn.Module):
    def __init__(self, vision_dim=512*7*7, embed_dim=64, num_types=28):
        super().__init__()

        self.coord_proj = nn.Linear(6, embed_dim)
        self.type_table = nn.Embedding(num_types, embed_dim, padding_idx=27)
        self.vision_proj = nn.Linear(vision_dim, embed_dim)
        self.screen_table = nn.Embedding(2, embed_dim)

    def forward(self, coords, types, visions, screen_idx):
        E_coord = self.coord_proj(coords)
        E_type = self.type_table(types)
        E_vision = self.vision_proj(visions)

        screen_idx_tensor = torch.full(types.shape, screen_idx, dtype=torch.long, device=coords.device)
        E_screen = self.screen_table(screen_idx_tensor)

        combined = torch.cat([E_coord, E_type, E_vision, E_screen], dim=2)
        
        return combined

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
    def __init__(self, embed_dim, num_heads, d_ff, dropout=0.3):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, d_ff, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x_ = self.norm1(x)
        attn_out = self.self_attn(x_, x_, x_, mask)
        x = x + self.dropout(attn_out)

        x_ = self.norm2(x)
        ff_out = self.ffn(x_)
        x = x + self.dropout(ff_out)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, d_ff):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

class SimilarityTransformer(nn.Module):
    def __init__(self, device, vision_dim=512*7*7, embed_dim=256, num_types=28, num_heads=8, num_layers=6, d_ff=1024, dropout=0.3):
        super().__init__()

        self.device = device

        self.gui_embedding = GUIEmbedding(vision_dim, 64, num_types)

        self.sim_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = torch.zeros((1, 1, embed_dim)).to(device)

        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, d_ff)

        self.similarity_predictor = nn.Linear(embed_dim, 1)
        self.coord_reconstructor = nn.Linear(embed_dim, 6)
        self.type_reconstructor = nn.Linear(embed_dim, num_types)
        self.vision_reconstructor = nn.Linear(embed_dim, vision_dim)

        self.coord_loss_fn = nn.MSELoss()
        self.type_loss_fn = nn.CrossEntropyLoss()
        self.vision_loss_fn = nn.MSELoss()

    def forward(
        self, 
        coords1, types1, visions1, padding_mask1, 
        coords2, types2, visions2, padding_mask2, 
        reconstruct_flag=True
    ):
        tokens1 = self.gui_embedding(coords1, types1, visions1, screen_idx=0) 
        tokens2 = self.gui_embedding(coords2, types2, visions2, screen_idx=1)  

        if reconstruct_flag:
            mask_indices1 = self.mask_gui(padding_mask1)
            tokens1 = torch.where(mask_indices1.unsqueeze(-1), self.mask_token, tokens1)
            mask_indices2 = self.mask_gui(padding_mask2)
            tokens2 = torch.where(mask_indices2.unsqueeze(-1), self.mask_token, tokens2)

        batch_size = types1.size(0)
        sim_tokens = self.sim_token.repeat(batch_size, 1, 1)
        tokens = torch.cat([sim_tokens, tokens1, tokens2], dim=1)

        true_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=self.device)
        padding_mask = torch.cat([true_mask, padding_mask1, padding_mask2], dim=1)
        
        enc = self.encoder(tokens, padding_mask)

        pred_sim = self.similarity_predictor(enc[:, 0]).squeeze(-1)

        if not reconstruct_flag:
            return pred_sim, None, None, None
        else:
            masked_coords1 = coords1[mask_indices1]
            masked_types1 = types1[mask_indices1]
            masked_visions1 = visions1[mask_indices1]
            masked_encodings1 = enc[:, 1:types1.size(1)+1][mask_indices1]
    
            masked_coords2 = coords2[mask_indices2]
            masked_types2 = types2[mask_indices2]
            masked_visions2 = visions2[mask_indices2]
            masked_encodings2 = enc[:, types1.size(1)+1:][mask_indices2]
    
            masked_coords = torch.cat([masked_coords1, masked_coords2], dim=0)
            masked_types = torch.cat([masked_types1, masked_types2], dim=0)
            masked_visions = torch.cat([masked_visions1, masked_visions2], dim=0)
            masked_encodings = torch.cat([masked_encodings1, masked_encodings2], dim=0)
    
            pred_coords = self.coord_reconstructor(masked_encodings)
            coord_loss = self.coord_loss_fn(pred_coords, masked_coords)
    
            types = self.type_reconstructor(masked_encodings)
            type_loss = self.type_loss_fn(types, masked_types)

            pred_visions = self.vision_reconstructor(masked_encodings)
            vision_loss = self.vision_loss_fn(pred_visions, masked_visions)
    
            return pred_sim, coord_loss, type_loss, vision_loss

    def mask_gui(self, padding_mask, mask_ratio=0.15):
        B, G = padding_mask.shape
    
        rand = torch.rand(B, G, device=self.device)
        rand[~padding_mask] = 1.1
    
        valid_counts = padding_mask.sum(dim=1)
        k = (valid_counts.float() * mask_ratio).ceil().long()
    
        sorted_indices = torch.argsort(rand, dim=1)
    
        mask_selector = torch.arange(G, device=self.device).expand(B, G) < k.unsqueeze(1)
    
        mask_indices = torch.zeros(B, G, dtype=torch.bool, device=self.device)
        mask_indices.scatter_(1, sorted_indices, mask_selector)
    
        return mask_indices
