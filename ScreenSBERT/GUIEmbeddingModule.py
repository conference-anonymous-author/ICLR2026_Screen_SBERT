import torch.nn as nn

class GUIEmbeddingModule(nn.Module):
    def __init__(self, vision_dim=512*7*7, text_dim=768, embed_dim=768, num_class=28, width=128, height=256):
        super().__init__()

        self.width = width
        self.height = height

        self.x0_table = nn.Embedding(width+1, embed_dim)
        self.y0_table = nn.Embedding(height+1, embed_dim)
        self.x1_table = nn.Embedding(width+1, embed_dim)
        self.y1_table = nn.Embedding(height+1, embed_dim)
        self.w_table = nn.Embedding(width+1, embed_dim)
        self.h_table = nn.Embedding(height+1, embed_dim)
        
        self.type_table = nn.Embedding(num_class, embed_dim, padding_idx=27)

        self.vision_proj = nn.Linear(vision_dim, embed_dim)

        self.text_proj = nn.Linear(text_dim, embed_dim)

    def forward(self, coords, types, visions, texts):        
        E_x0 = self.x0_table((coords[:, :, 0] * self.width).long())
        E_y0 = self.y0_table((coords[:, :, 1] * self.height).long())
        E_x1 = self.x1_table((coords[:, :, 2] * self.width).long())
        E_y1 = self.y1_table((coords[:, :, 3] * self.height).long())
        E_w = self.w_table((coords[:, :, 4] * self.width).long())
        E_h = self.h_table((coords[:, :, 5] * self.height).long())
        
        E_type = self.type_table(types)

        E_vision = self.vision_proj(visions)

        no_text = (texts == 0).all(dim=-1)
        E_text = self.text_proj(texts)
        E_text[no_text] = 0

        gui_tokens = E_type + E_text + E_vision + E_x0 + E_y0 + E_x1 + E_y1 + E_w + E_h
        
        return gui_tokens