import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from utils import possible_same_pairs, possible_different_pairs, print_gradients, get_different_pairs, valid_macro_f1
from screen_class import X_class2idx, Temu_class2idx, Instagram_class2idx, Facebook_class2idx, Coupang_class2idx, Amazon_class2idx
from ScreenCorrespondence import ScreenCorrespondence

class2idx_dict = {
    "Instagram": Instagram_class2idx,
    "Facebook": Facebook_class2idx,
    "X": X_class2idx,
    "Amazon": Amazon_class2idx,
    "Coupang": Coupang_class2idx,
    "Temu": Temu_class2idx
}

device = "cuda:0"

app_list = ["X", "Coupang"]
set_name = f"{app_list[0]}_{app_list[1]}"

model = ScreenCorrespondence(device=device).to(device)
model.load_state_dict(torch.load(f"./weights/other_apps/SC/{set_name}/bestmodel.pth", weights_only=True))
model.eval()

test_embeddings = {}
for app in app_list:
    test_embeddings[app] = {}

transforms_fn = transforms.ToTensor()

with torch.no_grad():
    for app in app_list:
        class_idx = class2idx_dict[app]
        for screen_indices in class_idx.values():
            for screen_idx in screen_indices:                
                coords = torch.tensor(np.load(f"./dataset/{app}/{screen_idx}/gui_coords.npy")).float().unsqueeze(0).to(device)
                class_idx = torch.tensor(np.load(f"./dataset/{app}/{screen_idx}/gui_class_idx.npy")).unsqueeze(0).to(device)
                visions = torch.tensor(np.load(f"./dataset/{app}/{screen_idx}/gui_visions.npy")).float().unsqueeze(0).to(device)
                texts = torch.tensor(np.load(f"./dataset/{app}/{screen_idx}/gui_text_embed.npy")).float().unsqueeze(0).to(device)
                padding_mask = torch.ones(1, class_idx.size(1), dtype=torch.bool, device=device)
    
                screen_emb = model.encode(
                    coords=coords, 
                    class_idx=class_idx,
                    visions=visions,
                    texts=texts,
                    padding_mask=padding_mask,    
                )

                mask = padding_mask.unsqueeze(-1)  # (B, G, 1)
                masked_emb = screen_emb * mask  # 패딩 위치는 0이 됨
                sum_emb = masked_emb.sum(dim=1)  # (B, D)
                valid_token_counts = mask.sum(dim=1)  # (B, 1)
                mean_emb = sum_emb / valid_token_counts.clamp(min=1)
                mean_emb = mean_emb.squeeze(0)

                test_embeddings[app][screen_idx] = mean_emb.detach().cpu().numpy()
                print(f"{screen_idx} of {app} completes")