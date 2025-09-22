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
from PW2SS import PW2SS

device = "cuda:0"
model = PW2SS(device=device).to(device)
app_list = ["X", "Temu"]
set_name = f"{app_list[0]}_{app_list[1]}"
model.load_state_dict(torch.load(f"./weights/other_apps/PW2SS/{set_name}/bestmodel.pth", weights_only=True))
model.eval()

test_embeddings = {}
for app in app_list:
    test_embeddings[app] = {}

transforms_fn = transforms.ToTensor()

with torch.no_grad():
    for app in app_list:
        class_idx = class2idx_dict[app]
        for screen_indices in class_idx.values():
            for si in screen_indices:         
                text_embed = torch.tensor(np.load(f"./dataset/{app}/{si}/screen_ocr_embed.npy")).float().unsqueeze(0).to(device)
                text_coords = torch.tensor(np.load(f"./dataset/{app}/{si}/screen_ocr_coords.npy")).float().unsqueeze(0).to(device)
                graphic_class_idx = torch.tensor(np.load(f"./dataset/{app}/{si}/gui_class_idx.npy")).long().unsqueeze(0).to(device)
                graphic_coords = torch.tensor(np.load(f"./dataset/{app}/{si}/gui_coords.npy")).float().unsqueeze(0).to(device)
                layout = transforms_fn(np.load(f"./dataset/{app}/{si}/layout.npy")).float().unsqueeze(0).to(device)

                text_mask = torch.ones(1, text_coords.size(1), dtype=torch.bool, device=device)
                graphic_mask = torch.ones(1, graphic_coords.size(1), dtype=torch.bool, device=device)
    
                enc = model.encode(
                    text_embed=text_embed, 
                    text_coords=text_coords, 
                    text_mask=text_mask, 
                    
                    graphic_class_idx=graphic_class_idx, 
                    graphic_coords=graphic_coords, 
                    graphic_mask=graphic_mask, 
                    
                    layout=layout
                ).squeeze(0)

                test_embeddings[app][si] = enc.detach().cpu().numpy()
                print(f"{si} of {app} completes")