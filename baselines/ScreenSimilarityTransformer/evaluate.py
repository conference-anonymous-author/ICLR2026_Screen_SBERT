import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
from itertools import combinations
from sklearn.manifold import TSNE
import copy
import matplotlib.pyplot as plt

from ScreenSimilarityTransformer import SimilarityTransformer

def possible_same_pairs(class2idx):
    same_pairs = []
    for screens in class2idx.values():
        if len(screens) >= 2:
            pairs = list(combinations(screens, 2))
            same_pairs += pairs
    return same_pairs

def possible_different_pairs(class2idx):
    different_pairs = []
    classes = list(class2idx.keys())
    for class1, class2 in combinations(classes, 2):
        for i in class2idx[class1]:
            for j in class2idx[class2]:
                different_pairs.append((i, j))
    return different_pairs

def prepare_testset(app_list):
    pairs = []
    app_names = []
    
    for app in app_list:
        class2idx = class2idx_dict[app]
        
        same_pairs = possible_same_pairs(class2idx)
        different_pairs = possible_different_pairs(class2idx)
        pairs += same_pairs
        pairs += different_pairs

        app_names += [app]*len(same_pairs)
        app_names += [app]*len(different_pairs)

    return pairs, app_names

class ScreenPairDataset(Dataset):
    def __init__(self, pairs, app_names):
        self.pairs = pairs
        self.app_names = app_names

    def __len__(self):
        return len(self.app_names)

    def __getitem__(self, idx):
        return {
            "pair": self.pairs[idx],
            "app_name": self.app_names[idx]
        }

def DataLoader_fn(batch):
    pairs = []
    app_names = []

    for item in batch:
        pairs.append(item["pair"])
        app_names.append(item["app_name"])
    
    return {
        "pairs": pairs,
        "app_names": app_names
    }

class2idx_dict = {
    "Instagram": Instagram_class2idx,
    "Facebook": Facebook_class2idx,
    "X": X_class2idx,
    "Amazon": Amazon_class2idx,
    "Coupang": Coupang_class2idx,
    "Temu": Temu_class2idx
}
class2num_dict = {
    "Instagram": generate_class2num(Instagram_class2idx),
    "Facebook": generate_class2num(Facebook_class2idx),
    "X": generate_class2num(X_class2idx),
    "Amazon": generate_class2num(Amazon_class2idx),
    "Coupang": generate_class2num(Coupang_class2idx),
    "Temu": generate_class2num(Temu_class2idx)
}
idx2class_dict = {
    "Instagram": generate_idx2class(Instagram_class2idx),
    "Facebook": generate_idx2class(Facebook_class2idx),
    "X": generate_idx2class(X_class2idx),
    "Amazon": generate_idx2class(Amazon_class2idx),
    "Coupang": generate_idx2class(Coupang_class2idx),
    "Temu": generate_idx2class(Temu_class2idx)
}
app_list = ["Facebook", "Coupang"]

device = "cuda:0"
model = SimilarityTransformer(device=device).to(device)
set_name = f"{app_list[0]}_{app_list[1]}"
model.load_state_dict(torch.load(f"./weights/0818/other_apps/SST/{set_name}/bestmodel.pth", weights_only=True))
model.eval()

pairs, app_names = prepare_testset(app_list)
dataset = ScreenPairDataset(pairs, app_names)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    collate_fn=DataLoader_fn
)
len(dataloader)

similarity_dict = {}

for app in app_list:
    similarity_dict[app] = {}
    idx2class = idx2class_dict[app]
    for si in idx2class.keys():
        similarity_dict[app][si] = {
            "key_indices": [],
            "similarities": []
        }

with torch.no_grad():
    step = 0
    for batch in dataloader:           
        pairs = batch["pairs"]
        app_names = batch["app_names"]
            
        coords_A = []
        class_idx_A = []
        visions_A = []
            
        coords_B = []
        class_idx_B = []
        visions_B = []
            
        for bi in range(len(app_names)):
            a, b = pairs[bi]
            app = app_names[bi]
                    
            coords_A.append(torch.tensor(np.load(f"./dataset/{app}/{a}/gui_coords.npy")))
            class_idx_A.append(torch.tensor(np.load(f"./dataset/{app}/{a}/gui_class_idx.npy")))
            visions_A.append(torch.tensor(np.load(f"./dataset/{app}/{a}/gui_visions.npy")))
            
            coords_B.append(torch.tensor(np.load(f"./dataset/{app}/{b}/gui_coords.npy")))
            class_idx_B.append(torch.tensor(np.load(f"./dataset/{app}/{b}/gui_class_idx.npy")))    
            visions_B.append(torch.tensor(np.load(f"./dataset/{app}/{b}/gui_visions.npy")))
                
        padded_coords_A = pad_sequence(coords_A, batch_first=True).float().to(device)
        padded_class_idx_A = pad_sequence(class_idx_A, batch_first=True, padding_value=27).to(device)
        padded_visions_A = pad_sequence(visions_A, batch_first=True).float().to(device)
                
        padded_coords_B = pad_sequence(coords_B, batch_first=True).float().to(device)
        padded_class_idx_B = pad_sequence(class_idx_B, batch_first=True, padding_value=27).to(device)
        padded_visions_B = pad_sequence(visions_B, batch_first=True).float().to(device)
            
        padding_mask_A = torch.tensor([
            [1]*len(a) + [0]*(padded_coords_A.size(1) - len(a)) for a in class_idx_A
        ], dtype=torch.bool).to(device)
        padding_mask_B = torch.tensor([
            [1]*len(b) + [0]*(padded_coords_B.size(1) - len(b)) for b in class_idx_B
        ], dtype=torch.bool).to(device)
            
        logits, _, _, _ = model(
            coords1=padded_coords_A, 
            class_idx1=padded_class_idx_A, 
            visions1=padded_visions_A,
            padding_mask1=padding_mask_A, 
                    
            coords2=padded_coords_B, 
            class_idx2=padded_class_idx_B,
            visions2=padded_visions_B,
            padding_mask2=padding_mask_B,
                    
            reconstruct_flag=False
        )
        probs = torch.sigmoid(logits)

        for bi in range(len(app_names)):
            a, b = pairs[bi]
            app = app_names[bi]

            similarity_dict[app][a]["key_indices"].append(b)
            similarity_dict[app][a]["similarities"].append(probs[bi].item())
            similarity_dict[app][b]["key_indices"].append(a)
            similarity_dict[app][b]["similarities"].append(probs[bi].item())

        print(f"{step}/{len(dataloader)} Complete")
        step += 1

np.save(f"./results/0818/other_apps/SST/{set_name}.npy", similarity_dict, allow_pickle=True)

set_name = f"{app_list[0]}_{app_list[1]}"
similarity_dict = np.load(f"./results/0818/other_apps/SST/{set_name}.npy", allow_pickle=True).item()

def weighted_topk_acc(truth, pred):
    corrects = {}
    counts = {}
    for i in range(len(truth)):
        if not truth[i] in corrects.keys():
            corrects[truth[i]] = 0
            counts[truth[i]] = 1
        else:
            counts[truth[i]] += 1
        if truth[i] in pred[i]:
            corrects[truth[i]] += 1

    corrects = np.array(list(corrects.values()))
    counts = np.array(list(counts.values()))
    acc = corrects / counts

    return np.mean(acc).item()

truth = []
top1 = []
top2 = []
top3 = []

for app in app_list:
    result = similarity_dict[app]
    for si in result.keys():
        true_page = idx2class_dict[app][si]
        if len(class2idx_dict[app][true_page]) < 2:
            continue
        
        pred_idx = np.argsort(result[si]["similarities"])[-3:]
        top3_sorted = np.array(result[si]["key_indices"])[pred_idx][::-1]
        
        truth.append(class2num_dict[app][true_page])
        top1.append([class2num_dict[app][idx2class_dict[app][top3_sorted[0]]]])
        top2.append([
            class2num_dict[app][idx2class_dict[app][top3_sorted[0]]],
            class2num_dict[app][idx2class_dict[app][top3_sorted[1]]]
        ])
        top3.append([
            class2num_dict[app][idx2class_dict[app][top3_sorted[0]]],
            class2num_dict[app][idx2class_dict[app][top3_sorted[1]]],
            class2num_dict[app][idx2class_dict[app][top3_sorted[2]]]
        ])

recall = recall_score(truth, top1, average='macro')
precision = precision_score(truth, top1, average='macro')
macro_f1 = f1_score(truth, top1, average='macro')
print(f"P: {precision:.3f}, R: {recall:.3f}, F1: {macro_f1:.3f}")

print(f"Top1: {weighted_topk_acc(truth, top1):.3f}")
print(f"Top2: {weighted_topk_acc(truth, top2):.3f}")
print(f"Top3: {weighted_topk_acc(truth, top3):.3f}")