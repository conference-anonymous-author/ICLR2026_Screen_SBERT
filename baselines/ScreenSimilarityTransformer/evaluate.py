import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score
from itertools import combinations
from tqdm import tqdm

from SimilarityTransformer import SimilarityTransformer
from dataset.page_labels import page2screen_indices_dict, page_indices_dict, screen_idx2page_dict

def possible_same_pairs(page2screen_indices):
    same_pairs = []
    for screen_indices in page2screen_indices.values():
        if len(screen_indices) >= 2:
            pairs = list(combinations(screen_indices, 2))
            same_pairs += pairs
    return same_pairs

def possible_different_pairs(page2screen_indices):
    different_pairs = []
    classes = list(page2screen_indices.keys())
    for class1, class2 in combinations(classes, 2):
        for i in page2screen_indices[class1]:
            for j in page2screen_indices[class2]:
                different_pairs.append((i, j))
    return different_pairs

def load_test_data(app_list):
    pairs = []
    app_names = []
    
    for app in app_list:
        page2screen_indices = page2screen_indices_dict[app]
        
        same_pairs = possible_same_pairs(page2screen_indices)
        different_pairs = possible_different_pairs(page2screen_indices)
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

def calculating_pair_similarity(model, evaluation_apps):
    pairs, app_names = load_test_data(evaluation_apps)
    dataset = ScreenPairDataset(pairs, app_names)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=DataLoader_fn
    )
    len(dataloader)

    similarity_dict = {}

    for app in evaluation_apps:
        similarity_dict[app] = {}
        idx2class = screen_idx2page_dict[app]
        for si in idx2class.keys():
            similarity_dict[app][si] = {
                "key_indices": [],
                "similarities": []
            }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Pair Similarities"):           
            pairs = batch["pairs"]
            app_names = batch["app_names"]
                
            coords_A = []
            types_A = []
            visions_A = []
                
            coords_B = []
            types_B = []
            visions_B = []
                
            for bi in range(len(app_names)):
                a, b = pairs[bi]
                app = app_names[bi]
                        
                coords_A.append(torch.tensor(np.load(f"../../dataset/{app}/{a}/gui_coords.npy")))
                types_A.append(torch.tensor(np.load(f"../../dataset/{app}/{a}/gui_functional_types.npy")))
                visions_A.append(torch.tensor(np.load(f"../../dataset/{app}/{a}/gui_vision_feature_maps.npy")))
        
                coords_B.append(torch.tensor(np.load(f"../../dataset/{app}/{b}/gui_coords.npy")))
                types_B.append(torch.tensor(np.load(f"../../dataset/{app}/{b}/gui_functional_types.npy")))    
                visions_B.append(torch.tensor(np.load(f"../../dataset/{app}/{b}/gui_vision_feature_maps.npy")))
                    
            padded_coords_A = pad_sequence(coords_A, batch_first=True).float().to(device)
            padded_types_A = pad_sequence(types_A, batch_first=True, padding_value=27).to(device)
            padded_visions_A = pad_sequence(visions_A, batch_first=True).float().to(device)
                    
            padded_coords_B = pad_sequence(coords_B, batch_first=True).float().to(device)
            padded_types_B = pad_sequence(types_B, batch_first=True, padding_value=27).to(device)
            padded_visions_B = pad_sequence(visions_B, batch_first=True).float().to(device)
                
            padding_mask_A = torch.tensor([
                [1]*len(a) + [0]*(padded_coords_A.size(1) - len(a)) for a in types_A
            ], dtype=torch.bool).to(device)
            padding_mask_B = torch.tensor([
                [1]*len(b) + [0]*(padded_coords_B.size(1) - len(b)) for b in types_B
            ], dtype=torch.bool).to(device)
                
            logits, _, _, _ = model(
                coords1=padded_coords_A, 
                types1=padded_types_A, 
                visions1=padded_visions_A,
                padding_mask1=padding_mask_A, 
                        
                coords2=padded_coords_B, 
                types2=padded_types_B,
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

def topk_accuracy(truth, pred):
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

if __name__ == "__main__":
    evaluation_apps = ["X", "Temu"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimilarityTransformer(device=device).to(device)
    model.load_state_dict(torch.load(f"./weights/evaluate_{evaluation_apps[0]}_{evaluation_apps[1]}.pth", weights_only=True))
    model.eval()

    similarity_file = f"./similarities/evaluate_{evaluation_apps[0]}_{evaluation_apps[1]}.npy"
    if os.path.exists(similarity_file):
        similarity_dict = np.load(similarity_file, allow_pickle=True).item()
    else:
        similarity_dict = calculating_pair_similarity(model, evaluation_apps)
        np.save(similarity_file, similarity_dict, allow_pickle=True)

    truth = []
    top1 = []
    top2 = []
    top3 = []

    for app in evaluation_apps:
        similarity = similarity_dict[app]
        for screen_idx in tqdm(similarity.keys(), desc=f"Evaluating {app}"):
            true_page = screen_idx2page_dict[app][screen_idx]
            if len(page2screen_indices_dict[app][true_page]) < 2:
                continue
            
            pred_idx = np.argsort(similarity[screen_idx]["similarities"])[-3:]
            top3_sorted = np.array(similarity[screen_idx]["key_indices"])[pred_idx][::-1]
            
            truth.append(page_indices_dict[app][true_page])
            top1.append([page_indices_dict[app][screen_idx2page_dict[app][top3_sorted[0]]]])
            top2.append([
                page_indices_dict[app][screen_idx2page_dict[app][top3_sorted[0]]],
                page_indices_dict[app][screen_idx2page_dict[app][top3_sorted[1]]]
            ])
            top3.append([
                page_indices_dict[app][screen_idx2page_dict[app][top3_sorted[0]]],
                page_indices_dict[app][screen_idx2page_dict[app][top3_sorted[1]]],
                page_indices_dict[app][screen_idx2page_dict[app][top3_sorted[2]]]
            ])

    recall = recall_score(truth, top1, average='macro')
    precision = precision_score(truth, top1, average='macro')
    macro_f1 = f1_score(truth, top1, average='macro')
    print(f"P: {precision:.3f}, R: {recall:.3f}, F1: {macro_f1:.3f}")

    print(f"Top1: {topk_accuracy(truth, top1):.3f}")
    print(f"Top2: {topk_accuracy(truth, top2):.3f}")
    print(f"Top3: {topk_accuracy(truth, top3):.3f}")