import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
import copy
from collections import defaultdict
from transformers import get_scheduler

from Transformer_2D_RPE import ScreenSBERT
from dataset.page_labels import page2screen_indices_dict, page_indices_dict

def load_train_data(app_list, train_valid_split):
    total_screen_indices = []
    page_classes = []
    app_names = []
    
    for app in app_list:
        page2screen_indices = train_valid_split[f"{app}_train"]
        for page_class, screen_indices in page2screen_indices.items():
            total_screen_indices += screen_indices
            page_classes += ([page_class]*len(screen_indices))
            app_names += ([app]*len(screen_indices))

    return total_screen_indices, page_classes, app_names

def load_valid_data(app_list, train_valid_split):
    valid_dataset = {}
    
    for app in app_list:
        valid_dataset[app] = {
            "screen_indices": [],
            "page_classes": []
        }
        page2screen_indices = train_valid_split[f"{app}_val"]
        for page_class, screen_indices in page2screen_indices.items():
            valid_dataset[app]["screen_indices"] += screen_indices
            valid_dataset[app]["page_classes"] += ([page_indices_dict[app][page_class]]*len(screen_indices))

    return valid_dataset
    
class ScreenDataset(Dataset):
    def __init__(self, screen_indices, page_classes, app_names):
        self.screen_indices = screen_indices
        self.page_classes = page_classes
        self.app_names = app_names

    def __len__(self):
        return len(self.app_names)

    def __getitem__(self, idx):   
        return {
            "screen_idx": self.screen_indices[idx],
            "page_class": self.page_classes[idx],
            "app_name": self.app_names[idx]
        }

def DataLoader_fn(batch):
    screen_indices = []
    page_classes = []
    app_names = []

    for item in batch:
        screen_indices.append(item["screen_idx"])
        page_classes.append(item["page_class"])
        app_names.append(item["app_name"])
    
    return {
        "screen_indices": screen_indices,
        "page_classes": page_classes,
        "app_names": app_names
    }

def make_contrastive_batch(anchor):
    anchor_screen_idx = anchor["screen_indices"][0]
    anchor_page_label = anchor["page_classes"][0]
    app_name = anchor["app_names"][0]

    page2screen_indices = page2screen_indices_dict[app_name]
    if len(page2screen_indices[anchor_page_label]) < 2:
        negative_only = True
    else:
        negative_only = False

    negative_pairs = []
    for page_class, screen_indices in page2screen_indices.items():
        if page_class == anchor_page_label:
            if negative_only:
                positive_pair = None
            else:
                positive_page = copy.deepcopy(screen_indices)
                positive_page.remove(anchor_screen_idx)
                positive = random.choice(positive_page)
                positive_pair = (anchor_screen_idx, positive)
        else:
            negative = random.choice(screen_indices)
            negative_pairs.append((anchor_screen_idx, negative))

    return positive_pair, negative_pairs, negative_only, app_name

# measuring how well screens corresponding to the same page cluster together in the embedding space
def validation_score(embeddings, page_classes):
    page_classes_tensor = torch.tensor(page_classes, device=embeddings.device)
    page_classes_unique = page_classes_tensor.unique()
    page2screen_indices = defaultdict(list)

    for screen_idx, page_class in enumerate(page_classes):
        page2screen_indices[page_class].append(screen_idx)

    cluster_centers = {}
    for page_class in page_classes_unique:
        screen_indices = page2screen_indices[page_class.item()]
        page_cluster = embeddings[screen_indices]
        cluster_center = page_cluster.mean(dim=0)
        cluster_centers[page_class.item()] = cluster_center

    # Intra-class distance: the distance of each screen from the center of its assigned page cluster
    intra_distance = []
    for page_class in page_classes_unique:
        screen_indices = page2screen_indices[page_class.item()]
        page_cluster = embeddings[screen_indices]
        mean = cluster_centers[page_class.item()]
        dists = 1 - F.cosine_similarity(page_cluster, mean.unsqueeze(0), dim=1)
        intra_distance.append(dists.max())

    intra_distance = torch.stack(intra_distance).mean().item()

    # Inter-class distance: the distance of each screen to the center of the nearest other page cluster
    inter_distance = 0
    for i, page_class in enumerate(page_classes):
        emb = embeddings[i]
        min_distance = float('inf')
        for other_page in page_classes_unique:
            if other_page == page_class:
                continue
            mean = cluster_centers[other_page.item()]
            distance = 1 - F.cosine_similarity(emb.unsqueeze(0), mean.unsqueeze(0)).item()
            if distance < min_distance:
                min_distance = distance
        inter_distance += min_distance

    inter_distance = inter_distance / len(page_classes)

    score = inter_distance - intra_distance

    return intra_distance, inter_distance, score

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScreenSBERT(device=device).to(device)

    train_valid_split = np.load("../dataset/train_valid_split.npy", allow_pickle=True).item()

    train_apps = ["X", "Instagram", "Coupang", "Temu"]

    best_valid_score = 0

    screen_indices_train, page_classes_train, app_names_train = load_train_data(train_apps, train_valid_split)
    dataset_train = ScreenDataset(screen_indices_train, page_classes_train, app_names_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        collate_fn=DataLoader_fn
    )

    valid_dataset = load_valid_data(train_apps, train_valid_split)

    max_epoch = 5
    total_steps = len(dataloader_train) * max_epoch
    warmup_steps = int(0.1*total_steps)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    temperature = 0.07
    eps = 1e-6

    num_types = 28

    step = 0
    model.train()
    optimizer.zero_grad()
    for epoch in range(max_epoch):
        for anchor in dataloader_train:         
            positive_pair, negative_pairs, negative_only, app = make_contrastive_batch(anchor)

            if negative_only:
                pairs = negative_pairs
            else:
                pairs = [positive_pair] + negative_pairs
            
            coords_A = []
            types_A = []
            visions_A = []
            texts_A = []
            
            coords_B = []
            types_B = []
            visions_B = []
            texts_B = []
        
            for batch_idx in range(len(pairs)):
                s1, s2 = pairs[batch_idx]

                coords_A.append(torch.tensor(np.load(f"../dataset/{app}/{s1}/gui_coords.npy")))
                types_A.append(torch.tensor(np.load(f"../dataset/{app}/{s1}/gui_functional_types.npy")))
                visions_A.append(torch.tensor(np.load(f"../dataset/{app}/{s1}/gui_vision_feature_maps.npy")))
                texts_A.append(torch.tensor(np.load(f"../dataset/{app}/{s1}/gui_text_embeds.npy")))
        
                coords_B.append(torch.tensor(np.load(f"../dataset/{app}/{s2}/gui_coords.npy")))
                types_B.append(torch.tensor(np.load(f"../dataset/{app}/{s2}/gui_functional_types.npy")))
                visions_B.append(torch.tensor(np.load(f"../dataset/{app}/{s2}/gui_vision_feature_maps.npy")))
                texts_B.append(torch.tensor(np.load(f"../dataset/{app}/{s2}/gui_text_embeds.npy")))
                
            padded_coords_A = pad_sequence(coords_A, batch_first=True).float().to(device)
            padded_types_A = pad_sequence(types_A, batch_first=True, padding_value=num_types-1).to(device)
            padded_visions_A = pad_sequence(visions_A, batch_first=True).float().to(device)
            padded_texts_A = pad_sequence(texts_A, batch_first=True).float().to(device)
                
            padded_coords_B = pad_sequence(coords_B, batch_first=True).float().to(device)
            padded_types_B = pad_sequence(types_B, batch_first=True, padding_value=num_types-1).to(device)
            padded_visions_B = pad_sequence(visions_B, batch_first=True).float().to(device)
            padded_texts_B = pad_sequence(texts_B, batch_first=True).float().to(device)
        
            padding_mask_A = torch.tensor([
                [1]*len(a) + [0]*(padded_coords_A.size(1) - len(a)) for a in types_A
            ], dtype=torch.bool).to(device)
            padding_mask_B = torch.tensor([
                [1]*len(b) + [0]*(padded_coords_B.size(1) - len(b)) for b in types_B
            ], dtype=torch.bool).to(device)

            cosine_similarities = model.cosine_similarity(
                coords1=padded_coords_A, 
                types1=padded_types_A,
                visions1=padded_visions_A,
                texts1=padded_texts_A,
                padding_mask1=padding_mask_A, 
                    
                coords2=padded_coords_B, 
                types2=padded_types_B, 
                visions2=padded_visions_B,
                texts2=padded_texts_B,
                padding_mask2=padding_mask_B,
            )
            
            if negative_only:
                contrastive_loss = -torch.log(1 - cosine_similarities.clamp(max=1 - eps))
                contrastive_loss = contrastive_loss.mean()
                contrastive_loss = torch.clamp(contrastive_loss, min=0.0)
            else:
                probs = torch.softmax(cosine_similarities / temperature, dim=0)
                contrastive_loss = -torch.log(probs[0] + eps)

            contrastive_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            print(f"Train {step} >> Loss: {contrastive_loss.item():.4f}, Negative Only: {negative_only}")

            step += 1
            if step % 10 == 0: # Validation
                model.eval()
        
                total_inter_distance = 0
                total_intra_distance = 0
                total_score = 0
                
                with torch.no_grad():
                    for app in train_apps:
                        screen_indices = valid_dataset[app]["screen_indices"]
                        page_classes = valid_dataset[app]["page_classes"]
                    
                        coords = []
                        types = []
                        visions = []
                        texts = []
                        layout = []
                    
                        for s in screen_indices:                 
                            coords.append(torch.tensor(np.load(f"../dataset/{app}/{s}/gui_coords.npy")))
                            types.append(torch.tensor(np.load(f"../dataset/{app}/{s}/gui_functional_types.npy")))
                            visions.append(torch.tensor(np.load(f"../dataset/{app}/{s}/gui_vision_feature_maps.npy")))
                            texts.append(torch.tensor(np.load(f"../dataset/{app}/{s}/gui_text_embeds.npy")))
                        
                        padded_coords = pad_sequence(coords, batch_first=True).float().to(device)
                        padded_class_idx = pad_sequence(types, batch_first=True, padding_value=num_types-1).to(device)
                        padded_visions = pad_sequence(visions, batch_first=True).float().to(device)
                        padded_texts = pad_sequence(texts, batch_first=True).float().to(device)
                    
                        padding_mask = torch.tensor([
                            [1]*len(a) + [0]*(padded_coords.size(1) - len(a)) for a in types
                        ], dtype=torch.bool).to(device)
                    
                        embeds = model.encode(
                            coords=padded_coords, 
                            types=padded_class_idx,
                            visions=padded_visions,
                            texts=padded_texts,
                            padding_mask=padding_mask
                        )
        
                        app_intra_d, app_inter_d, app_score = validation_score(embeds, page_classes)
                        total_intra_distance += app_intra_d
                        total_inter_distance += app_inter_d
                        total_score += app_score
        
                print(f"Val {step} >> Intra: {total_intra_distance:.6f}, Inter: {total_inter_distance:.6f}, Score: {total_score:.6f}")
                if best_valid_score < total_score:
                    best_valid_score = total_score
                    torch.save(model.state_dict(), "./weights/bestmodel.pth")

                model.train()