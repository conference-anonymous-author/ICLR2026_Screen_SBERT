import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_recall_curve, f1_score
from transformers import get_scheduler
from itertools import combinations
import random

from utils import possible_same_pairs, possible_different_pairs, get_different_pairs, valid_macro_f1
from SimilarityTransformer import SimilarityTransformer

def possible_same_pairs(page2screen_indices):
    same_pairs = []
    for screen_indices in page2screen_indices.values():
        if len(screen_indices) >= 2:
            pairs = list(combinations(screen_indices, 2))
            reversed_pairs = [(b, a) for (a, b) in pairs]
            same_pairs += pairs
            same_pairs += reversed_pairs
    return same_pairs

def possible_different_pairs(page2screen_indices):
    different_pairs = []
    page_classes = list(page2screen_indices.keys())
    for class1, class2 in combinations(page_classes, 2):
        for i in page2screen_indices[class1]:
            for j in page2screen_indices[class2]:
                different_pairs.append((i, j))
                different_pairs.append((j, i))
    return different_pairs

def get_different_pairs(page2screen_indices, k, ignore_pair=(-1, -1)):
    different_pairs = []
    possible = [page_class for page_class, screen_indices in page2screen_indices.items() if len(screen_indices) > 0]
    for n in range(k):
        while True:
            classes = random.sample(possible, k=2)
            index1 = random.sample(page2screen_indices[classes[0]], k=1)[0]
            index2 = random.sample(page2screen_indices[classes[1]], k=1)[0]
            if (index1, index2) != ignore_pair and (index2, index1) != ignore_pair:
                break
        if random.random() < 0.5:
            different_pairs.append((index1, index2))
        else:
            different_pairs.append((index2, index1))

    return different_pairs

def valid_macro_f1(val_labels, val_probs):
    thresholds = np.linspace(0.0, 1.0, num=1000)
    
    macro_f1_list = []
    f1_pos_list = []
    f1_neg_list = []
    for t in thresholds:
        val_preds = (val_probs >= t).astype(int)

        f1_pos = f1_score(val_labels, val_preds, pos_label=1)
        f1_neg = f1_score(1 - val_labels, 1 - val_preds, pos_label=1)
        macro_f1 = (f1_pos + f1_neg)/2
        
        macro_f1_list.append(macro_f1)
        f1_pos_list.append(f1_pos)
        f1_neg_list.append(f1_neg)

    best_idx = np.argmax(macro_f1_list)

    return macro_f1_list[best_idx], f1_pos_list[best_idx], f1_neg_list[best_idx]

def load_train_data(app_list, train_valid_split):
    pairs = []
    app_names = []
    sim_labels = []
    
    for app in app_list:
        page2screen_indices = train_valid_split[f"{app}_train"]
        
        same_pairs = possible_same_pairs(page2screen_indices)
        same_labels = [1]*len(same_pairs)
    
        pairs += same_pairs
        app_names += [app]*len(same_pairs)
        sim_labels += same_labels

    return pairs, app_names, sim_labels

def load_valid_data(app_list, train_valid_split):
    pairs = []
    app_names = []
    sim_labels = []
    
    for app in app_list:
        page2screen_indices = train_valid_split[f"{app}_val"]
        
        same_pairs = possible_same_pairs(page2screen_indices)
        same_labels = [1]*len(same_pairs)
    
        different_pairs = possible_different_pairs(page2screen_indices)
        different_labels = [0]*len(different_pairs)
    
        pairs += same_pairs
        pairs += different_pairs
        app_names += [app]*len(same_pairs + different_pairs)
        sim_labels += same_labels
        sim_labels += different_labels

    return pairs, app_names, sim_labels

class ScreenPairDataset(Dataset):
    def __init__(self, pairs, app_names, sim_labels):
        self.pairs = pairs
        self.app_names = app_names
        self.sim_labels = sim_labels

    def __len__(self):
        return len(self.sim_labels)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        app = self.app_names[idx]
        
        return {
            "pair": self.pairs[idx],
            "app_name": self.app_names[idx],
            "sim_label": self.sim_labels[idx]
        }

def DataLoader_fn(batch):
    pairs = []
    app_names = []
    sim_labels = []

    for item in batch:
        pairs.append(item["pair"])
        app_names.append(item["app_name"])
        sim_labels.append(item["sim_label"])
    
    return {
        "pairs": pairs,
        "app_names": app_names,
        "true_sim": torch.tensor(sim_labels)
    }

def count_app(app_list, app_names):
    app_cnt = {}
    for app in app_list:
        app_cnt[app] = 0
    for app in app_names:
        app_cnt[app] += 1

    return app_cnt

def make_different_set(app_cnt, train_valid_split, flag):
    different_pairs = []
    different_app_names = []
    for app in app_cnt.keys():
        num_app = app_cnt[app]
        page2screen_indices = train_valid_split[f"{app}_{flag}"]
        pairs = get_different_pairs(page2screen_indices, k=num_app)
        different_pairs += pairs
        different_app_names += [app] * num_app

    return different_pairs, different_app_names

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimilarityTransformer(device=device).to(device)

    train_valid_split = train_valid_split = np.load("../../dataset/train_valid_split.npy", allow_pickle=True).item()

    batch_size = 4
    max_epoch = 100

    app_list = ["Instagram", "Facebook", "Amazon", "Coupang"]

    total_best_f1 = 0

    pairs_train, app_names_train, labels_train = load_train_data(app_list, train_valid_split)
    dataset_train = ScreenPairDataset(pairs_train, app_names_train, labels_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataLoader_fn
    )
    pairs_val, app_names_val, labels_val = load_valid_data(app_list, train_valid_split)
    dataset_val = ScreenPairDataset(pairs_val, app_names_val, labels_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataLoader_fn
    )
    num_batch = len(dataloader_train)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    total_steps = len(dataloader_train) * max_epoch
    warmup_steps = int(0.1*total_steps)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    sim_loss_fn = nn.BCEWithLogitsLoss()
    
    for epoch in range(max_epoch):
        model.train()
        sim_losses_epoch = []
        coord_losses_epoch = []
        class_losses_epoch = []
        vision_losses_epoch = []
        batch_count = 0
        
        for batch in dataloader_train:
            optimizer.zero_grad()
            
            same_pairs = batch["pairs"]
            same_app_names = batch["app_names"]
            same_true_sim = batch["true_sim"].float()

            app_cnt = count_app(app_list, same_app_names)
            different_pairs, different_app_names = make_different_set(app_cnt, train_valid_split, flag="train")

            pairs = same_pairs + different_pairs
            app_names = same_app_names + different_app_names
            true_sim = torch.cat([same_true_sim, 1-same_true_sim], dim=0).to(device)
        
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
        
            logits, coord_loss, class_loss, vision_loss = model(
                coords1=padded_coords_A, 
                types1=padded_types_A,
                visions1=padded_visions_A,
                padding_mask1=padding_mask_A, 
                
                coords2=padded_coords_B, 
                types2=padded_types_B, 
                visions2=padded_visions_B,
                padding_mask2=padding_mask_B,
                
                reconstruct_flag=True
            )
            sim_loss = sim_loss_fn(logits, true_sim)

            sim_losses_epoch.append(sim_loss.item())
            coord_losses_epoch.append(coord_loss.item())
            class_losses_epoch.append(class_loss.item())
            vision_losses_epoch.append(vision_loss.item())
            
            train_loss = sim_loss + coord_loss + class_loss + vision_loss
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_count += 1
            if batch_count % (num_batch//10) == 0:
                log_msg = f"Epoch {epoch}, Batch {batch_count}/{num_batch} >> "
                log_msg += f"Sim: {np.mean(sim_losses_epoch).item():.4f}, "
                log_msg += f"Crd: {np.mean(coord_losses_epoch).item():.4f}, "
                log_msg += f"Cls: {np.mean(class_losses_epoch).item():.4f}, "
                log_msg += f"Vis: {np.mean(vision_losses_epoch).item():.4f}, "
                print(log_msg)
        
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch in dataloader_val:           
                pairs = batch["pairs"]
                app_names = batch["app_names"]
                true_sim = batch["true_sim"].float().to(device)
            
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

                val_probs.append(probs.cpu().numpy())
                val_labels.append(true_sim.cpu().numpy())

        val_probs = np.concat(val_probs)
        val_labels = np.concat(val_labels)

        macro_f1, f1_pos, f1_neg = valid_macro_f1(val_labels, val_probs)
        print(f"Val {epoch} >> Macro F1: {macro_f1:.4f}, Pos: {f1_pos:.4f}, Neg: {f1_neg:.4f}")

        precision, recall, thresholds = precision_recall_curve(val_labels, val_probs)
        f1 = 2*precision*recall / (precision+recall+1e-8)
        best_index = np.argmax(f1)
        print(f"Val {epoch} >> Best F1: {f1[best_index]:.4f}, P: {precision[best_index]:.4f}, R: {recall[best_index]:.4f}")

        if total_best_f1 < macro_f1:
            total_best_f1 = macro_f1
            torch.save(model.state_dict(), "./weights/bestmodel.pth")