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
from transformers import get_scheduler

from utils import possible_same_pairs, possible_different_pairs, print_gradients, get_different_pairs, valid_macro_f1
from SimilarityTransformer import SimilarityTransformer

def prepare_trainset(app_list, split_data):
    pairs = []
    app_names = []
    sim_labels = []
    
    for app in app_list:
        class2idx = split_data[f"{app}_train"]
        
        same_pairs = possible_same_pairs(class2idx)
        same_labels = [1]*len(same_pairs)
    
        pairs += same_pairs
        app_names += [app]*len(same_pairs)
        sim_labels += same_labels

    return pairs, app_names, sim_labels

def prepare_valset(app_list, split_data, flag="val"):
    pairs = []
    app_names = []
    sim_labels = []
    
    for app in app_list:
        class2idx = split_data[f"{app}_{flag}"]
        
        same_pairs = possible_same_pairs(class2idx)
        same_labels = [1]*len(same_pairs)
    
        different_pairs = possible_different_pairs(class2idx)
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

def make_different_set(app_cnt, split_data, flag):
    different_pairs = []
    different_app_names = []
    for app in app_cnt.keys():
        num_app = app_cnt[app]
        class2idx = split_data[f"{app}_{flag}"]
        pairs = get_different_pairs(class2idx, k=num_app)
        different_pairs += pairs
        different_app_names += [app] * num_app

    return different_pairs, different_app_names

if __name__ == "__main__":
    device = "cuda:0"
    model = SimilarityTransformer(device=device).to(device)

    split_data = np.load("./splits/split_0724.npy", allow_pickle=True).item()

    batch_size = 4
    num_workers = 0
    max_epoch = 100
    model_path = "./weights/0818/other_apps/SST/X_Temu"

    app_list = ["Instagram", "Facebook", "Amazon", "Coupang"]

    total_best_f1 = 0
    best_epochs = []
    best_threshold_list = []

    pairs_train, app_names_train, labels_train = prepare_trainset(app_list, split_data)
    dataset_train = ScreenPairDataset(pairs_train, app_names_train, labels_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DataLoader_fn
    )
    pairs_val, app_names_val, labels_val = prepare_valset(app_list, split_data, flag="val")
    dataset_val = ScreenPairDataset(pairs_val, app_names_val, labels_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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

    transforms_fn = transforms.ToTensor()
    
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
            different_pairs, different_app_names = make_different_set(app_cnt, split_data, flag="train")

            pairs = same_pairs + different_pairs
            app_names = same_app_names + different_app_names
            true_sim = torch.cat([same_true_sim, 1-same_true_sim], dim=0).to(device)
        
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
        
            logits, coord_loss, class_loss, vision_loss = model(
                coords1=padded_coords_A, 
                class_idx1=padded_class_idx_A,
                visions1=padded_visions_A,
                padding_mask1=padding_mask_A, 
                
                coords2=padded_coords_B, 
                class_idx2=padded_class_idx_B, 
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
            #print_gradients(model)
            #break
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_count += 1
            if batch_count % (num_batch//10) == 0:
                log_msg = f"(X_Temu) Epoch {epoch}, Batch {batch_count}/{num_batch} >> "
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
            best_epochs.append(epoch)
            #torch.save(model.state_dict(), f"{model_path}/checkpoint_{epoch}.pth")
            torch.save(model.state_dict(), f"{model_path}/bestmodel.pth")
            np.save(f"{model_path}/best_epochs.npy", best_epochs)
            with open(f"{model_path}/val_log.txt", 'a') as f:
                f.write(f"Val {epoch} >> Macro F1: {macro_f1:.4f}, Pos: {f1_pos:.4f}, Neg: {f1_neg:.4f}\n")
                f.write(f"Val {epoch} >> Best F1: {f1[best_index]:.4f}, P: {precision[best_index]:.4f}, R: {recall[best_index]:.4f}\n")