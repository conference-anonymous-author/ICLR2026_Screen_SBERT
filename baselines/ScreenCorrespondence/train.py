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
import copy
from collections import defaultdict
from transformers import get_scheduler

from utils import possible_same_pairs, possible_different_pairs, print_gradients, get_different_pairs, valid_macro_f1, generate_class2num
from screen_class import Instagram_class2idx, Facebook_class2idx, Amazon_class2idx, Coupang_class2idx, X_class2idx, Temu_class2idx
from ScreenCorrespondence import ScreenCorrespondence

def prepare_dataset(app_list, split_data, flag):
    screen_indices = []
    app_names = []
    
    for app in app_list:
        class2idx = split_data[f"{app}_{flag}"]
        for page_label, values in class2idx.items():
            screen_indices += values
            app_names += ([app]*len(values))

    return screen_indices, app_names
    
class ScreenDataset(Dataset):
    def __init__(self, screen_indices, app_names):
        self.screen_indices = screen_indices
        self.app_names = app_names

    def __len__(self):
        return len(self.app_names)

    def __getitem__(self, idx):   
        return {
            "screen_idx": self.screen_indices[idx],
            "app_name": self.app_names[idx]
        }

def DataLoader_fn(batch):
    screen_indices = []
    app_names = []

    for item in batch:
        screen_indices.append(item["screen_idx"])
        app_names.append(item["app_name"])
    
    return {
        "screen_indices": screen_indices,
        "app_names": app_names
    }

if __name__ == "__main__":
    device = "cuda:1"
    model = ScreenCorrespondence(device=device).to(device)

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

    split_data = np.load("./splits/split_0724.npy", allow_pickle=True).item()

    num_workers = 0
    model_path = "./weights/other_apps/SC/X_Temu"

    app_list = ["Facebook", "Instagram", "Amazon", "Coupang"]

    best_val_loss = float('inf')

    screen_indices_train, app_names_train = prepare_dataset(app_list, split_data, flag="train")
    dataset_train = ScreenDataset(screen_indices_train, app_names_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DataLoader_fn
    )
    num_batch = len(dataloader_train)

    screen_indices_val, app_names_val = prepare_dataset(app_list, split_data, flag="val")
    dataset_val = ScreenDataset(screen_indices_val, app_names_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DataLoader_fn
    )

    max_epoch = 200
    total_steps = len(dataloader_train) * max_epoch
    warmup_steps = int(0.1*total_steps)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    temperature = 0.07
    eps = 1e-6

    transforms_fn = transforms.ToTensor()

    step = 0
    model.train()
    optimizer.zero_grad()
    for epoch in range(max_epoch):
        model.train()
        train_losses = []
        for batch in dataloader_train:         
            screen_indices = batch["screen_indices"]
            app_names = batch["app_names"]
            
            coords = []
            class_idx = []
            visions = []
            texts = []
            
            for bi in range(len(screen_indices)):
                si = screen_indices[bi]
                app = app_names[bi]
    
                coords.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_coords.npy")))
                class_idx.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_class_idx.npy")))
                visions.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_visions.npy")))
                texts.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_text_embed.npy")))
    
            padded_coords = pad_sequence(coords, batch_first=True).float().to(device)
            padded_class_idx = pad_sequence(class_idx, batch_first=True, padding_value=27).to(device)
            padded_visions = pad_sequence(visions, batch_first=True).float().to(device)
            padded_texts = pad_sequence(texts, batch_first=True).float().to(device)
            
            padding_mask = torch.tensor([
                [1]*len(a) + [0]*(padded_coords.size(1) - len(a)) for a in class_idx
            ], dtype=torch.bool).to(device)
    
            loss = model.mask_predict(
                coords=padded_coords, 
                class_idx=padded_class_idx, 
                visions=padded_visions, 
                texts=padded_texts, 
                padding_mask=padding_mask
            )
            
            loss.backward()
            train_losses.append(loss.item())
            #print_gradients(model)
            #break
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #break

        log_msg = f"(X_Temu) Train {epoch} >> "
        log_msg += f"Loss: {np.mean(train_losses):.6f}"
        print(log_msg)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in dataloader_val:         
                screen_indices = batch["screen_indices"]
                app_names = batch["app_names"]

                coords = []
                class_idx = []
                visions = []
                texts = []
            
                for bi in range(len(screen_indices)):
                    si = screen_indices[bi]
                    app = app_names[bi]
    
                    coords.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_coords.npy")))
                    class_idx.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_class_idx.npy")))
                    visions.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_visions.npy")))
                    texts.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_text_embed.npy")))
    
                padded_coords = pad_sequence(coords, batch_first=True).float().to(device)
                padded_class_idx = pad_sequence(class_idx, batch_first=True, padding_value=27).to(device)
                padded_visions = pad_sequence(visions, batch_first=True).float().to(device)
                padded_texts = pad_sequence(texts, batch_first=True).float().to(device)
            
                padding_mask = torch.tensor([
                    [1]*len(a) + [0]*(padded_coords.size(1) - len(a)) for a in class_idx
                ], dtype=torch.bool).to(device)
    
                val_loss = model.mask_predict(
                    coords=padded_coords, 
                    class_idx=padded_class_idx, 
                    visions=padded_visions, 
                    texts=padded_texts, 
                    padding_mask=padding_mask
                )
                val_losses.append(val_loss.item())
    
            val_loss = np.mean(val_losses)
            log_msg = f"Val {epoch} >> "
            log_msg += f"Loss: {val_loss:.6f}"
            print(log_msg)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #torch.save(model.state_dict(), f"{model_path}/checkpoint_{epoch}.pth")
                torch.save(model.state_dict(), f"{model_path}/bestmodel.pth")
                with open(f"{model_path}/val_log.txt", 'a') as f:
                    f.write(log_msg + "\n")