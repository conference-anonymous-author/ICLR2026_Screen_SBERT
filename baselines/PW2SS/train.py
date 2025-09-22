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
from PW2SS import PW2SS

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
    device = "cuda:0"
    model = PW2SS(device=device).to(device)

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
    model_path = "./weights/other_apps/PW2SS/X_Temu"

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

    transforms_fn = transforms.ToTensor()

    step = 0
    optimizer.zero_grad()
    for epoch in range(max_epoch):
        model.train()
        train_losses = []
        for batch in dataloader_train:         
            screen_indices = batch["screen_indices"]
            app_names = batch["app_names"]
            
            text_embed = []
            text_coords = []
            graphic_class_idx = []
            graphic_coords = []
            layout = []
        
            for bi in range(len(screen_indices)):
                si = screen_indices[bi]
                app = app_names[bi]
                
                text_embed.append(torch.tensor(np.load(f"./dataset/{app}/{si}/screen_ocr_embed.npy")))
                text_coords.append(torch.tensor(np.load(f"./dataset/{app}/{si}/screen_ocr_coords.npy")))
                graphic_class_idx.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_class_idx.npy")))
                graphic_coords.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_coords.npy")))
                layout.append(transforms_fn(np.load(f"./dataset/{app}/{si}/layout.npy")))

            padded_text_embed = pad_sequence(text_embed, batch_first=True).float().to(device)
            padded_text_coords = pad_sequence(text_coords, batch_first=True).float().to(device)
            padded_graphic_class_idx = pad_sequence(graphic_class_idx, batch_first=True, padding_value=27).to(device)
            padded_graphic_coords = pad_sequence(graphic_coords, batch_first=True).float().to(device)
            padded_layout = pad_sequence(layout, batch_first=True).float().to(device)
        
            text_mask = torch.tensor([
                [1]*tc.size(0) + [0]*(padded_text_coords.size(1) - tc.size(0)) for tc in text_coords
            ], dtype=torch.bool).to(device)
            graphic_mask = torch.tensor([
                [1]*gc.size(0) + [0]*(padded_graphic_coords.size(1) - gc.size(0)) for gc in graphic_coords
            ], dtype=torch.bool).to(device)

            loss = model.mask_predict(
                text_embed=padded_text_embed, 
                text_coords=padded_text_coords, 
                text_mask=text_mask, 
                
                graphic_class_idx=padded_graphic_class_idx, 
                graphic_coords=padded_graphic_coords, 
                graphic_mask=graphic_mask, 
                
                layout=padded_layout
            )
            loss.backward()
            train_losses.append(loss.item())
            #print_gradients(model)
            #break
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #break

        log_msg = f"(PW2SS_X_Temu) Train {epoch} >> "
        log_msg += f"Loss: {np.mean(train_losses):.6f}"
        print(log_msg)

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in dataloader_val:         
                screen_indices = batch["screen_indices"]
                app_names = batch["app_names"]
                
                text_embed = []
                text_coords = []
                graphic_class_idx = []
                graphic_coords = []
                layout = []
            
                for bi in range(len(screen_indices)):
                    si = screen_indices[bi]
                    app = app_names[bi]
                    
                    text_embed.append(torch.tensor(np.load(f"./dataset/{app}/{si}/screen_ocr_embed.npy")))
                    text_coords.append(torch.tensor(np.load(f"./dataset/{app}/{si}/screen_ocr_coords.npy")))
                    graphic_class_idx.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_class_idx.npy")))
                    graphic_coords.append(torch.tensor(np.load(f"./dataset/{app}/{si}/gui_coords.npy")))
                    layout.append(transforms_fn(np.load(f"./dataset/{app}/{si}/layout.npy")))
    
                padded_text_embed = pad_sequence(text_embed, batch_first=True).float().to(device)
                padded_text_coords = pad_sequence(text_coords, batch_first=True).float().to(device)
                padded_graphic_class_idx = pad_sequence(graphic_class_idx, batch_first=True, padding_value=27).to(device)
                padded_graphic_coords = pad_sequence(graphic_coords, batch_first=True).float().to(device)
                padded_layout = pad_sequence(layout, batch_first=True).float().to(device)
            
                text_mask = torch.tensor([
                    [1]*tc.size(0) + [0]*(padded_text_coords.size(1) - tc.size(0)) for tc in text_coords
                ], dtype=torch.bool).to(device)
                graphic_mask = torch.tensor([
                    [1]*gc.size(0) + [0]*(padded_graphic_coords.size(1) - gc.size(0)) for gc in graphic_coords
                ], dtype=torch.bool).to(device)
    
                val_loss = model.mask_predict(
                    text_embed=padded_text_embed, 
                    text_coords=padded_text_coords, 
                    text_mask=text_mask, 
                    
                    graphic_class_idx=padded_graphic_class_idx, 
                    graphic_coords=padded_graphic_coords, 
                    graphic_mask=graphic_mask, 
                    
                    layout=padded_layout
                )
                val_losses.append(val_loss.item())
    
            val_loss = np.mean(val_losses)
            log_msg = f"(PW2SS) Val {epoch} >> "
            log_msg += f"Loss: {val_loss:.6f}"
            print(log_msg)
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                #torch.save(model.state_dict(), f"{model_path}/checkpoint_{epoch}.pth")
                torch.save(model.state_dict(), f"{model_path}/bestmodel.pth")
                with open(f"{model_path}/val_log.txt", 'a') as f:
                    f.write(log_msg + "\n")