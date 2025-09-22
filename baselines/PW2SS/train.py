import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from transformers import get_scheduler

from PW2SS import PW2SS

def load_dataset(app_list, train_valid_split, flag):
    screen_indices = []
    app_names = []
    
    for app in app_list:
        page2screen_indices = train_valid_split[f"{app}_{flag}"]
        for _, values in page2screen_indices.items():
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PW2SS(device=device).to(device)

    train_valid_split = np.load("../../dataset/train_valid_split.npy", allow_pickle=True).item()

    app_list = ["Facebook", "Instagram", "Amazon", "Coupang"]

    best_val_loss = float('inf')

    screen_indices_train, app_names_train = load_dataset(app_list, train_valid_split, flag="train")
    dataset_train = ScreenDataset(screen_indices_train, app_names_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=8,
        shuffle=True,
        collate_fn=DataLoader_fn
    )

    screen_indices_val, app_names_val = load_dataset(app_list, train_valid_split, flag="val")
    dataset_val = ScreenDataset(screen_indices_val, app_names_val)
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=True,
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
        for batch in dataloader_train:         
            screen_indices = batch["screen_indices"]
            app_names = batch["app_names"]
            
            text_embed = []
            text_coords = []
            graphic_types = []
            graphic_coords = []
            layout = []
        
            for bi in range(len(screen_indices)):
                si = screen_indices[bi]
                app = app_names[bi]
                
                # The input format used in PW2SS differs from that of other models.
                # The corresponding dataset is not publicly available in the current repository.
                # For details on the PW2SS input format, please refer to the original paper.
                # "You can immediately evaluate the retrieval performance using the provided embeddings file."

                text_embed.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/screen_ocr_embed.npy")))
                text_coords.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/screen_ocr_coords.npy")))
                graphic_types.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/gui_class_idx.npy")))
                graphic_coords.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/gui_coords.npy")))
                layout.append(transforms_fn(np.load(f"../../dataset/{app}/{si}/layout_image.npy")))

            padded_text_embed = pad_sequence(text_embed, batch_first=True).float().to(device)
            padded_text_coords = pad_sequence(text_coords, batch_first=True).float().to(device)
            padded_graphic_types = pad_sequence(graphic_types, batch_first=True, padding_value=27).to(device)
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
                
                graphic_types=padded_graphic_types, 
                graphic_coords=padded_graphic_coords, 
                graphic_mask=graphic_mask, 
                
                layout=padded_layout
            )
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for batch in dataloader_val:         
                screen_indices = batch["screen_indices"]
                app_names = batch["app_names"]
                
                text_embed = []
                text_coords = []
                graphic_types = []
                graphic_coords = []
                layout = []
            
                for bi in range(len(screen_indices)):
                    si = screen_indices[bi]
                    app = app_names[bi]
                    
                    text_embed.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/screen_ocr_embed.npy")))
                    text_coords.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/screen_ocr_coords.npy")))
                    graphic_types.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/gui_class_idx.npy")))
                    graphic_coords.append(torch.tensor(np.load(f"../../dataset/{app}/{si}/gui_coords.npy")))
                    layout.append(transforms_fn(np.load(f"../../dataset/{app}/{si}/layout_image.npy")))
    
                padded_text_embed = pad_sequence(text_embed, batch_first=True).float().to(device)
                padded_text_coords = pad_sequence(text_coords, batch_first=True).float().to(device)
                padded_graphic_types = pad_sequence(graphic_types, batch_first=True, padding_value=27).to(device)
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
                    
                    graphic_types=padded_graphic_types, 
                    graphic_coords=padded_graphic_coords, 
                    graphic_mask=graphic_mask, 
                    
                    layout=padded_layout
                )
                val_losses.append(val_loss.item())
    
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "./weights/bestmodel.pth")