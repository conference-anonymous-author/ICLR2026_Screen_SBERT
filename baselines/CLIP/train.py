import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import copy
from collections import defaultdict
from transformers import get_scheduler, CLIPProcessor, CLIPModel
from PIL import Image

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
    def __init__(self, screen_indices, page_labels, app_names):
        self.screen_indices = screen_indices
        self.page_labels = page_labels
        self.app_names = app_names

    def __len__(self):
        return len(self.app_names)

    def __getitem__(self, idx):   
        return {
            "screen_idx": self.screen_indices[idx],
            "page_class": self.page_labels[idx],
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
    anchor_idx = anchor["screen_indices"][0]
    anchor_page = anchor["page_classes"][0]
    app_name = anchor["app_names"][0]

    page2screen_indices = page2screen_indices_dict[app_name]
    if len(page2screen_indices[anchor_page]) < 2:
        negative_only = True
    else:
        negative_only = False

    samples = [anchor_idx]
    for page_class, screen_indices in page2screen_indices.items():
        if page_class == anchor_page:
            if not negative_only:
                positive_set = copy.deepcopy(screen_indices)
                positive_set.remove(anchor_idx)
                positive = random.choice(positive_set)
                samples.append(positive)

    for page_class, screen_indices in page2screen_indices.items():
        if page_class != anchor_page:
            negative = random.choice(screen_indices)
            samples.append(negative)

    return samples, negative_only, app_name

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
    model_name = "openai/clip-vit-base-patch32" # or "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    train_valid_split = np.load("../dataset/train_valid_split.npy", allow_pickle=True).item()

    app_list = ["Instagram", "X", "Coupang", "Temu"]

    best_val_score = 0

    screen_indices_train, page_classes_train, app_names_train = load_train_data(app_list, train_valid_split)
    train_dataset = ScreenDataset(screen_indices_train, page_classes_train, app_names_train)
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=DataLoader_fn
    )
    num_batch = len(dataloader_train)

    valid_dataset = load_valid_data(app_list, train_valid_split)

    max_epoch = 5
    total_steps = len(dataloader_train) * max_epoch
    warmup_steps = int(0.1*total_steps)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    temperature = 0.07
    eps = 1e-6

    step = 0
    model.train()
    optimizer.zero_grad()
    for epoch in range(max_epoch):
        for anchor in dataloader_train:         
            samples, negative_only, app = make_contrastive_batch(anchor)
            
            images = []
            for si in samples:
                images.append(Image.open(f"./screenshots/{app}/{si}.jpg").convert("RGB"))

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            vision_outputs = model.vision_model(**inputs)
            embeddings = vision_outputs.pooler_output
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            anchor = embeddings[0:1]
            others = embeddings[1:]
            
            cosine_similarities = (others @ anchor.T).squeeze(-1)
            
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
            if step % 10 == 0: ## Validation ##
                model.eval()
        
                total_inter_d = 0
                total_intra_d = 0
                total_score = 0
                
                with torch.no_grad():
                    for app in app_list:
                        screen_indices = valid_dataset[app]["screen_indices"]
                        page_labels = valid_dataset[app]["page_classes"]
                    
                        images = []
                    
                        for si in screen_indices:       
                            images.append(Image.open(f"./screenshots/{app}/{si}.png").convert("RGB"))
                    
                        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                        vision_outputs = model.vision_model(**inputs)
                        embeddings = vision_outputs.pooler_output
                        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        
                        app_intra_d, app_inter_d, app_score = validation_score(embeddings, page_labels)
                        total_intra_d += app_intra_d
                        total_inter_d += app_inter_d
                        total_score += app_score
        
                print(f"Val {step} >> Intra: {total_intra_d:.6f}, Inter: {total_inter_d:.6f}, Score: {total_score:.6f}")
                if best_val_score < total_score:
                    best_val_score = total_score
                    model.save_pretrained(f"./weights/B_32")

                model.train()