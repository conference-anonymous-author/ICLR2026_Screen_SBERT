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
from transformers import get_scheduler, CLIPProcessor, CLIPModel
from PIL import Image

from utils import possible_same_pairs, possible_different_pairs, print_gradients, get_different_pairs, valid_macro_f1, generate_class2num
from screen_class import Instagram_class2idx, Facebook_class2idx, X_class2idx, Amazon_class2idx, Coupang_class2idx, Temu_class2idx

def prepare_trainset(app_list, split_data):
    screen_indices = []
    page_labels = []
    app_names = []
    
    for app in app_list:
        class2idx = split_data[f"{app}_train"]
        for page_label, values in class2idx.items():
            screen_indices += values
            page_labels += ([page_label]*len(values))
            app_names += ([app]*len(values))

    return screen_indices, page_labels, app_names

def prepare_valset(app_list, split_data):
    valid_set = {}
    
    for app in app_list:
        valid_set[app] = {
            "screen_indices": [],
            "page_labels": []
        }
        class2idx = split_data[f"{app}_val"]
        for page_text, values in class2idx.items():
            valid_set[app]["screen_indices"] += values
            valid_set[app]["page_labels"] += ([class2num_dict[app][page_text]]*len(values))

    return valid_set
    
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
            "page_label": self.page_labels[idx],
            "app_name": self.app_names[idx]
        }

def DataLoader_fn(batch):
    screen_indices = []
    page_labels = []
    app_names = []

    for item in batch:
        screen_indices.append(item["screen_idx"])
        page_labels.append(item["page_label"])
        app_names.append(item["app_name"])
    
    return {
        "screen_indices": screen_indices,
        "page_labels": page_labels,
        "app_names": app_names
    }

def make_contrastive_batch(anchor):
    anchor_idx = anchor["screen_indices"][0]
    anchor_page = anchor["page_labels"][0]
    app_name = anchor["app_names"][0]

    class2idx = class2idx_dict[app_name]
    if len(class2idx[anchor_page]) < 2:
        negative_only = True
    else:
        negative_only = False

    samples = [anchor_idx]
    for page_label, screen_indices in class2idx.items():
        if page_label == anchor_page:
            if not negative_only:
                positive_set = copy.deepcopy(screen_indices)
                positive_set.remove(anchor_idx)
                positive = random.choice(positive_set)
                #positive_pair = (anchor_idx, positive)
                samples.append(positive)

    for page_label, screen_indices in class2idx.items():
        if page_label != anchor_page:
            negative = random.choice(screen_indices)
            samples.append(negative)

    return samples, negative_only, app_name

def valid_score(embeddings, labels):
    labels_tensor = torch.tensor(labels, device=embeddings.device)
    unique_labels = labels_tensor.unique()
    class_to_indices = defaultdict(list)

    for idx, label in enumerate(labels):
        class_to_indices[label].append(idx)

    class_means = {}
    for label in unique_labels:
        idxs = class_to_indices[label.item()]
        class_embeds = embeddings[idxs]
        class_mean = class_embeds.mean(dim=0)
        class_means[label.item()] = class_mean

    # Intra-class distance
    intra_dists = []
    for label in unique_labels:
        idxs = class_to_indices[label.item()]
        class_embeds = embeddings[idxs]
        mean = class_means[label.item()]
        dists = 1 - F.cosine_similarity(class_embeds, mean.unsqueeze(0), dim=1)
        intra_dists.append(dists.max()) # 가장 먼 것 하나만

    intra_class_distance = torch.stack(intra_dists).mean().item()

    # Inter-class distance (mean of all pairwise distances between class centers)
    inter_dist = 0
    for ei, true_label in enumerate(labels):
        emb = embeddings[ei]
        min_dist = float('inf')
        for other_label in unique_labels:
            if other_label == true_label:
                continue
            mean = class_means[other_label.item()]
            dist = 1 - F.cosine_similarity(emb.unsqueeze(0), mean.unsqueeze(0)).item()
            if dist < min_dist:
                min_dist = dist
        inter_dist += min_dist

    inter_class_distance = inter_dist / len(labels)

    #score = inter_class_distance / (intra_class_distance + 1e-8)
    score = inter_class_distance - intra_class_distance

    return intra_class_distance, inter_class_distance, score

if __name__ == "__main__":
    device = "cuda:0"
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

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
    model_path = "./weights/CLIP32/Facebook_Amazon"

    app_list = ["Instagram", "X", "Coupang", "Temu"]

    best_val_score = 0

    screen_indices_train, page_labels_train, app_names_train = prepare_trainset(app_list, split_data)
    dataset_train = ScreenDataset(screen_indices_train, page_labels_train, app_names_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=DataLoader_fn
    )
    num_batch = len(dataloader_train)

    valid_set = prepare_valset(app_list, split_data)

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

    transforms_fn = transforms.ToTensor()

    step = 0
    model.train()
    optimizer.zero_grad()
    for epoch in range(max_epoch):
        for anchor in dataloader_train:         
            samples, negative_only, app = make_contrastive_batch(anchor)
            
            images = []
            for s in samples:
                images.append(Image.open(f"./screenshots/{app}/{s}.png").convert("RGB"))

            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            vision_outputs = model.vision_model(**inputs)
            embeddings = vision_outputs.pooler_output
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

            # embeddings: (N, D), L2 정규화 완료 가정
            anchor = embeddings[0:1]           # (1, D)
            others = embeddings[1:]            # (N-1, D)
            
            # 코사인 유사도: (N-1,)
            cosine_similarities = (others @ anchor.T).squeeze(-1)
            
            if negative_only:
                contrastive_loss = -torch.log(1 - cosine_similarities.clamp(max=1 - eps))
                contrastive_loss = contrastive_loss.mean()
                contrastive_loss = torch.clamp(contrastive_loss, min=0.0)
            else:
                probs = torch.softmax(cosine_similarities / temperature, dim=0)
                contrastive_loss = -torch.log(probs[0] + eps) # 항상 0번째가 positive pair

            contrastive_loss.backward()
            #print_gradients(model)
            #break
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log_msg = f"(Facebook_Amazon) Train {step} >> "
            log_msg += f"Loss: {contrastive_loss.item():.4f}, "
            log_msg += f"NO: {negative_only}"
            print(log_msg)

            step += 1
            if step % 10 == 0: ## Validation ##
                model.eval()
        
                total_inter_d = 0
                total_intra_d = 0
                total_score = 0
                
                with torch.no_grad():
                    for app in app_list:
                        screen_indices = valid_set[app]["screen_indices"]
                        page_labels = valid_set[app]["page_labels"]
                    
                        images = []
                    
                        for si in screen_indices:       
                            images.append(Image.open(f"./screenshots/{app}/{si}.png").convert("RGB"))
                    
                        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                        vision_outputs = model.vision_model(**inputs)
                        embeddings = vision_outputs.pooler_output
                        embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
        
                        app_intra_d, app_inter_d, app_score = valid_score(embeddings, page_labels)
                        total_intra_d += app_intra_d
                        total_inter_d += app_inter_d
                        total_score += app_score
        
                print(f"Val {step} >> Intra: {total_intra_d:.6f}, Inter: {total_inter_d:.6f}, Score: {total_score:.6f}")
                if best_val_score < total_score:
                    best_val_score = total_score
                    #torch.save(model.state_dict(), f"{model_path}/checkpoint_{step}.pth")
                    #torch.save(model.state_dict(), f"{model_path}/bestmodel.pth")
                    model.save_pretrained(f"{model_path}")
                    processor.save_pretrained(f"{model_path}")
                    with open(f"{model_path}/val_log.txt", 'a') as f:
                        f.write(f"Val {step} >> Intra: {total_intra_d:.6f}, Inter: {total_inter_d:.6f}, Score: {total_score:.6f}\n")

                model.train()

        #break