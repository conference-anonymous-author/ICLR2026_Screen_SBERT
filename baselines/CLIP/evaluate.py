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
from transformers import get_scheduler, CLIPProcessor, CLIPModel
from PIL import Image

from utils import possible_same_pairs, possible_different_pairs, print_gradients, get_different_pairs, valid_macro_f1
from screen_class import X_class2idx, Temu_class2idx, Instagram_class2idx, Facebook_class2idx, Coupang_class2idx, Amazon_class2idx

device = "cuda:1"
app_list = ["Facebook", "Coupang"]
set_name = f"{app_list[0]}_{app_list[1]}"
model = CLIPModel.from_pretrained(f"./weights/CLIP32/{set_name}").to(device)
processor = CLIPProcessor.from_pretrained(f"./weights/CLIP32/{set_name}")

model.eval()

test_embeddings = {}
for app in app_list:
    test_embeddings[app] = {}

transforms_fn = transforms.ToTensor()

with torch.no_grad():
    for app in app_list:
        class_idx = class2idx_dict[app]
        for screen_indices in class_idx.values():
            for screen_idx in screen_indices:                
                image = [Image.open(f"./screenshots/{app}/{screen_idx}.png").convert("RGB")]
    
                inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
                vision_outputs = model.vision_model(**inputs)
                embeddings = vision_outputs.pooler_output
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                embedding = embeddings[0]

                test_embeddings[app][screen_idx] = embedding.detach().cpu().numpy()
                print(f"{screen_idx} of {app} completes")