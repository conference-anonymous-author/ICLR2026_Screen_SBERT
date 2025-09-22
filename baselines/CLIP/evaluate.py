import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from tqdm import tqdm
import numpy as np

from dataset.page_labels import page2screen_indices_dict, page_indices_dict, screen_idx2page_dict

def embed(model, evaluation_apps):
    embeddings = {}
    for app in evaluation_apps:
        embeddings[app] = {}

    with torch.no_grad():
        for app in evaluation_apps:
            page2screen_indices = page2screen_indices_dict[app]
            for screen_indices in tqdm(page2screen_indices.values(), desc=f"Embedding {app}"):
                for screen_idx in screen_indices:                
                    image = [Image.open(f"./screenshots/{app}/{screen_idx}.jpg")]
        
                    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
                    vision_outputs = model.vision_model(**inputs)
                    embeddings = vision_outputs.pooler_output
                    embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)
                    embedding = embeddings[0]

                    embeddings[app][screen_idx] = embedding.detach().cpu().numpy()

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "./weights/B_16/evaluate_X_Temu"
    model = CLIPModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    evaluation_apps = ["X", "Temu"]
    embeddings_file = f"./embeddings/B_16/evaluate_{evaluation_apps[0]}_{evaluation_apps[1]}.npy"
    if os.path.exists(embeddings_file):
        embeddings = np.load(embeddings_file, allow_pickle=True).item()
    else:
        embeddings = embed(model, evaluation_apps)
        np.save(embeddings_file, embeddings, allow_pickle=True)

    truth = []
    top1 = []
    top2 = []
    top3 = []

    for app in evaluation_apps:
        page2screen_indices = page2screen_indices_dict[app]
        one_app_embeddings = embeddings[app]

        for page_idx, screen_indices in tqdm(enumerate(page2screen_indices.values()), desc=f"Evaluating {app}"):
            if len(screen_indices) < 2:
                continue
            for query in screen_indices:    
                embedding_space = []
                si_list = []
                for si, embed in one_app_embeddings.items():
                    if si == query:
                        continue
                    embedding_space.append(embed)
                    si_list.append(si)
                embedding_space = np.array(embedding_space)
                
                query_emb = one_app_embeddings[query].reshape(1, -1)

                nn1 = NearestNeighbors(n_neighbors=1, metric='cosine')
                nn1.fit(embedding_space)
                _, indices1 = nn1.kneighbors(query_emb)
                try:
                    pred1 = [page_indices_dict[app][screen_idx2page_dict[app][si_list[indices1[0].item()]]]]
                except:
                    print(si_list[indices1[0].item()])
        
                nn2 = NearestNeighbors(n_neighbors=2, metric='cosine')
                nn2.fit(embedding_space)
                _, indices2 = nn2.kneighbors(query_emb)
                pred2 = [page_indices_dict[app][screen_idx2page_dict[app][si_list[indices2[0][k].item()]]] for k in range(2)] 
        
                nn3 = NearestNeighbors(n_neighbors=3, metric='cosine')
                nn3.fit(embedding_space)
                _, indices3 = nn3.kneighbors(query_emb)
                pred3 = [page_indices_dict[app][screen_idx2page_dict[app][si_list[indices3[0][k].item()]]] for k in range(3)] 
        
                truth.append(page_idx)
                top1.append(pred1)
                top2.append(pred2)
                top3.append(pred3)

    recall = recall_score(truth, top1, average='macro')
    precision = precision_score(truth, top1, average='macro')
    macro_f1 = f1_score(truth, top1, average='macro')
    print(f"P: {precision:.3f}, R: {recall:.3f}, F1: {macro_f1:.3f}")

    print(f"Top1: {topk_accuracy(truth, top1):.3f}")
    print(f"Top2: {topk_accuracy(truth, top2):.3f}")
    print(f"Top3: {topk_accuracy(truth, top3):.3f}")