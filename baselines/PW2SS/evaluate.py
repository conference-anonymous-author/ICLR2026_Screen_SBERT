import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
import torchvision.transforms as transforms
from tqdm import tqdm

from PW2SS import PW2SS
from dataset.page_labels import page2screen_indices_dict, page_indices_dict, screen_idx2page_dict

def embed(model, evaluation_apps):
    transforms_fn = transforms.ToTensor()

    embeddings = {}
    for app in evaluation_apps:
        embeddings[app] = {}

    with torch.no_grad():
        for app in evaluation_apps:
            page2screen_indices = page2screen_indices_dict[app]
            for screen_indices in tqdm(page2screen_indices.values(), desc=f"Embedding {app}"):
                for si in screen_indices:         
                    text_embed = torch.tensor(np.load(f".../dataset/{app}/{si}/screen_ocr_embed.npy")).float().unsqueeze(0).to(device)
                    text_coords = torch.tensor(np.load(f".../dataset/{app}/{si}/screen_ocr_coords.npy")).float().unsqueeze(0).to(device)
                    graphic_class_idx = torch.tensor(np.load(f".../dataset/{app}/{si}/gui_class_idx.npy")).long().unsqueeze(0).to(device)
                    graphic_coords = torch.tensor(np.load(f".../dataset/{app}/{si}/gui_coords.npy")).float().unsqueeze(0).to(device)
                    layout = transforms_fn(np.load(f".../dataset/{app}/{si}/layout_image.npy")).float().unsqueeze(0).to(device)

                    text_mask = torch.ones(1, text_coords.size(1), dtype=torch.bool, device=device)
                    graphic_mask = torch.ones(1, graphic_coords.size(1), dtype=torch.bool, device=device)
        
                    enc = model.encode(
                        text_embed=text_embed, 
                        text_coords=text_coords, 
                        text_mask=text_mask, 
                        
                        graphic_types=graphic_class_idx, 
                        graphic_coords=graphic_coords, 
                        graphic_mask=graphic_mask, 
                        
                        layout=layout
                    ).squeeze(0)

                    embeddings[app][si] = enc.detach().cpu().numpy()

    return embeddings

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
    model = PW2SS(device=device).to(device)
    model.load_state_dict(torch.load(f"./weights/evaluate_X_Temu.pth", weights_only=True))
    model.eval()

    evaluation_apps = ["X", "Temu"]
    embeddings_file = f"./embeddings/evaluate_{evaluation_apps[0]}_{evaluation_apps[1]}.npy"
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