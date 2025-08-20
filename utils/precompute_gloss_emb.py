import csv
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from underthesea.pipeline.text_normalize import token_normalize
from utils.custom_uts_tokenize import tokenize
import torch
from tqdm import tqdm
import argparse
import faiss
import os
import numpy as np
from sentence_transformers import SentenceTransformer


def load_gloss_embeddings(path: str):
    print(f"🔹 Loading gloss embeddings from: {path}")
    data = torch.load(path, map_location='cpu')  # dict: synset_id -> tensor
    synset_ids = list(data.keys())
    embeddings = torch.stack([data[sid] for sid in synset_ids])
    return embeddings, synset_ids

def index_embeddings(embeddings: torch.Tensor, use_gpu=False):
    """
    Input: embeddings [N, D], torch.Tensor
    Output: FAISS index
    """
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    dim = embeddings_np.shape[1]

    if use_gpu:
        print("🚀 Using GPU FAISS index...")
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(dim)
        index = faiss.GpuIndexFlatIP(res, index_flat)
    else:
        print("🧠 Using CPU FAISS index...")
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings_np)
    print(f"✅ Indexed {index.ntotal} vectors of dim {dim}")
    return index

def save_index(index, path: str):
    faiss.write_index(index, path)
    print(f"💾 Saved FAISS index to: {path}")

def save_synset_ids(synset_ids, path: str):
    torch.save(synset_ids, path)
    print(f"💾 Saved synset_id mapping to: {path}")
    
    
def precompute_and_save(gloss_dict_path, save_path, gloss_encoder):
    """
    gloss_dict_path: đường dẫn tới file JSON hoặc CSV chứa mapping synset_id→gloss text
    save_path: file .pt sẽ lưu dict {synset_id: tensor_embedding}
    gloss_encoder: object có method .encode(text) → numpy array
    """
    gloss_dict = {}
    with open(gloss_dict_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            syn_id = row['synset_id']
            gloss = row['gloss']
            if syn_id not in gloss_dict:
                gloss_dict[int(syn_id)] = gloss


    embeddings = {}
    from pyvi.ViTokenizer import tokenize as pvi_tokenize
    for syn_id, gloss in tqdm(gloss_dict.items(), desc="Precomputing gloss embs", ascii=True):
        vec = gloss_encoder.encode(pvi_tokenize(gloss))          
        embeddings[syn_id] = torch.tensor(vec)      

    torch.save(embeddings, save_path)
    print(f"Saved {len(embeddings)} gloss embeddings to {save_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gloss_path", type=str, required=True, help="Path to gloss_embeddings.pt")
    parser.add_argument("--index_out", type=str, default="gloss_faiss.index", help="FAISS index output file")
    parser.add_argument("--id_out", type=str, default="synset_ids.pt", help="Synset ID output file")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for indexing")
    parser.add_argument("--gloss_enc", type=str, default="dangvantuan/vietnamese-embedding",help="Gloss Encoder model")
    args = parser.parse_args()

    gloss_enc = SentenceTransformer(args.gloss_enc) 
    precompute_and_save(
        "data/raw/stage_1_pseudo_sents/poly_only.csv",
        args.gloss_path,
        gloss_enc
        )
    embd, synset_ids = load_gloss_embeddings(args.gloss_path)
    index = index_embeddings(embd, use_gpu=args.use_gpu)
    save_index(index, args.index_out)
    save_synset_ids(synset_ids, args.id_out)