import json
import os
import csv
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from underthesea.pipeline.text_normalize import token_normalize
from utils.custom_uts_tokenize import tokenize
import torch
from tqdm import tqdm


def split_contrastive_stage1_data(pseudo_sent_path, word_synsets_path, output_dir):
    word_synsets = pd.read_csv(word_synsets_path)
    with open(pseudo_sent_path, encoding='utf-8') as f:
        data = json.load(f)
        
    train_data = defaultdict(list)
    valid_data = defaultdict(list)

    for word_id, sentences in data.items():
        synset_row  = word_synsets[word_synsets["word_id"] == int(word_id)]
        synset_id = int(synset_row["synset_id"].values[0])
        train_sents, valid_sents = train_test_split(sentences, test_size=0.2, random_state=42)
        target_word = synset_row["word"].values[0]  
        supersense = synset_row["pos"].values[0]
        train_data[synset_id].extend([(word_id,target_word,supersense, sent) for sent in train_sents])
        valid_data[synset_id].extend([(word_id,target_word,supersense, sent) for sent in valid_sents])

    final_train = []
    final_valid = []
    for synset_id in train_data.keys():
        min_samples = min(len(train_data[synset_id]), len(valid_data[synset_id]))
        final_train.extend(train_data[synset_id][:min_samples])
        final_valid.extend(valid_data[synset_id][:min_samples])

    # Lưu dưới dạng danh sách các cặp (word_id, sentence)
    with open(os.path.join(output_dir, "train_data_v3.json"), 'w', encoding='utf-8') as f:
        json.dump([{"word_id": w, "sentence": s,"target_word": target,"supersense":supersense, "synset_id": synset_id} 
                for synset_id, group in train_data.items() 
                for w,target,supersense, s in group], f,ensure_ascii=False,indent=2)
        
    with open(os.path.join(output_dir, "valid_data_v3.json"), 'w', encoding='utf-8') as f:
        json.dump([{"word_id": w, "sentence": s,"target_word": target,"supersense":supersense, "synset_id": synset_id} 
                for synset_id, group in valid_data.items() 
                for w,target,supersense, s in group], f, ensure_ascii=False,indent=2)
        
def text_normalize(text, tokenizer='underthesea'):
      """

      Args:
          tokenizer (str): space or underthesea
      """

      if tokenizer == 'underthesea':
          tokens = tokenize(text, fixed_words=["3D","t\"rưng", "kế hoạch 34A","Kế hoạch 34A"])
      else:
          tokens = text.split(" ")
      normalized_tokens = [token_normalize(token) for token in tokens]
      normalized_text = " ".join(normalized_tokens)
      return normalized_text


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
    
import torch
import faiss
import argparse
import os
import numpy as np

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


# if __name__ == "__main__":
#     split_contrastive_stage1_data(
#         "data/raw/stage_1_pseudo_sents/pseudo_sents_v1.json",
#         "data/raw/stage_1_pseudo_sents/word_synsets_with_pos_with_gloss_v2.csv",
#         "data/processed/stage1_pseudo_sents"
#     )
# from sentence_transformers import SentenceTransformer

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--gloss_path", type=str, required=True, help="Path to gloss_embeddings.pt")
#     parser.add_argument("--index_out", type=str, default="gloss_faiss.index", help="FAISS index output file")
#     parser.add_argument("--id_out", type=str, default="synset_ids.pt", help="Synset ID output file")
#     parser.add_argument("--use_gpu", action="store_true", help="Use GPU for indexing")
#     gloss_enc = SentenceTransformer('dangvantuan/vietnamese-embedding') 
#     args = parser.parse_args()
#     precompute_and_save(
#         "data/raw/stage_1_pseudo_sents/word_synsets_with_pos_with_gloss_v2.csv",
#         args.gloss_path,
#         gloss_enc
#         )
#     embd, synset_ids = load_gloss_embeddings(args.gloss_path)
#     index = index_embeddings(embd, use_gpu=args.use_gpu)
#     save_index(index, args.index_out)
#     save_synset_ids(synset_ids, args.id_out)
