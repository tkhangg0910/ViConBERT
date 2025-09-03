import json
import torch
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, required=True, help="")
parser.add_argument("--output_path", type=str, default="gloss_faiss.index", help="FAISS index output file")
parser.add_argument("--use_gpu", action="store_true", help="Use GPU for indexing")
parser.add_argument("--gloss_enc", type=str, default="dangvantuan/vietnamese-embedding",help="Gloss Encoder model")
args = parser.parse_args()
# 1. Load dataset JSON
with open(args.json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Load model để embedding
model = SentenceTransformer(args.gloss_enc)

# 3. Encode gloss theo batch
gloss_texts = [item["gloss"] for item in data]
gloss_ids = [item["gloss_id"] for item in data]
tok_text =  [tokenize(sentence) for sentence in gloss_texts]
embeddings = model.encode(tok_text)

# 4. Chuyển embeddings về tensor và tạo dict {gloss_id: tensor}
gloss_embeddings = {gid: torch.tensor(emb) for gid, emb in zip(gloss_ids, embeddings)}

# 5. Lưu dict thành file .pt
torch.save(gloss_embeddings, args.output_path)

print("Saved gloss embeddings to gloss_embeddings.pt")
