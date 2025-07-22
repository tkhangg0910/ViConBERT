import json
import pandas as pd
import csv
import random
import numpy as np
import torch
from tqdm import tqdm
import faiss
import argparse
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize
from models.base_model import ViSynoSenseEmbedding
from transformers import PhobertTokenizerFast

def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--csv_file_path", type=str)
    parser.add_argument("--json_file_path", type=str)
    parser.add_argument("--vector_aggerate", type=str, default="one")
    args = parser.parse_args()
    return args 

class Pipeline:
    def __init__(self, tokenizer, span_ex, model) -> None:
        self.tokenizer=tokenizer
        self.span_ex=span_ex
        self.model=model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def extract(self, query, target):
        query_norm=text_normalize(query)
        tokenized_query = self.tokenizer(query_norm,return_tensors="pt").to(self.device)
        span_idx = self.span_ex.get_span_indices(query_norm, target)
        span =torch.Tensor(span_idx).unsqueeze(0).to(self.device)
        self.model.eval()
        query_vec = self.model(tokenized_query, span)
        return query_vec

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = setup_args()
    csv_file_path=args.csv_file_path
    metadata_list = []
    json_file_path= args.json_file_path
    model_path = args.model_path
    dimension = 768
    index = faiss.IndexFlatL2(dimension)
    word_ids=[]
    df = pd.read_csv(csv_file_path)
    with open(json_file_path, 'r', encoding='utf-8') as jf:
        sentence_dict = json.load(jf)
    tokenizer = PhobertTokenizerFast.from_pretrained(model_path)
    model = ViSynoSenseEmbedding.from_pretrained(model_path,tokenizer).to(device)
    span_ex =SpanExtractor(tokenizer)
    pipeline=Pipeline(tokenizer,span_ex, model )
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing", ascii=True):
        word_id = str(row['word_id']) 
        word = row['word']
        gloss = row['gloss']
        pos = row['pos']
        
        if word_id in sentence_dict:
            sentences = sentence_dict[word_id]
            if sentences:
                if args.vector_aggerate =="one":
                    chosen_sentence = random.choice(sentences)
                    vec = pipeline.extract(chosen_sentence, word).cpu().detach().numpy()
                vec = np.array(vec).astype('float32').reshape(1, -1)

                index.add(vec)
                word_ids.append(word_id)  
                metadata_list.append({
                    "word_id": word_id,
                    "pos":pos,
                    "word": word,
                    "gloss": gloss,
                })

            else:
                print(f"{word} ({word_id}): [Không có câu nào]")
        else:
            print(f"{word} ({word_id}): [Không tìm thấy trong JSON]")

    faiss.write_index(index, 'index.faiss')
    with open('metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)

    # ✅ Lưu word_ids mapping nếu cần
    with open('word_ids.json', 'w', encoding='utf-8') as f:
        json.dump(word_ids, f, ensure_ascii=False, indent=2)