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
        train_sents, valid_sents = train_test_split(sentences, test_size=0.1, random_state=42)
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
    with open(os.path.join(output_dir, "train_data.json"), 'w', encoding='utf-8') as f:
        json.dump([{"word_id": w, "sentence": s,"target_word": target,"supersense":supersense, "synset_id": synset_id} 
                for synset_id, group in train_data.items() 
                for w,target,supersense, s in group], f,ensure_ascii=False,indent=2)
        
    with open(os.path.join(output_dir, "valid_data.json"), 'w', encoding='utf-8') as f:
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



if __name__ == "__main__":
    split_contrastive_stage1_data(
        "data/raw/stage_1_pseudo_sents/poly_sents.json",
        "data/raw/stage_1_pseudo_sents/poly_only.csv",
        "data/processed/polybert_exp"
    )