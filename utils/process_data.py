import json
import os
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

def split_contrastive_stage1_data(pseudo_sent_path, word_synsets_path, output_dir):
    word_synsets = pd.read_csv(word_synsets_path)
    with open(pseudo_sent_path, encoding='utf-8') as f:
        data = json.load(f)
        
    train_data = defaultdict(list)
    valid_data = defaultdict(list)

    for word_id, sentences in data.items():
        synset_id = word_synsets[word_synsets["word_id"] == int(word_id)]["synset_id"].values[0]
        train_sents, valid_sents = train_test_split(sentences, test_size=0.2, random_state=42)
        train_data[synset_id].extend([(word_id, sent) for sent in train_sents])
        valid_data[synset_id].extend([(word_id, sent) for sent in valid_sents])

    final_train = []
    final_valid = []
    for synset_id in train_data.keys():
        min_samples = min(len(train_data[synset_id]), len(valid_data[synset_id]))
        final_train.extend(train_data[synset_id][:min_samples])
        final_valid.extend(valid_data[synset_id][:min_samples])

    # Lưu dưới dạng danh sách các cặp (word_id, sentence)
    with open(os.path.join(output_dir,"train_data.json"), "w") as f:
        json.dump([{"word_id": w, "sentence": s, "synset_id": synset_id} 
                for synset_id, group in train_data.items() 
                for w, s in group], f)
        
    with open(os.path.join(output_dir,"valid_data.json"), "w") as f:
        json.dump([{"word_id": w, "sentence": s, "synset_id": synset_id} 
                for synset_id, group in valid_data.items() 
                for w, s in group], f)

    
if __name__=="__main__":
    pseudo_sent_path="data/raw/stage_1_pseudo_sents/pseudo_sent.json"
    word_synsets_path = "data/raw/stage_1_pseudo_sents/word_synsets_with_pos_with_gloss.csv"
    output_dir = "data/processed/stage1_pseudo_sents"
    # split_contrastive_stage1_data(pseudo_sent_path, word_synsets_path,output_dir)
    with open(pseudo_sent_path, encoding='utf-8') as f:
        data = json.load(f)
    total={}
    for key, value in data.items():
        if len(value) < 10:
            total[key]=10-len(value)
            
    print(total)
    print(f"total: {len(total)}")