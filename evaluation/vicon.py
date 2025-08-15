import argparse
import os
import torch
import random
import pandas as pd
import torch.nn.functional as F
from transformers import PhobertTokenizerFast
from utils.span_extractor import SpanExtractor
from models.base_model import ViSynoSenseEmbedding
from tqdm import tqdm
from sklearn.metrics import average_precision_score

frameworks = {
    'N': [
        "{WORD} là một vật thể.",
        "Tôi nhìn thấy {WORD} hôm nay.",
        "Có nhiều {WORD} trong phòng."
    ],
    'V': [
        "Tôi muốn {WORD} vào buổi sáng.",
        "Cô ấy thường {WORD} mỗi ngày.",
        "Họ đã {WORD} trong tuần qua."
    ],
    'A': [
        "Cô ấy rất {WORD}.",
        "Tôi thấy món ăn này thật {WORD}.",
        "Một ngày thật {WORD}."
    ]
}

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pseudo_sent", action="store_true")
    return parser.parse_args()

def get_vicon():
    import kagglehub
    import pandas as pd
    path = kagglehub.dataset_download("dinhvietahn19021217/vicon400")
    
    noun_data = pd.read_csv(os.path.join(path,"400_noun_pairs.txt"), sep="\t")
    noun_data["POS"] = "N"

    verb_data = pd.read_csv(os.path.join(path,"400_verb_pairs.txt"), sep="\t")
    verb_data["POS"] = "V"

    adj_data = pd.read_csv(os.path.join(path,"600_adj_pairs.txt"), sep="\t")
    adj_data["POS"] = "A"

    # Gộp lại nếu muốn evaluate chung
    all_data = pd.concat([noun_data, verb_data, adj_data], ignore_index=True)
    return all_data, noun_data, verb_data, adj_data

def construct_pseudo_sent(word, pos):
    if pos not in frameworks:
        raise ValueError(f"POS {pos} not supported.")
    return random.choice(frameworks[pos]).format(WORD=word)

def get_embedding(model, samples, tokenizer, device):
    span_extractor = SpanExtractor(tokenizer)
    sentences = samples["sentence"]
    target_words = samples["target_word"]

    span_indices = [span_extractor.get_span_indices(sent, word)
                    for sent, word in zip(sentences, target_words)]

    toks = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.inference_mode():
        outputs = model(toks, target_span=torch.tensor(span_indices, dtype=torch.long).to(device))
    return outputs

def evaluate_model_vicon_by_pos(model, tokenizer, df, device, use_pseudo_sent=False):
    pos_tags = ["N", "V", "A"]
    results = []
    print(df.head(5))
    for pos_tag in pos_tags:
        df_pos = df[df["POS"] == pos_tag]
        if df_pos.empty:
            continue

        ap_all = _evaluate_ap(model, tokenizer, df_pos, device, use_pseudo_sent)

        ap_syn = _evaluate_ap(model, tokenizer, df_pos[df_pos["Relation"].str.lower() == "synonym"], device, use_pseudo_sent)
        ap_ant = _evaluate_ap(model, tokenizer, df_pos[df_pos["Relation"].str.lower() == "antonym"], device, use_pseudo_sent)

        results.append({
            "POS": pos_tag,
            "AP_all": ap_all,
            "AP_syn": ap_syn,
            "AP_ant": ap_ant
        })

    df_results = pd.DataFrame(results)
    print("\n=== AP by POS ===")
    print(df_results.to_string(index=False, float_format="%.4f"))
    return df_results


def _evaluate_ap(model, tokenizer, df, device, use_pseudo_sent):
    words1 = df["Word1"].tolist()
    words2 = df["Word2"].tolist()
    pos     = df["POS"].tolist()
    label_str = df["Relation"].tolist()

    labels = [1 if rel.lower() == "synonym" else 0 for rel in label_str]

    if use_pseudo_sent:
        word_1_sample = {
            "sentence": [construct_pseudo_sent(w.replace("_", " "), p) for w, p in zip(words1, pos)],
            "target_word": [w.replace("_", " ") for w in words1]
        }
        word_2_sample = {
            "sentence": [construct_pseudo_sent(w.replace("_", " "), p) for w, p in zip(words2, pos)],
            "target_word": [w.replace("_", " ") for w in words2]
        }
    else:
        word_1_sample = {
            "sentence": [w.replace("_", " ") for w in words1],
            "target_word": [w.replace("_", " ") for w in words1]
        }
        word_2_sample = {
            "sentence": [w.replace("_", " ") for w in words2],
            "target_word": [w.replace("_", " ") for w in words2]
        }

    word_1_embd = get_embedding(model, word_1_sample, tokenizer, device)
    word_2_embd = get_embedding(model, word_2_sample, tokenizer, device)

    word_1_embd = F.normalize(word_1_embd, p=2, dim=1)
    word_2_embd = F.normalize(word_2_embd, p=2, dim=1)

    sims = F.cosine_similarity(word_1_embd, word_2_embd, dim=1).cpu().numpy()

    if len(set(labels)) < 2:
        return float('nan')  # Không tính được AP nếu toàn 1 hoặc toàn 0

    ap = average_precision_score(labels, sims)
    return ap


if __name__ == "__main__":
    args = setup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_data, noun_data, verb_data, adj_data = get_vicon()

    tokenizer = PhobertTokenizerFast.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = ViSynoSenseEmbedding.from_pretrained(args.model_path).to(device)

    evaluate_model_vicon_by_pos(model, tokenizer, all_data, device, use_pseudo_sent=args.pseudo_sent)
