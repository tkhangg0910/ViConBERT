#!/usr/bin/env python3
import argparse
import os
import torch
import random
import pandas as pd
import torch.nn.functional as F
from transformers import PhobertTokenizerFast, XLMRobertaTokenizerFast
from utils.span_extractor import SpanExtractor
from models.viconbert import ViConBERT
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
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--pseudo_sent", action="store_true")
    return parser.parse_args()

def get_vicon():
    import kagglehub
    path = kagglehub.dataset_download("dinhvietahn19021217/vicon400")

    noun_data = pd.read_csv(os.path.join(path,"400_noun_pairs.txt"), sep="\t", names=["Word1","Word2","Relation"], header=0)
    noun_data["POS"] = "N"

    verb_data = pd.read_csv(os.path.join(path,"400_verb_pairs.txt"), sep="\t", names=["Word1","Word2","Relation"], header=0)
    verb_data["POS"] = "V"

    adj_data = pd.read_csv(os.path.join(path,"600_adj_pairs.txt"), sep="\t", names=["Word1","Word2","Relation"], header=0)
    adj_data["POS"] = "A"

    all_data = pd.concat([noun_data, verb_data, adj_data], ignore_index=True)
    # normalize relation labels to uppercase SYN / ANT
    all_data["Relation"] = all_data["Relation"].astype(str).str.strip().str.upper()
    return all_data, noun_data, verb_data, adj_data

def construct_pseudo_sent(word, pos):
    if pos not in frameworks:
        raise ValueError(f"POS {pos} not supported.")
    return random.choice(frameworks[pos]).format(WORD=word)

def get_embedding(model, samples, tokenizer, device):
    """
    samples: {"sentence": List[str], "target_word": List[str]}
    returns: tensor shape [N, dim]
    """
    if len(samples["sentence"]) == 0:
        return torch.empty((0, model.context_projection.output_layer.out_features), device=device)

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
    )

    # remove keys that RobertaModel.forward doesn't accept (if any)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}

    with torch.inference_mode():
        span_tensor = torch.tensor(span_indices, dtype=torch.long, device=device)
        outputs = model(toks, target_span=span_tensor)  # model returns [N, dim]
    return outputs

def evaluate_model_vicon_by_pos(model, tokenizer, df, device, use_pseudo_sent=False):
    pos_tags = ["N", "V", "A"]
    results = []

    for pos_tag in pos_tags:
        df_pos = df[df["POS"] == pos_tag].reset_index(drop=True)
        if df_pos.empty:
            # no pairs for this POS
            results.append({"POS": pos_tag, "AP_SYN": float("nan"), "AP_ANT": float("nan")})
            continue

        # prepare sentences / targets
        words1 = df_pos["Word1"].astype(str).tolist()
        words2 = df_pos["Word2"].astype(str).tolist()
        poses  = df_pos["POS"].astype(str).tolist()
        relations = df_pos["Relation"].astype(str).tolist()

        if use_pseudo_sent:
            word_1_sample = {
                "sentence": [construct_pseudo_sent(w.replace("_"," "), pos_tag) for w in words1],
                "target_word": [w.replace("_"," ") for w in words1]
            }
            word_2_sample = {
                "sentence": [construct_pseudo_sent(w.replace("_"," "), pos_tag) for w in words2],
                "target_word": [w.replace("_"," ") for w in words2]
            }
        else:
            word_1_sample = {
                "sentence": [w.replace("_"," ") for w in words1],
                "target_word": [w.replace("_"," ") for w in words1]
            }
            word_2_sample = {
                "sentence": [w.replace("_"," ") for w in words2],
                "target_word": [w.replace("_"," ") for w in words2]
            }

        # get embeddings once per side
        emb1 = get_embedding(model, word_1_sample, tokenizer, device)  # [N, dim]
        emb2 = get_embedding(model, word_2_sample, tokenizer, device)  # [N, dim]

        if emb1.shape[0] == 0 or emb2.shape[0] == 0:
            results.append({"POS": pos_tag, "AP_SYN": float("nan"), "AP_ANT": float("nan")})
            continue

        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)

        sims = F.cosine_similarity(emb1, emb2, dim=1).cpu().numpy()

        # labels for SYN (1 if SYN else 0)
        labels_syn = [1 if r == "SYN" else 0 for r in relations]
        labels_ant = [1 if r == "ANT" else 0 for r in relations]

        # compute AP for SYN: requires at least one positive and one negative
        if len(set(labels_syn)) < 2:
            ap_syn = float("nan")
        else:
            ap_syn = average_precision_score(labels_syn, sims)

        # compute AP for ANT
        if len(set(labels_ant)) < 2:
            ap_ant = float("nan")
        else:
            ap_ant = average_precision_score(labels_ant, sims)

        results.append({"POS": pos_tag, "AP_SYN": ap_syn, "AP_ANT": ap_ant})

    df_results = pd.DataFrame(results)
    # print in the paper-like layout
    print("\n=== AP (SYN / ANT) by POS ===")
    print(df_results.to_string(index=False, float_format="%.4f"))
    return df_results

if __name__ == "__main__":
    args = setup_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_data, noun_data, verb_data, adj_data = get_vicon()
    if args.model_path=="phobert":
        tokenizer = PhobertTokenizerFast.from_pretrained(args.model_path)
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = ViConBERT.from_pretrained(args.model_path, tokenizer=tokenizer).to(device)
    model.eval()

    evaluate_model_vicon_by_pos(model, tokenizer, all_data, device, use_pseudo_sent=args.pseudo_sent)
