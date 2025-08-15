import argparse
import json
import os
import torch
from transformers import PhobertTokenizerFast, XLMRobertaTokenizerFast
from utils.load_config import load_config
from models.base_model import ViSynoSenseEmbedding
import torch
from tqdm import tqdm
from transformers.utils import is_torch_available
from torch.amp import autocast
from utils.span_extractor import SpanExtractor
from scipy.stats import spearmanr
import torch.nn.functional as F
import random
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

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--pseudo_sent", action='store_true', help="Use pseudo sentences")
    args = parser.parse_args()
    return args 

def get_visim400():
    import kagglehub
    import pandas as pd
    # Download latest version
    path = kagglehub.dataset_download("dinhvietahn19021217/visim400csv")

    return pd.read_csv(os.path.join(path,"visim-40401.csv"))

def get_embedding(model,samples,tokenizer, device):
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
            max_length=256,
            return_attention_mask=True,
            return_offsets_mapping=True
    ).to(device)
    
    with torch.inference_mode():
        outputs = model(**toks,target_span=torch.tensor(span_indices, dtype=torch.long) )
    return outputs

def construct_pseudo_sent(word ,pos): 
    if pos not in frameworks: 
        raise ValueError(f"POS {pos} have not had template yet.") 
    sentence = random.choice(frameworks[pos]).format(WORD=word) 
    return sentence

def evaluate_model(model, tokenizer, data, device, use_pseudo_sent=False):
    """Enhanced evaluation with detailed metrics"""
    words1 = data["Word1"].tolist()
    words2 = data["Word2"].tolist()
    sim_gt = data["Sim2"].tolist()  
    pos = data["POS"].tolist()
    if use_pseudo_sent:
        word_1_sample = {"sentence": [construct_pseudo_sent(w,p) for w,p in zip(words1,pos)], "target_word": words1}
        word_2_sample = {"sentence": [construct_pseudo_sent(w,p) for w,p in zip(words2,pos)], "target_word": words2}
    else:
        word_1_sample = {"sentence": words1, "target_word":words1}
        word_2_sample = {"sentence": words2, "target_word":words2}
    
    word_1_embd = get_embedding(model,word_1_sample,tokenizer,device)
    word_2_embd = get_embedding(model,word_2_sample,tokenizer,device)
    sims = F.cosine_similarity(word_1_embd, word_2_embd, dim=1)

    correlation, _ = spearmanr(sims.cpu().numpy(), sim_gt)
    print("Spearman correlation:", correlation)
    return correlation

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(42) 
    args = setup_args()
    
    benchmark = get_visim400()
    
    tokenizer = PhobertTokenizerFast.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    
    model = ViSynoSenseEmbedding.from_pretrained(args.model_path).to(device)


    valid_metrics = evaluate_model(model, tokenizer,benchmark, device)
