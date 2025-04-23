# This script loads the fine-tuned RoBERTa model and computes E_disc scores

import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from bert_score import BERTScorer

# Paths
MODEL_DIR = "./final"
TEST_PARALLEL = "./processed/test_parallel.csv"
OUTPUT_CSV = "./processed/test_energy_scores.csv"

# Load model, tokenizer, device
def load_model_and_tokenizer():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Compute E_disc(x) = -log P(Shakespearean | x)
def compute_E_disc(sentence, model, tokenizer, device):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    return -np.log(probs[1] + 1e-12), probs

# Compute rescaled-BERTScore on CPU to avoid GPU OOM (Not energy)
def rescaled_BERTScore(candidates, references, batch_size=64):
    # Initialize BERTScorer with rescaling enabled
    scorer = BERTScorer(
        lang="en",
        model_type="roberta-base",
        rescale_with_baseline=True,  # Enable rescaling with baseline
        baseline_path=None,  # Will use default baseline
        device="cpu"  # Force CPU computation to avoid OOM
    )
    
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch_cand = candidates[i:i+batch_size]
        batch_ref = references[i:i+batch_size]
        
        # Get P, R, F1 scores (using F1 as recommended)
        P, R, F1 = scorer.score(batch_cand, batch_ref)
        batch_scores = [f.item() for f in F1]
        scores.extend(batch_scores)
    
    return scores