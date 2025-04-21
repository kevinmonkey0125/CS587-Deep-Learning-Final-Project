# # File 3: evaluate_disc.py
# # This script loads the fine-tuned RoBERTa model and computes E_disc scores

import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from bert_score import BERTScorer

# Paths
MODEL_DIR = "/home/scharng/scratch/final_project/roberta-style-clf/final"
TEST_PARALLEL = "/home/scharng/scratch/final_project/processed/test_parallel.csv"
OUTPUT_CSV = "/home/scharng/scratch/final_project/processed/test_energy_scores.csv"

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

# Compute E_BERTScore on CPU to avoid GPU OOM
def compute_E_BERTScore(candidates, references, batch_size=64):
    # Force CPU computation
    scorer = BERTScorer(lang="en", model_type="roberta-base", rescale_with_baseline=True, device="cpu")
    scores = []
    for i in range(0, len(candidates), batch_size):
        batch_cand = candidates[i:i+batch_size]
        batch_ref = references[i:i+batch_size]
        _, _, F1 = scorer.score(batch_cand, batch_ref)
        scores.extend([1 - f.item() for f in F1])
    return scores

# Main scoring function
def evaluate_disc_and_bertscore():
    model, tokenizer, device = load_model_and_tokenizer()
    df_test = pd.read_csv(TEST_PARALLEL)

    # Step 1: Compute E_disc
    results = []
    for _, row in df_test.iterrows():
        sent = row['original']
        E, probs = compute_E_disc(sent, model, tokenizer, device)
        results.append({
            'sentence': sent,
            'p_modern': float(probs[0]),
            'p_shakespeare': float(probs[1]),
            'E_disc': float(E)
        })

    df_scores = pd.DataFrame(results)

    # Step 2: Compute E_BERTScore on CPU
    candidates = df_test["original"].tolist()
    references = df_test["modern"].tolist()
    df_scores["E_BERTScore"] = compute_E_BERTScore(candidates, references)

    # Save and preview
    df_scores.to_csv(OUTPUT_CSV, index=False)
    print(df_scores.head())

# Run evaluation

if __name__ == "__main__":
    evaluate_disc_and_bertscore()







# import torch
# import numpy as np
# import pandas as pd
# from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from bert_score import score as bertscore

# # Paths
# MODEL_DIR = "/home/scharng/scratch/final_project/roberta-style-clf/final"
# TEST_PARALLEL = "/home/scharng/scratch/final_project/processed/test_parallel.csv"
# OUTPUT_CSV = "/home/scharng/scratch/final_project/processed/test_energy_scores.csv"

# # Load fine-tuned model and tokenizer
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
# model.eval()

# # Device setup (CPU/GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Function to compute E_disc for a single sentence
# # E_disc(x) = -log P(style=Shakespearean | x)
# def compute_E_disc(sentence):
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.no_grad():
#         logits = model(**inputs).logits  # shape [1, 2]
#         probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()  # [p_modern, p_shakespeare]
#     p_shakespeare = probs[1]
#     E_disc = -np.log(p_shakespeare + 1e-12)
#     return E_disc, probs

# # Load test parallel data to compute scores for each pair
# df_test = pd.read_csv(TEST_PARALLEL)

# # Step 1: Compute E_disc
# eros = []  # store results
# for idx, row in df_test.iterrows():
#     sent = row['original']  # candidate (Shakespeare-style sentence)
#     E, probs = compute_E_disc(sent)
#     eros.append({
#         'sentence': sent,
#         'p_modern': float(probs[0]),
#         'p_shakespeare': float(probs[1]),
#         'E_disc': float(E)
#     })

# df_scores = pd.DataFrame(eros)

# # Step 2: Compute E_BERTScore = 1 - bert_score(original, modern)
# candidates = df_test["original"].tolist()     # x: generated/converted sentence
# references = df_test["modern"].tolist()       # x': original modern English

# P, R, F1 = bertscore(
#     candidates,
#     references,
#     lang="en",
#     model_type="roberta-base",  # 指定較小的模型
#     rescale_with_baseline=True
# )

# # Save 1 - F1 as E_BERTScore (semantic distance)
# df_scores["E_BERTScore"] = [1 - f.item() for f in F1]


# df_scores.to_csv(OUTPUT_CSV, index=False)
# print(df_scores.head())







