import math
import random
from transformers import pipeline
from Evaluate_Disc_BertScore import load_model_and_tokenizer, compute_E_disc, compute_E_BERTScore

# --- Configuration ---
ALPHA_DISC = 1.0    # weight of E_disc
ALPHA_BERT = 1.0    # weight of E_BERTScore
N_ITER = 25         # MH iterations per chain
SEED_TEXT = "How are you today?"

# --- Initialize models ---
# 1) Style discriminator and BERTScore energy functions
model, tokenizer, device = load_model_and_tokenizer()
# 2) Weak Flan-T5-XXL proposal model (這裡用 flan-t5-small 作示範)
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    device="cuda" if device.type == "cuda" else -1,
    do_sample=True,
    top_k=50,
)

# --- Define target energy EC(x) = alpha1*E_disc + alpha2*E_bertscore(x, seed) ---
def energy(x, seed):
    # E_disc
    E_disc, _ = compute_E_disc(x, model, tokenizer, device)
    # E_BERTScore
    E_bert = compute_E_BERTScore([x], [seed])[0]
    return ALPHA_DISC * E_disc + ALPHA_BERT * E_bert

# --- Metropolis-Hastings sampling ---
chain = []
current = SEED_TEXT
E_current = energy(current, SEED_TEXT)
chain.append((current, E_current))

for t in range(1, N_ITER + 1):
    # 1) Proposal: paraphrase current via LLM
    prompt = f"Rewrite this sentence in the style of William Shakespeare: \"{current}\""
    out = generator(prompt, max_length=128, num_return_sequences=1)[0]["generated_text"]
    proposal = out.strip()

    # 2) Compute energy of proposal
    E_prop = energy(proposal, SEED_TEXT)

    # 3) Acceptance probability (對稱 q 分佈簡化)
    alpha = min(1.0, math.exp(-(E_prop - E_current)))

    # 4) Decide accept/reject
    if random.random() < alpha:
        current = proposal
        E_current = E_prop
    chain.append((current, E_current))

# --- 結果輸出 ---
print("MH chain (x, E):")
for i, (x, e) in enumerate(chain):
    print(f"{i:>2d}: E={e:.4f}\n   {x}\n")
