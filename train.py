import math
import random
import torch
import pandas as pd
from transformers import pipeline, T5Tokenizer, EncoderDecoderCache
from Evaluate_Disc_BertScore import load_model_and_tokenizer, compute_E_disc, rescaled_BERTScore
from typing import List, Tuple
import json
from datetime import datetime
from transformers import logging
from tqdm import tqdm
import nltk  # Add at the top of the file
nltk.download('punkt')  # Download tokenizer data
logging.set_verbosity_error()

# Configuration 
ALPHA_DISC = 0.5    # weight of E_disc
ALPHA_BERT = 0.5    # weight of E_BERTScore
N_ITER = 25         # MH iterations per chain
S = 5  # Numebr of seed sentences from test set
NUM_CHAINS = 3  # Number of parallel chains per seed
NUM_EXPERIMENTS = 5  # Number of experiments to run
SEED = 42  # Random seed for reproducibility

random.seed(SEED)
torch.manual_seed(SEED)

def energy(x, seed):
    # E_disc
    E_disc, _ = compute_E_disc(x, model, tokenizer, device)
    # E_BERTScore
    E_bert = -rescaled_BERTScore([x], [seed])[0]
    return ALPHA_DISC * E_disc + ALPHA_BERT * E_bert


def compute_inidvidual_Jscore(output_sentence, seed_text):
    # p(Shakespeare | output_sentence)
    _, probs = compute_E_disc(output_sentence, model, tokenizer, device)
    ACC = 1 if probs[1] > probs[0] else 0
    print(f"ACC: {ACC}")
    
    # Similarity score
    SIM = rescaled_BERTScore([output_sentence], [seed_text])[0]
    print(f"SIM: {SIM}")
    
    # Formality score
    formality_result = formality_classifier(output_sentence)[0]
    
    FL = 1 if formality_result['label'] == 'LABEL_1' else 0
    print(f"FL: {FL}")
    
    # Compute individual score
    return ACC * SIM * FL

def get_lowest_energy_sample(seed_chains):
    all_samples = []
    for chain in seed_chains:
        all_samples.extend(chain)  
            
    # Sort by energy (second element of tuple) and get the lowest
    if all_samples:
        best_sample = min(all_samples, key=lambda x: x[1])
        return best_sample[0]  # Return the sentence only
    return None

# Load test parallel data and randomly sample S seed texts
test_df = pd.read_csv("./processed/test_parallel.csv")
seed_texts = random.sample(test_df['modern'].tolist(), S)
for i, seed_text in enumerate(seed_texts):
    print(f"Seed {i}: {seed_text}")

# Create experiment logger
experiment_results = {
    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    'experiments': []
}


# Instntiate models
# Style discriminator and BERTScore energy functions
model, tokenizer, device = load_model_and_tokenizer()
# Proposal model
generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    device="cuda" if device.type == "cuda" else -1,
    do_sample=True,
    top_k=50,
)
# Formality classifier
formality_classifier = pipeline(
    "text-classification",
    model="cointegrated/roberta-base-formality",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# Run experiments
# Jscores over experiments
Jscores_summary = []
Jscores_block = []

for exp_id in tqdm(range(NUM_EXPERIMENTS), desc="Experiments", position=0):
    SEED += 1
    random.seed(SEED)
    experiment_data = {
        'experiment_id': exp_id,
        'seeds': []
    }
    Jscore_summary_num = 0
    Jscore_block_num = 0

    for seed_idx, seed_text in tqdm(enumerate(seed_texts), desc=f"Seeds (Exp {exp_id})", position=1, leave=False, total=len(seed_texts)):
        seed_data = {
            'seed_id': seed_idx,
            'seed_text': seed_text,
            'chains': [],
            'blockMH': None,
            'summary': None,
            'scores': {}
        }
        seed_chains: List[List[Tuple[str, float]]] = []
        
        # Run multiple chains for each seed
        for chain_id in tqdm(range(NUM_CHAINS), desc=f"Chains (Seed {seed_idx})", position=2, leave=False):
            current = seed_text
            chain = [(current, energy(current, seed_text))]  # Add initial state to chain

            for t in tqdm(range(1, N_ITER + 1), desc=f"Iterations (Chain {chain_id})", position=3, leave=False):
                E_current = energy(current, seed_text)
                
                # Propose a new sentence
                ### May have to modify prompt.
                prompt = f'Rewrite this sentence in the style of Shakespeare: "{current}"'
                ### Could consider different temperatures later
                ### proposal may add unnecessary words: change propmt????
                out = generator(
                    prompt, 
                    max_length=128, 
                    num_return_sequences=1
                )[0]["generated_text"]
                proposal = out.strip()

                # Compute elements for MH
                E_prop = energy(proposal, seed_text)

                # Compute forward probability (probability of generating proposal from current)
                forward_inputs = t5_tokenizer(
                    f"Rewrite this sentence in the style of Shakespeare: \"{current}\"", 
                    return_tensors="pt"
                ).to(device)
                proposal_ids = t5_tokenizer(proposal, return_tensors="pt").input_ids.to(device)

                # Get log probabilities of the proposal sequence
                with torch.no_grad():
                    forward_outputs = generator.model(
                        **forward_inputs, 
                        labels=proposal_ids,
                        past_key_values=EncoderDecoderCache() if hasattr(generator.model, 'past_key_values') else None
                    )
                    q_prop = math.exp(-forward_outputs.loss.item()) + 1e-10

                # Compute backward probability (probability of generating current from current)
                backward_inputs = t5_tokenizer(
                    f"Rewrite this Shakespeare text in modern English: \"{current}\"", 
                    return_tensors="pt"
                ).to(device)
                current_ids = t5_tokenizer(current, return_tensors="pt").input_ids.to(device)

                # Get log probabilities of the current sequence
                with torch.no_grad():
                    backward_outputs = generator.model(
                        **backward_inputs, 
                        labels=current_ids,
                        past_key_values=EncoderDecoderCache() if hasattr(generator.model, 'past_key_values') else None
                    )
                    q_current = math.exp(-backward_outputs.loss.item()) + 1e-10

                # Accept-Reject step
                alpha = min(1.0, math.exp(-(E_prop - E_current))*q_current/q_prop)

                if random.random() < alpha:
                    current = proposal
                    E_current = E_prop
                    chain.append((current, E_current))

        seed_chains.append(chain)
        seed_data['chains'].append({
            'chain_id': chain_id,
            'samples': [(text, float(energy)) for text, energy in chain]
        })

        # After storing chains for the seed
        blockMH = get_lowest_energy_sample(seed_chains)
        if blockMH is None:
            blockMH = seed_text  # fallback to seed text if no samples available

        # Summarize this seed by BART
        # Initialize BART summarizer
        summarizer = pipeline(
            task="summarization",
            model="facebook/bart-large-cnn",
            device=0 if device.type == "cuda" else -1,
            max_length=50,
            min_length=25,
            do_sample=False
        )
        
        # Collect all chains and summarize
        all_chains = []
        for chain in seed_chains:
            for sentence, _ in chain:
                all_chains.append(sentence)

        # Remove duplicates and join
        all_chains = list(set(all_chains))
        joined_text = " ".join(all_chains)

        # Get first sentence using BART
        full_summary = summarizer(joined_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        summary = nltk.sent_tokenize(full_summary)
    
        print(f"Summary: {summary}")

        # After processing each seed
        torch.cuda.empty_cache()  # Add after summarizer usage
        
        # Compute Jscore for this summary/BlockMH
        Jscore_summary_num += compute_inidvidual_Jscore(summary, seed_text)
        Jscore_block_num += compute_inidvidual_Jscore(blockMH, seed_text)

        seed_data['blockMH'] = {
            'text': blockMH,
            'energy': float(energy(blockMH, seed_text))
        }
        
        seed_data['summary'] = {
            'text': summary,
            'energy': float(energy(summary, seed_text))
        }
        
        # Add scores
        seed_data['scores'] = {
            'jscore_summary': float(compute_inidvidual_Jscore(summary, seed_text)),
            'jscore_blockMH': float(compute_inidvidual_Jscore(blockMH, seed_text))
        }
        
        experiment_data['seeds'].append(seed_data)

    Jscore_summary = Jscore_summary_num/S
    Jscore_block = Jscore_block_num/S

    print(f"Jscore Summary (one expt): {Jscore_summary}")
    print(f"Jscore Block (one expt): {Jscore_block}")

    Jscores_summary.append(Jscore_summary)
    Jscores_block.append(Jscore_block)
    experiment_results['experiments'].append(experiment_data)

print("Done with all experiments!")
print(f"Jscore Summary: {Jscores_summary}")
print(f"Jscore Block: {Jscores_block}")

# Save results
output_file = f"experiment_results_{experiment_results['timestamp']}.json"
with open(output_file, 'w') as f:
    json.dump(experiment_results, f, indent=2)










