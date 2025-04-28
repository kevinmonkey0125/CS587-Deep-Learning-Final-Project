import math
import random
import torch
import pandas as pd
import numpy as np
from transformers import pipeline, T5Tokenizer, EncoderDecoderCache
from Evaluate_Disc_BertScore import load_model_and_tokenizer, compute_E_disc, rescaled_BERTScore
from utlis import energy, compute_inidvidual_Jscore, get_lowest_energy_sample
from typing import List, Tuple
import json
from datetime import datetime
from transformers import logging
from tqdm import tqdm
import nltk  # Add at the top of the file
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
logging.set_verbosity_error()

# Configuration 
ALPHA_DISC = 0.5    # weight of E_disc
ALPHA_BERT = 0.5    # weight of E_BERTScore
N_ITER = 25         # MH iterations per chain
S = 20  # Numebr of seed sentences from test set
NUM_CHAINS = 10  # Number of parallel chains per seed
NUM_EXPERIMENTS = 30  # Number of experiments to run
SEED = 42  # Random seed for reproducibility

random.seed(SEED)
torch.manual_seed(SEED)

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
    # model="google/flan-t5-small",
    model="google/flan-t5-base",
    device="cuda" if device.type == "cuda" else -1,
    do_sample=True,
    top_k=50,
)
generator.model.to(device)

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
            # chain = [(current, energy(current, seed_text))]  # Add initial state to chain
            chain = [(current, energy(current, seed_text, model, tokenizer, device))]

            for t in tqdm(range(1, N_ITER + 1), desc=f"Iterations (Chain {chain_id})", position=3, leave=False):
                E_current = energy(current, seed_text, model, tokenizer, device)
                print(f"Current: {current}")
                
                # Propose a new sentence
                ### May have to modify prompt.
                prompt = f'Rewrite this sentence in the style of Shakespeare: "{current}"'
                # prompt = f'Paraphrase the following sentence using early modern English without mentioning Shakespeare or writing about styles:\n"{current}"'
                # prompt = (
                # "Paraphrase the following sentence in the style of William Shakespeare, "
                # "using archaic pronouns (thou, thee, thy), inverted syntax, and Elizabethan diction, "
                # "while preserving the exact meaning and punctuation. "
                # "Do not include any direct quotations from Shakespeare's works or mention his name.\n\n"
                # f"Sentence: \"{current}\""
                # )

                ### Could consider different temperatures later
                ### proposal may add unnecessary words: change propmt????
                out = generator(
                    prompt, 
                    max_length=128, 
                    num_return_sequences=1
                )[0]["generated_text"]
                proposal = out.strip()
                print(f"Proposal: {proposal}")

                # Compute elements for MH
                E_prop = energy(proposal, seed_text, model, tokenizer, device)

                # Compute forward probability (probability of generating proposal from current)
                forward_inputs = t5_tokenizer(
                    f"Rewrite this sentence in the style of Shakespeare: \"{current}\"", 
                    return_tensors="pt"
                ).to(device)
                # forward_inputs = t5_tokenizer(
                # "Paraphrase the following sentence in the style of William Shakespeare, "
                # "using archaic pronouns (thou, thee, thy), inverted syntax, and Elizabethan diction, "
                # "while preserving the exact meaning and punctuation. "
                # "Do not include any direct quotations from Shakespeare's works or mention his name.\n\n"
                # f"Sentence: \"{current}\"",
                # return_tensors="pt"
                # ).to(device)



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
                # backward_inputs = t5_tokenizer(
                # "Paraphrase the following sentence in the style of William Shakespeare, "
                # "using archaic pronouns (thou, thee, thy), inverted syntax, and Elizabethan diction, "
                # "while preserving the exact meaning and punctuation. "
                # "Do not include any direct quotations from Shakespeare's works or mention his name.\n\n"
                # f"Sentence: \"{current}\"",
                # return_tensors="pt"
                # ).to(device)

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
        summary = nltk.sent_tokenize(full_summary)[0]
        print(f"Summary: {summary}")
        print(f"BlockMH: {blockMH}")

        # After processing each seed
        torch.cuda.empty_cache()  # Add after summarizer usage
        
        # Compute Jscore for this summary/BlockMH
        Jscore_summary_num += compute_inidvidual_Jscore(summary, seed_text, model, tokenizer, device, formality_classifier)
        Jscore_block_num += compute_inidvidual_Jscore(blockMH, seed_text, model, tokenizer, device, formality_classifier)

        seed_data['blockMH'] = {
            'text': blockMH,
            'energy': float(energy(blockMH, seed_text, model, tokenizer, device))
        }
        
        seed_data['summary'] = {
            'text': summary,
            # 'energy': float(energy(summary, seed_text, model, tokenizer, device))
            'energy': float(energy(summary, seed_text, model, tokenizer, device))

        }
        
        # Add scores
        seed_data['scores'] = {
            'jscore_summary': float(compute_inidvidual_Jscore(summary, seed_text, model, tokenizer, device, formality_classifier)),
            'jscore_blockMH': float(compute_inidvidual_Jscore(blockMH, seed_text, model, tokenizer, device, formality_classifier))
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





