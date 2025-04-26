from Evaluate_Disc_BertScore import compute_E_disc, rescaled_BERTScore
def energy(x, seed, model, tokenizer, device, ALPHA_DISC=0.5, ALPHA_BERT=0.5):
    # E_disc
    E_disc, _ = compute_E_disc(x, model, tokenizer, device)
    # E_BERTScore
    E_bert = -rescaled_BERTScore([x], [seed])[0]
    return ALPHA_DISC * E_disc + ALPHA_BERT * E_bert


def compute_inidvidual_Jscore(output, seed, model, tokenizer, device, formality_classifier):
    if isinstance(output, list):  # 如果是 summary (list of sentences)
        print(" Running Jscore for SUMMARY (multi-sentence)")
        accs, sims, fls = [], [], []
        for i, sent in enumerate(output):
            print(f"\n Sentence {i+1}: {sent}")

            # Accuracy (probability it's Shakespeare)
            _, probs = compute_E_disc(sent, model, tokenizer, device)
            acc = 1 if probs[1] > probs[0] else 0
            accs.append(acc)
            print(f"   ACC: {acc} (probs = {probs})")

            # Similarity to seed
            sim = rescaled_BERTScore([sent], [seed])[0]
            sims.append(sim)
            print(f"   SIM: {sim}")

            # Formality level
            formality_result = formality_classifier(sent)[0]
            fl = 1 if formality_result['label'] == 'LABEL_1' else 0
            fls.append(fl)
            print(f"  🧑‍⚖️ FL: {fl} (label = {formality_result['label']})")

        # Final score
        avg_score = (sum(accs)/len(accs)) * (sum(sims)/len(sims)) * (sum(fls)/len(fls))
        print(f"\n Final averaged Jscore = {avg_score:.4f}")
        return avg_score

    else:  # 如果是單句 blockMH
        print(" Running Jscore for BLOCK (single sentence)")
        print(f"\n Sentence: {output}")

        _, probs = compute_E_disc(output, model, tokenizer, device)
        acc = 1 if probs[1] > probs[0] else 0
        print(f"   ACC: {acc} (probs = {probs})")

        sim = rescaled_BERTScore([output], [seed])[0]
        print(f"   SIM: {sim}")

        formality_result = formality_classifier(output)[0]
        fl = 1 if formality_result['label'] == 'LABEL_1' else 0
        print(f"   FL: {fl} (label = {formality_result['label']})")

        final_score = acc * sim * fl
        print(f"\n Final Jscore = {final_score:.4f}")
        return final_score






def get_lowest_energy_sample(seed_chains):
    all_samples = []
    for chain in seed_chains:
        all_samples.extend(chain)  
            
    # Sort by energy (second element of tuple) and get the lowest
    if all_samples:
        best_sample = min(all_samples, key=lambda x: x[1])
        return best_sample[0]  # Return the sentence only
    return None