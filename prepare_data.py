# File 1: prepare_data.py
# This script splits the original parallel corpus into train/valid/test sets,
# and generates labeled datasets for training and evaluating a style classifier.

from pathlib import Path
import pandas as pd

# Load original files
modern_path = Path("/home/scharng/scratch/final_project/Shakespeare/data/align/model_16and7plays/data/train_plays1and2_clean.modern")
original_path = Path("/home/scharng/scratch/final_project/Shakespeare/data/align/model_16and7plays/data/train_plays1and2_clean.original")

with modern_path.open("r", encoding="utf-8") as f:
    modern_lines = [line.strip() for line in f.readlines()]

with original_path.open("r", encoding="utf-8") as f:
    original_lines = [line.strip() for line in f.readlines()]

assert len(modern_lines) == len(original_lines), "Mismatch!"

# Build parallel sentence DataFrame
df = pd.DataFrame({"modern": modern_lines, "original": original_lines})

# Shuffle and split: 80% train / 10% valid / 10% test
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df_shuffled)
df_train = df_shuffled[:int(0.8 * n)]
df_valid = df_shuffled[int(0.8 * n):int(0.9 * n)]
df_test  = df_shuffled[int(0.9 * n):]

# Build classifier training data (0: modern, 1: original) using training portion only
df_train_labeled = pd.concat([
    pd.DataFrame({"sentence": df_train["modern"], "label": 0}),
    pd.DataFrame({"sentence": df_train["original"], "label": 1})
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Also create labeled versions of validation and test sets for evaluation
df_valid_labeled = pd.concat([
    pd.DataFrame({"sentence": df_valid["modern"], "label": 0}),
    pd.DataFrame({"sentence": df_valid["original"], "label": 1})
]).sample(frac=1, random_state=42).reset_index(drop=True)

df_test_labeled = pd.concat([
    pd.DataFrame({"sentence": df_test["modern"], "label": 0}),
    pd.DataFrame({"sentence": df_test["original"], "label": 1})
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save all datasets to CSV for later use
output_dir = Path("/home/scharng/scratch/final_project/processed")
output_dir.mkdir(exist_ok=True)

df_train.to_csv(output_dir / "train_parallel.csv", index=False)
df_valid.to_csv(output_dir / "valid_parallel.csv", index=False)
df_test.to_csv(output_dir / "test_parallel.csv", index=False)
df_train_labeled.to_csv(output_dir / "train_labeled.csv", index=False)
df_valid_labeled.to_csv(output_dir / "valid_labeled.csv", index=False)
df_test_labeled.to_csv(output_dir / "test_labeled.csv", index=False)

print("✅ Data preparation complete!")