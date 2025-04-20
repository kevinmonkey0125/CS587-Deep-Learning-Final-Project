from pathlib import Path
import pandas as pd

modern_path = Path("/home/scharng/scratch/final_project/Shakespeare/data/align/model_16and7plays/data/train_plays1and2_clean.modern")
original_path = Path("/home/scharng/scratch/final_project/Shakespeare/data/align/model_16and7plays/data/train_plays1and2_clean.original")

with modern_path.open("r", encoding="utf-8") as f:
    modern_lines = [line.strip() for line in f.readlines()]

with original_path.open("r", encoding="utf-8") as f:
    original_lines = [line.strip() for line in f.readlines()]

assert len(modern_lines) == len(original_lines), "Mismatch!"

df = pd.DataFrame({     #total 27797 smaples
    "modern": modern_lines,
    "original": original_lines
})

# 隨機打亂資料 + 分割資料集 80% train, 10% valid, 10% test
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df_shuffled)
df_train = df_shuffled[:int(0.8 * n)]
df_valid = df_shuffled[int(0.8 * n):int(0.9 * n)]
df_test  = df_shuffled[int(0.9 * n):]

# 建立 classifier 資料（0: modern, 1: original）
df_classifier = pd.concat([
    pd.DataFrame({"sentence": df_train["modern"], "label": 0}),
    pd.DataFrame({"sentence": df_train["original"], "label": 1})
]).sample(frac=1, random_state=42).reset_index(drop=True)

# 顯示資料狀況
# print("Train Pairs:", len(df_train))
# print("Valid Pairs:", len(df_valid))
# print("Test Pairs:", len(df_test))
# print("\nClassifier preview:")
# print(df_classifier.head())


########################### fine tuning ###########################
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
df = df_classifier 

train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 設定模型
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="./roberta-style-clf",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    save_steps=500,
    save_total_limit=1,
    logging_steps=100,
)


# 指標（這邊只做 accuracy）
def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 開始訓練
trainer.train()
