# This script fine-tunes a RoBERTa model for sequence classification
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import IntervalStrategy


# Load the classifier training data
data_path = "./processed/train_labeled.csv"
df = pd.read_csv(data_path)

# Split the data into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

# Apply tokenizer
train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Load model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./roberta-style-clf",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=2e-5,
    save_steps=500,
    save_total_limit=1,
    logging_steps=100,
    logging_dir="./logs"
)



# Evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Save final model
trainer.save_model("./final")
print("Fine-tuning complete and model saved!")