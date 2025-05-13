import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import os

# ----------------------------
# 1. Dataset and Config
# ----------------------------
NUM_LABELS = 4
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 3

DATA_PATH = "data/labeling_outputs/weak_labeled_with_probs.csv"
VAL_PATH = "data/processed/val_processed.csv"
MODEL_OUTPUT_DIR = "models/bert_weak_label"

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 2. Define PyTorch Dataset
# ----------------------------
class AGNewsDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.encodings = {
            'input_ids': df['input_ids'].apply(eval).tolist(),
            'attention_mask': df['attention_mask'].apply(eval).tolist()
        }
        self.labels = df['snorkel_label'].tolist() if 'snorkel_label' in df.columns else df['label'].tolist()


    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# ----------------------------
# 3. Load Data
# ----------------------------
train_df = pd.read_csv(DATA_PATH)
train_df = train_df[train_df['snorkel_label'] != -1].reset_index(drop=True)
val_df = pd.read_csv(VAL_PATH)

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

train_dataset = AGNewsDataset(train_df, tokenizer)
val_dataset = AGNewsDataset(val_df, tokenizer)

# ----------------------------
# 4. Evaluation Metrics
# ----------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted')
    }

# ----------------------------
# 5. Model + Trainer
# ----------------------------
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir=f"{MODEL_OUTPUT_DIR}/logs"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save final model
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print(f"✅ Model saved to {MODEL_OUTPUT_DIR}")
