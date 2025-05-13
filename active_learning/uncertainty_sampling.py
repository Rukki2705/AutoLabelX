import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader


# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "models/bert_weak_label"
DATA_PATH = "data/processed/weak_unlabeled_processed.csv"
OUTPUT_PATH = "data/active_learning/low_confidence_samples.csv"
TOP_K = 100  # number of samples to send for human review
BATCH_SIZE = 16

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ----------------------------
# Load Model & Tokenizer
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------
# Prepare Dataset
# ----------------------------
class UnlabeledDataset(Dataset):
    def __init__(self, df):
        self.input_ids = df['input_ids'].apply(eval).tolist()
        self.attention_mask = df['attention_mask'].apply(eval).tolist()

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx]),
            'attention_mask': torch.tensor(self.attention_mask[idx])
        }

    def __len__(self):
        return len(self.input_ids)

# Load data
df = pd.read_csv(DATA_PATH)
dataset = UnlabeledDataset(df)

# ----------------------------
# Run Model & Get Uncertainty
# ----------------------------
all_probs = []

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

for batch in tqdm(dataloader, total=len(dataloader)):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=-1)
        all_probs.extend(probs.cpu().numpy())

# ----------------------------
# Compute Entropy
# ----------------------------
def compute_entropy(prob_row):
    return -np.sum(prob_row * np.log(prob_row + 1e-10))

entropies = np.array([compute_entropy(p) for p in all_probs])
df['uncertainty'] = entropies
df['predicted_label'] = np.argmax(all_probs, axis=1)

# ----------------------------
# Select Top-K Uncertain Samples
# ----------------------------
df_sorted = df.sort_values(by="uncertainty", ascending=False).head(TOP_K)
df_sorted.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved {TOP_K} low-confidence samples to {OUTPUT_PATH}")
