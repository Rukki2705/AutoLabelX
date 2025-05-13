import pandas as pd
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import os

tqdm.pandas()

# Initialize tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Basic cleaning function
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)                      # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)               # Remove special characters
    text = re.sub(r"\s+", " ", text)                         # Collapse multiple spaces
    return text.strip().lower()

# Tokenization function
def tokenize_text(text: str) -> dict:
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
        return_attention_mask=True
    )

# Paths
INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files_to_process = [
    "weak_unlabeled.csv",
    "labeled_small.csv",
    "val.csv",
    "test.csv"
]

for file in files_to_process:
    print(f"🔄 Processing {file}")
    df = pd.read_csv(os.path.join(INPUT_DIR, file))

    # Clean
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    # Tokenize
    tokenized = df['clean_text'].progress_apply(lambda x: tokenize_text(x))

    # Extract token ids and attention masks
    df['input_ids'] = tokenized.apply(lambda x: x['input_ids'][0].tolist())
    df['attention_mask'] = tokenized.apply(lambda x: x['attention_mask'][0].tolist())

    # Save processed data
    output_file = os.path.join(OUTPUT_DIR, file.replace(".csv", "_processed.csv"))
    df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")
