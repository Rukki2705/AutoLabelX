from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load AG News dataset
dataset = load_dataset("ag_news")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Confirm the correct column names
print(train_df.columns)  # Should show: ['text', 'label']

# No need to rename — keep as is
# Ensure label is integer
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = test_df['label'].astype(int)

# Check class balance
print("Train class distribution:\n", train_df['label'].value_counts())

# Shuffle
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into weakly labeled, labeled, validation
weak_unlabeled, labeled = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df['label'],
    random_state=42
)

labeled, val_df = train_test_split(
    labeled,
    test_size=0.25,
    stratify=labeled['label'],
    random_state=42
)

# Drop labels to simulate unlabeled data
weak_unlabeled = weak_unlabeled.drop(columns=['label'])

# Save
os.makedirs("data/raw", exist_ok=True)
weak_unlabeled.to_csv("data/raw/weak_unlabeled.csv", index=False)
labeled.to_csv("data/raw/labeled_small.csv", index=False)
val_df.to_csv("data/raw/val.csv", index=False)
test_df.to_csv("data/raw/test.csv", index=False)

print("✅ Dataset preparation completed.")
