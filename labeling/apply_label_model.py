import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier, LFAnalysis, labeling_function
from labeling.labeling_functions import lfs, WORLD, SPORTS, BUSINESS, SCITECH


import os

# ----------------------------
# 1. Constants
# ----------------------------
DATA_PATH = "data/processed/weak_unlabeled_processed.csv"
OUTPUT_PATH = "data/labeling_outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Define class values
class_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}
NUM_CLASSES = len(class_map)

# ----------------------------
# 2. Load Unlabeled Data
# ----------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} examples")

# ----------------------------
# 3. Apply LFs
# ----------------------------
applier = PandasLFApplier(lfs)
L_matrix = applier.apply(df)

# Optional: Show LF summary
LFAnalysis(L_matrix, lfs).lf_summary()

# ----------------------------
# 4. Train Snorkel LabelModel
# ----------------------------
label_model = LabelModel(cardinality=NUM_CLASSES, verbose=True)
label_model.fit(L_train=L_matrix, n_epochs=500, log_freq=100, seed=42)

# ----------------------------
# 5. Predict Labels
# ----------------------------
df["snorkel_label"] = label_model.predict(L_matrix)
df["snorkel_prob"] = label_model.predict_proba(L_matrix).max(axis=1)

# Save to CSV
df.to_csv(f"{OUTPUT_PATH}/weak_labeled_with_probs.csv", index=False)
print(f"✅ Weakly labeled data saved to {OUTPUT_PATH}/weak_labeled_with_probs.csv")
