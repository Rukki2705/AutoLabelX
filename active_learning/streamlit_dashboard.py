import streamlit as st
import pandas as pd
import os

# ----------------------------
# Config
# ----------------------------
DATA_PATH = "data/active_learning/low_confidence_samples.csv"
REVIEW_PATH = "data/active_learning/human_labels.csv"

# Ensure folder exists
os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)

# ----------------------------
# Load Data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if os.path.exists(REVIEW_PATH):
        reviewed = pd.read_csv(REVIEW_PATH)
        reviewed_ids = set(reviewed["text"].tolist())
        df = df[~df["text"].isin(reviewed_ids)]  # remove already reviewed
    return df.reset_index(drop=True)

df = load_data()

# ----------------------------
# App Interface
# ----------------------------
st.title("🧠 Human-in-the-Loop Labeling UI")
st.markdown("Review low-confidence samples and provide corrected labels.")

if df.empty:
    st.success("✅ All samples reviewed!")
    st.stop()

sample = df.iloc[0]
st.subheader("📝 Sample Text")
st.write(sample["clean_text"])

st.markdown(f"**Model Prediction:** `{sample['predicted_label']}`")
st.markdown(f"**Uncertainty Score:** `{sample['uncertainty']:.4f}`")

# ----------------------------
# Label Selection
# ----------------------------
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}
label_dict = {v: k for k, v in label_map.items()}

label_strings = [v for _, v in label_map.items()]
default_label = label_map[int(sample["predicted_label"])]
user_label = st.radio("Select the correct label:", label_strings, index=label_strings.index(default_label))


if st.button("Submit Label"):
    # Save review
    new_entry = {
        "text": sample["text"],
        "clean_text": sample["clean_text"],
        "model_pred": sample["predicted_label"],
        "uncertainty": sample["uncertainty"],
        "human_label": label_dict[user_label]
    }
    if os.path.exists(REVIEW_PATH):
        reviewed_df = pd.read_csv(REVIEW_PATH)
        reviewed_df = pd.concat([reviewed_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        reviewed_df = pd.DataFrame([new_entry])

    reviewed_df.to_csv(REVIEW_PATH, index=False)
    st.success("✅ Label submitted. Reloading...")
    st.rerun()
