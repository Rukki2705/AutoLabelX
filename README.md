# 🚀 AutoLabelX: Human-in-the-Loop Text Labeling Framework

AutoLabelX is a semi-automated pipeline for scalable and intelligent text data labeling. It combines **weak supervision (Snorkel)**, **active learning (uncertainty sampling)**, and **human-in-the-loop feedback** to reduce manual labeling effort and iteratively improve model performance.

---

## 🔍 Project Highlights

- ⚙️ **Weak Supervision with Snorkel**: Label large unlabeled text datasets using pattern-based Labeling Functions (LFs).
- 🧠 **DistilBERT Classifier**: Trains an initial classifier on weak labels.
- ❓ **Uncertainty Sampling**: Identifies low-confidence predictions to prioritize human review.
- 🙋‍♂️ **Streamlit Review Dashboard**: Lets users correct uncertain predictions via a user-friendly interface.
- 🔁 **Retrain Loop**: Human feedback is merged and used to refine the model.

---

## 📁 Project Structure

```
AutoLabelX/
│
├── data/ # Raw, processed, and labeled datasets
│ ├── raw/
│ ├── processed/
│ └── labeling_outputs/
│
├── labeling/ # Snorkel labeling functions + label model
│ ├── labeling_functions.py
│ └── apply_label_model.py
│
├── active_learning/ # Training and review loop
│ ├── train_model.py
│ ├── uncertainty_sampling.py
│ ├── streamlit_dashboard.py
│ └── label_store/
│
├── models/ # Fine-tuned transformer models
│
├── scripts/ # Data preparation and preprocessing
│ └── prepare_agnews.py
│ └── preprocess_text.py
│
├── requirements.txt
```
