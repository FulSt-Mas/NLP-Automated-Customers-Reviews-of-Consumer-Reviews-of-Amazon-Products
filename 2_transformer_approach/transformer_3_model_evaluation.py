"""
STEP 3: Transformer Model Evaluation (Base + Fine-Tuned)
-------------------------------------------------------

What this does
- Loads preprocessed transformer CSV with columns: cleaned_text, sentiment_label (and sentiment_id)
- Loads base transformer model (e.g. distilbert-base-uncased)
- Loads fine-tuned model from artifacts folder (prefer saved pretrained dir)
- Splits data into train/validation and evaluates both models on the same validation set
- Computes: accuracy, precision, recall, f1 (per-class + macro), confusion matrix
- Saves: text report, CSV summary, confusion matrix PNG(s)

Usage:
    python 3_transformer_model_evaluation.py

Author: adapted for Massih project
"""

import os
import logging
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import joblib
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from matplotlib import pyplot as plt
from tqdm import tqdm

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths and constants (adjust if needed) ----
BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
PREPROCESSED_CSV = os.path.join(
    BASE_DIR, "artifacts/transformer_artifacts_1_data_preprocessing/preprocessed_reviews_transformer.csv"
)

# fine-tuned artifacts (directory where model.save_pretrained(...) wrote files)
FINETUNED_DIR = os.path.join(
    BASE_DIR,
    "artifacts/transformer_artifacts_2_model_building/2_finetune_model_transfarlearning/fine_tuned_model_distilbert_transfer_learning",
)

# fallback joblib/pickle versions (if you created them)
FINETUNED_JOBLIB = os.path.join(
    BASE_DIR,
    "artifacts/transformer_artifacts_2_model_building/2_finetune_model_transfarlearning/fine_tuned_model_distilbert_transfer_learning_model.joblib",
)
FINETUNED_PICKLE = os.path.join(
    BASE_DIR,
    "artifacts/transformer_artifacts_2_model_building/2_finetune_model_transfarlearning/fine_tuned_model_distilbert_transfer_learning_model.pkl",
)

# base model name used earlier
BASE_MODEL_NAME = "distilbert-base-uncased"

# output evaluation artifacts
OUT_DIR = os.path.join(BASE_DIR, "artifacts/transformer_artifacts_3_evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)


# ---------------- Helper functions ----------------
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed CSV not found: {path}")
    df = pd.read_csv(path)
    return df


def label_to_codes(series: pd.Series) -> Tuple[List[str], np.ndarray]:
    """
    Convert sentiment_label column (strings) to categorical codes and return mapping.
    Returns: (categories_list_in_order, codes_array)
    """
    cat = series.astype("category")
    categories = list(cat.cat.categories)
    codes = cat.cat.codes.to_numpy()
    return categories, codes


def safe_load_tokenizer_and_model_from_dir(path: str) -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSequenceClassification]]:
    """
    Try to load tokenizer and model from a HuggingFace 'save_pretrained' directory.
    Returns (tokenizer, model) or (None, None) if not possible.
    """
    if not os.path.isdir(path):
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(DEVICE)
        logger.info("Loaded model/tokenizer from directory: %s", path)
        return tokenizer, model
    except Exception as e:
        logger.warning("Failed to load HF model/tokenizer from dir %s: %s", path, e)
        return None, None


def safe_load_joblib_or_pickle(path_joblib: str, path_pickle: str):
    """Attempt to load model object from joblib or pickle (fallback)"""
    if os.path.exists(path_joblib):
        try:
            obj = joblib.load(path_joblib)
            logger.info("Loaded joblib model from: %s", path_joblib)
            return obj
        except Exception as e:
            logger.warning("joblib load failed: %s", e)
    if os.path.exists(path_pickle):
        try:
            with open(path_pickle, "rb") as f:
                obj = pickle.load(f)
            logger.info("Loaded pickle model from: %s", path_pickle)
            return obj
        except Exception as e:
            logger.warning("pickle load failed: %s", e)
    return None


def predict_batch(model, tokenizer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Batch predictions returning integer class indices."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=128).to(DEVICE)
            outputs = model(**enc)
            logits = outputs.logits.cpu().numpy()
            batch_preds = np.argmax(logits, axis=1)
            preds.append(batch_preds)
    if preds:
        return np.concatenate(preds)
    return np.array([], dtype=int)


def save_confusion_matrix_png(cm: np.ndarray, labels: List[str], out_path: str, title: str):
    """Plot and save confusion matrix using matplotlib (no seaborn)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    # Annotate numbers
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix image: %s", out_path)


# ---------------- Main evaluator class ----------------
class TransformerEvaluator:
    """
    Loads base and fine-tuned transformer models, evaluates on a holdout validation set,
    and writes evaluation artifacts (reports, CSV summary, confusion matrices).
    """

    def __init__(
        self,
        preprocessed_csv: str,
        base_model_name: str = BASE_MODEL_NAME,
        finetuned_dir: str = FINETUNED_DIR,
        finetuned_joblib: str = FINETUNED_JOBLIB,
        finetuned_pickle: str = FINETUNED_PICKLE,
        out_dir: str = OUT_DIR,
    ):
        self.csv_path = preprocessed_csv
        self.base_model_name = base_model_name
        self.finetuned_dir = finetuned_dir
        self.finetuned_joblib = finetuned_joblib
        self.finetuned_pickle = finetuned_pickle
        self.out_dir = out_dir

        # placeholders
        self.df: Optional[pd.DataFrame] = None
        self.tokenizer_base = None
        self.model_base = None
        self.tokenizer_ft = None
        self.model_ft = None
        self.label_names: List[str] = []

    def load_data_and_prepare(self, test_size: float = 0.2, random_state: int = 42):
        logger.info("Loading preprocessed CSV: %s", self.csv_path)
        df = load_csv(self.csv_path)

        # check columns you told me exist
        for col in ("cleaned_text", "sentiment_label"):
            if col not in df.columns:
                raise KeyError(f"Expected column '{col}' in preprocessed CSV. Found: {list(df.columns)}")

        # drop NAs in those columns
        df = df.dropna(subset=["cleaned_text", "sentiment_label"]).reset_index(drop=True)

        # convert sentiment_label to categorical codes, preserve order
        categories, codes = label_to_codes(df["sentiment_label"])
        self.label_names = categories
        df["sentiment_code"] = codes

        # split for evaluation (train unused here; validation is held-out test)
        _, df_val = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["sentiment_code"])

        self.df = df_val.reset_index(drop=True)
        logger.info("Prepared validation set: %d rows; label distribution:\n%s", len(self.df), self.df["sentiment_label"].value_counts().to_dict())

    def load_base_model(self):
        """Load standard pretrained model/tokenizer (no fine-tuning)."""
        logger.info("Loading base tokenizer and model: %s", self.base_model_name)
        self.tokenizer_base = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model_base = AutoModelForSequenceClassification.from_pretrained(self.base_model_name)
        self.model_base.to(DEVICE)

    def load_finetuned_model(self):
        """Load fine-tuned model/tokenizer from saved directory (preferred)."""
        # try dir first
        tok, mdl = safe_load_tokenizer_and_model_from_dir(self.finetuned_dir)
        if tok is not None and mdl is not None:
            self.tokenizer_ft = tok
            self.model_ft = mdl
            return

        # else try joblib/pickle fallback
        logger.info("Attempting fallback load from joblib/pickle...")
        obj = safe_load_joblib_or_pickle(self.finetuned_joblib, self.finetuned_pickle)
        if obj is None:
            raise FileNotFoundError("Could not find or load the fine-tuned model (dir or joblib/pickle).")
        # If obj is a transformers model, set
        if isinstance(obj, (AutoModelForSequenceClassification,)):
            self.model_ft = obj
            # tokenizer not available in joblib/pickle fallback; try directory tokenizer
            if os.path.isdir(self.finetuned_dir):
                self.tokenizer_ft = AutoTokenizer.from_pretrained(self.finetuned_dir)
            else:
                raise RuntimeError("Tokenizer missing for the fine-tuned model. Prefer saved model dir.")
        else:
            # If joblib contained a huggingface model instance pickled, we'll try to use it directly
            self.model_ft = obj
            if os.path.isdir(self.finetuned_dir):
                self.tokenizer_ft = AutoTokenizer.from_pretrained(self.finetuned_dir)
            else:
                # if tokenizer not found, attempt to reuse base tokenizer (less ideal)
                logger.warning("Tokeniser not found for fine-tuned model; using base tokenizer as fallback")
                self.tokenizer_ft = self.tokenizer_base

        if self.model_ft is not None:
            self.model_ft.to(DEVICE)
            logger.info("Fine-tuned model loaded.")

    def evaluate_model(self, model, tokenizer, texts: List[str], true_codes: np.ndarray, name: str) -> Dict:
        """Run predictions and compute metrics for a model/tokenizer pair on held-out texts."""
        logger.info("Running predictions for: %s", name)
        preds = predict_batch(model, tokenizer, texts, batch_size=32)
        acc = accuracy_score(true_codes, preds)
        precision, recall, f1, support = precision_recall_fscore_support(true_codes, preds, labels=range(len(self.label_names)), zero_division=0)
        report_text = classification_report(true_codes, preds, target_names=self.label_names, zero_division=0)
        cm = confusion_matrix(true_codes, preds, labels=range(len(self.label_names)))

        # Save per-model artifacts
        result = {
            "name": name,
            "accuracy": float(acc),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "report": report_text,
            "confusion_matrix": cm,
            "predictions": preds,
        }
        return result

    def run(self):
        if self.df is None:
            self.load_data_and_prepare()

        # ensure base tokenizer/model loaded
        if self.tokenizer_base is None or self.model_base is None:
            self.load_base_model()

        # load finetuned
        try:
            self.load_finetuned_model()
        except Exception as e:
            logger.warning("Fine-tuned model load failed: %s. Continuing with base-only evaluation.", e)
            self.tokenizer_ft = None
            self.model_ft = None

        texts = self.df["cleaned_text"].tolist()
        true_codes = self.df["sentiment_code"].to_numpy()

        results = []

        # Evaluate base model
        res_base = self.evaluate_model(self.model_base, self.tokenizer_base, texts, true_codes, name="base_model")
        results.append(res_base)
        # Save confusion matrix img
        cm_path_base = os.path.join(self.out_dir, "confusion_matrix_base.png")
        save_confusion_matrix_png(res_base["confusion_matrix"], self.label_names, cm_path_base, "Base model confusion matrix")

        # Evaluate fine-tuned if present
        if self.model_ft is not None and self.tokenizer_ft is not None:
            res_ft = self.evaluate_model(self.model_ft, self.tokenizer_ft, texts, true_codes, name="fine_tuned_model")
            results.append(res_ft)
            cm_path_ft = os.path.join(self.out_dir, "confusion_matrix_finetuned.png")
            save_confusion_matrix_png(res_ft["confusion_matrix"], self.label_names, cm_path_ft, "Fine-tuned model confusion matrix")
        else:
            logger.info("No fine-tuned model available; skipping fine-tuned evaluation.")

        # Save summary CSV and text report(s)
        summary_rows = []
        for r in results:
            row = {
                "model": r["name"],
                "accuracy": r["accuracy"],
                "precision_macro": float(np.mean(r["precision_per_class"])),
                "recall_macro": float(np.mean(r["recall_per_class"])),
                "f1_macro": float(np.mean(r["f1_per_class"])),
            }
            summary_rows.append(row)

            # write detailed report
            txt_path = os.path.join(self.out_dir, f"{r['name']}_classification_report.txt")
            with open(txt_path, "w") as f:
                f.write(f"Model: {r['name']}\n")
                f.write(f"Accuracy: {r['accuracy']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(r["report"])
                f.write("\n\nConfusion Matrix:\n")
                f.write(np.array2string(r["confusion_matrix"]))
            logger.info("Saved detailed report: %s", txt_path)

        # write summary csv
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = os.path.join(self.out_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info("Saved evaluation summary CSV: %s", summary_csv_path)

        # overall summary
        logger.info("Evaluation complete. Summary:\n%s", summary_df.to_string(index=False))


# ---------------- Run as script ----------------
if __name__ == "__main__":
    evaluator = TransformerEvaluator(
        preprocessed_csv=PREPROCESSED_CSV,
        base_model_name=BASE_MODEL_NAME,
        finetuned_dir=FINETUNED_DIR,
        finetuned_joblib=FINETUNED_JOBLIB,
        finetuned_pickle=FINETUNED_PICKLE,
        out_dir=OUT_DIR,
    )

    # prepare validation set (20% of data)
    evaluator.load_data_and_prepare(test_size=0.2, random_state=42)

    # run evaluation (will attempt to load fine-tuned model; if fails run base-only)
    evaluator.run()
