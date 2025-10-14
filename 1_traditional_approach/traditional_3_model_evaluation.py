"""
STEP 3: Model Evaluation
========================
Purpose:
- Load preprocessed text data and trained model from Step 2
- Evaluate model performance on both train and test sets using metrics: accuracy, precision, recall, F1
- Generate confusion matrix and classification report for both sets
- Save all evaluation results as artifacts
Author: Massih Project | "NLP Automated Customers Reviews"
"""
import os
import joblib
import logging
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# ---- Logging setup ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
DATA_DIR = os.path.join(BASE_DIR, "artifacts/artifacts_1_data_preprocessing")
MODEL_DIR = os.path.join(BASE_DIR, "artifacts/artifacts_2_model_building")
EVAL_DIR = os.path.join(BASE_DIR, "artifacts/artifacts_3_model_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Artifacts paths
DTM_PATH = os.path.join(DATA_DIR, "dtm_1.npz")
CSV_PATH = os.path.join(DATA_DIR, "preprocessed_reviews_1.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

# Evaluation artifacts paths
EVAL_REPORT_TXT = os.path.join(EVAL_DIR, "evaluation_report.txt")
EVAL_SUMMARY_CSV = os.path.join(EVAL_DIR, "evaluation_summary.csv")
CONF_MATRIX_TEST_IMG = os.path.join(EVAL_DIR, "confusion_matrix_test.png")
CONF_MATRIX_TRAIN_IMG = os.path.join(EVAL_DIR, "confusion_matrix_train.png")

# ---- Model Evaluator ----
class ModelEvaluator:
    def __init__(self, model_path, dtm_path, csv_path):
        self.model_path = model_path
        self.dtm_path = dtm_path
        self.csv_path = csv_path
        self.metrics_train = {}
        self.metrics_test = {}

    def load_artifacts(self):
        logger.info("📦 Loading evaluation artifacts...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not os.path.exists(self.dtm_path):
            raise FileNotFoundError(f"DTM not found: {self.dtm_path}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.model = joblib.load(self.model_path)
        self.X = load_npz(self.dtm_path)
        self.df = pd.read_csv(self.csv_path)
        if "sentiment_label" not in self.df.columns:
            raise KeyError("Missing column 'sentiment_label' in dataset.")
        self.y = self.df["sentiment_label"]
        logger.info("✅ Artifacts loaded successfully.")

    def evaluate_model(self):
        logger.info("🧠 Evaluating model performance...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Evaluate on train set
        train_preds = self.model.predict(X_train)
        self.metrics_train = {
            "accuracy": accuracy_score(y_train, train_preds),
            "precision": precision_score(y_train, train_preds, average="macro", zero_division=0),
            "recall": recall_score(y_train, train_preds, average="macro", zero_division=0),
            "f1_score": f1_score(y_train, train_preds, average="macro", zero_division=0),
        }
        self.report_train = classification_report(y_train, train_preds)
        self.cm_train = confusion_matrix(y_train, train_preds)

        # Evaluate on test set
        test_preds = self.model.predict(X_test)
        self.metrics_test = {
            "accuracy": accuracy_score(y_test, test_preds),
            "precision": precision_score(y_test, test_preds, average="macro", zero_division=0),
            "recall": recall_score(y_test, test_preds, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, test_preds, average="macro", zero_division=0),
        }
        self.report_test = classification_report(y_test, test_preds)
        self.cm_test = confusion_matrix(y_test, test_preds)

        # Save confusion matrix visualizations
        self._save_confusion_matrix(self.cm_train, CONF_MATRIX_TRAIN_IMG, "Confusion Matrix (Train)")
        self._save_confusion_matrix(self.cm_test, CONF_MATRIX_TEST_IMG, "Confusion Matrix (Test)")

        logger.info(f"✅ Model evaluation complete | Train F1: {self.metrics_train['f1_score']:.3f} | Test F1: {self.metrics_test['f1_score']:.3f}")
        return self.metrics_train, self.metrics_test

    def _save_confusion_matrix(self, cm, path, title):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"📊 Confusion matrix saved: {path}")

    def save_reports(self):
        logger.info("💾 Saving evaluation reports...")
        # Save text report
        with open(EVAL_REPORT_TXT, "w") as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("Train Metrics Summary:\n")
            for k, v in self.metrics_train.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\nTrain Classification Report:\n")
            f.write(self.report_train)
            f.write("\n\n" + "=" * 50 + "\n\n")
            f.write("Test Metrics Summary:\n")
            for k, v in self.metrics_test.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\nTest Classification Report:\n")
            f.write(self.report_test)

        # Save summary CSV
        pd.DataFrame([self.metrics_train, self.metrics_test], index=["train", "test"]).to_csv(EVAL_SUMMARY_CSV)
        logger.info(f"✅ Evaluation text report saved: {EVAL_REPORT_TXT}")
        logger.info(f"✅ Evaluation summary CSV saved: {EVAL_SUMMARY_CSV}")

    def run_pipeline(self):
        self.load_artifacts()
        self.evaluate_model()
        self.save_reports()
        logger.info("🏁 Model evaluation pipeline complete.")

# ---- MAIN ----
if __name__ == "__main__":
    evaluator = ModelEvaluator(MODEL_PATH, DTM_PATH, CSV_PATH)
    evaluator.run_pipeline()
