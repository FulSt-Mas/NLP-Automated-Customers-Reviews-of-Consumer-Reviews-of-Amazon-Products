# # """
# # STEP 3: MODEL EVALUATION
# # -------------------------
# # This script evaluates the trained sentiment classification model.
# # It loads the best saved model, evaluates it on test data, calculates metrics,
# # and produces visual & textual reports.

# # Author: Massih
# # """

# # import os
# # import logging
# # import joblib
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import (
# #     accuracy_score,
# #     precision_recall_fscore_support,
# #     classification_report,
# #     confusion_matrix,
# #     ConfusionMatrixDisplay,
# # )
# # from sklearn.model_selection import train_test_split

# # # Setup logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format="%(asctime)s - %(levelname)s - %(message)s",
# # )

# # # -------------------- PATHS --------------------
# # DATA_PATH = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts_1_data_preprocessing/preprocessed_reviews_1.csv"
# # VECTORIZER_PATH = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts_1_data_preprocessing/vectorizer_1.joblib"
# # BEST_MODEL_PATH = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts_2_model_building/best_model.joblib"
# # OUTPUT_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts_3_model_evaluation"


# # class ModelEvaluator:
# #     """Handles model evaluation, metrics computation, and reporting."""

# #     def __init__(self):
# #         os.makedirs(OUTPUT_DIR, exist_ok=True)
# #         self.data_path = DATA_PATH
# #         self.vectorizer_path = VECTORIZER_PATH
# #         self.model_path = BEST_MODEL_PATH

# #     def load_artifacts(self):
# #         """Load data, vectorizer, and model."""
# #         logging.info("📦 Loading evaluation artifacts...")
# #         self.df = pd.read_csv(self.data_path)
# #         self.vectorizer = joblib.load(self.vectorizer_path)
# #         self.model = joblib.load(self.model_path)
# #         logging.info("✅ Artifacts loaded successfully.")

# #     def prepare_data(self):
# #         """Split preprocessed data into train/test and vectorize reviews."""
# #         logging.info("🧹 Preparing data for evaluation...")

# #         X = self.df["cleaned_text"]
# #         y = self.df["sentiment"]

# #         # Train-test split (ensure same random state as training)
# #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
# #             X, y, test_size=0.2, random_state=42, stratify=y
# #         )

# #         # Transform test data using saved vectorizer
# #         self.X_test_vect = self.vectorizer.transform(self.X_test)
# #         logging.info(f"✅ Data prepared | Test samples: {self.X_test.shape[0]}")

# #     def evaluate_model(self):
# #         """Compute metrics and generate evaluation results."""
# #         logging.info("🚀 Evaluating model...")

# #         y_pred = self.model.predict(self.X_test_vect)

# #         # Compute metrics
# #         acc = accuracy_score(self.y_test, y_pred)
# #         precision, recall, f1, _ = precision_recall_fscore_support(
# #             self.y_test, y_pred, average="macro"
# #         )

# #         class_report = classification_report(
# #             self.y_test, y_pred, output_dict=True
# #         )

# #         cm = confusion_matrix(self.y_test, y_pred, labels=["negative", "neutral", "positive"])

# #         # Store metrics
# #         self.results = {
# #             "accuracy": acc,
# #             "precision": precision,
# #             "recall": recall,
# #             "f1": f1,
# #             "classification_report": class_report,
# #             "confusion_matrix": cm,
# #         }

# #         logging.info(f"✅ Model evaluation complete | Accuracy: {acc:.3f}")

# #     def save_reports(self):
# #         """Save metrics, confusion matrix, and detailed report."""
# #         logging.info("💾 Saving evaluation reports...")

# #         # --- Save metrics summary ---
# #         summary = {
# #             "Accuracy": [self.results["accuracy"]],
# #             "Precision": [self.results["precision"]],
# #             "Recall": [self.results["recall"]],
# #             "F1 Score": [self.results["f1"]],
# #         }
# #         pd.DataFrame(summary).to_csv(
# #             os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False
# #         )

# #         # --- Save detailed classification report ---
# #         report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
# #         with open(report_path, "w") as f:
# #             f.write("MODEL EVALUATION REPORT\n")
# #             f.write("=======================\n\n")
# #             f.write(f"Accuracy: {self.results['accuracy']:.3f}\n")
# #             f.write(f"Precision: {self.results['precision']:.3f}\n")
# #             f.write(f"Recall: {self.results['recall']:.3f}\n")
# #             f.write(f"F1 Score: {self.results['f1']:.3f}\n\n")
# #             f.write("DETAILED CLASSIFICATION REPORT\n")
# #             f.write("==============================\n")
# #             f.write(pd.DataFrame(self.results["classification_report"]).to_string())

# #         # --- Confusion Matrix Visualization ---
# #         cm_display = ConfusionMatrixDisplay(
# #             confusion_matrix=self.results["confusion_matrix"],
# #             display_labels=["Negative", "Neutral", "Positive"],
# #         )
# #         cm_display.plot(cmap="Blues", values_format="d")
# #         plt.title("Confusion Matrix")
# #         plt.tight_layout()
# #         plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
# #         plt.close()

# #         logging.info(f"✅ Reports saved successfully in {OUTPUT_DIR}")

# #     def run_pipeline(self):
# #         """Run the full model evaluation pipeline."""
# #         self.load_artifacts()
# #         self.prepare_data()
# #         self.evaluate_model()
# #         self.save_reports()


# # if __name__ == "__main__":
# #     evaluator = ModelEvaluator()
# #     evaluator.run_pipeline()
# #     logging.info("🎯 Model evaluation pipeline complete.")

# """
# STEP 3: Model Evaluation
# ========================

# Evaluates the trained model on a test dataset using accuracy, precision, recall, and F1-score.
# Also displays a confusion matrix.

# Author: Massih Project | "NLP Automated Customers Reviews"
# """

# import os
# import logging
# import pandas as pd
# import joblib
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     confusion_matrix,
#     classification_report,
# )
# from sklearn.model_selection import train_test_split


# # ---------------------- Logging ----------------------
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )


# class ModelEvaluator:
#     def __init__(self, data_path, model_path, vectorizer_path):
#         """
#         Initializes model evaluator.

#         Args:
#             data_path (str): Path to preprocessed dataset (.csv)
#             model_path (str): Path to saved trained model (.pkl)
#             vectorizer_path (str): Path to saved vectorizer (.joblib)
#         """
#         self.data_path = data_path
#         self.model_path = model_path
#         self.vectorizer_path = vectorizer_path
#         self.df = None
#         self.model = None
#         self.vectorizer = None

#     # ---------------------- Step 1: Load Artifacts ----------------------
#     def load_artifacts(self):
#         logging.info("📦 Loading evaluation artifacts...")
#         self.df = pd.read_csv(self.data_path)
#         self.model = joblib.load(self.model_path)
#         self.vectorizer = joblib.load(self.vectorizer_path)
#         logging.info("✅ Artifacts loaded successfully.")

#     # ---------------------- Step 2: Prepare Data ----------------------
#     def prepare_data(self):
#         """Select text and sentiment columns, split into train/test."""
#         logging.info("🧹 Preparing data for evaluation...")

#         # Automatically detect which text column exists
#         possible_text_cols = ["cleaned_text", "reviews.text", "text", "review"]
#         text_col = next((col for col in possible_text_cols if col in self.df.columns), None)

#         if not text_col:
#             raise KeyError("❌ No valid text column found (expected e.g. 'reviews.text').")

#         # Detect sentiment column
#         sentiment_col = next(
#             (col for col in ["sentiment", "sentiment_label", "label"] if col in self.df.columns),
#             None,
#         )

#         if not sentiment_col:
#             raise KeyError("❌ No sentiment column found (expected 'sentiment_label').")

#         logging.info(f"📝 Using text column: {text_col}")
#         logging.info(f"🧭 Using sentiment column: {sentiment_col}")

#         X = self.df[text_col]
#         y = self.df[sentiment_col]

#         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42, stratify=y
#         )
#         logging.info(f"✅ Data prepared: {len(self.X_test)} test samples")

#     # ---------------------- Step 3: Evaluate Model ----------------------
#     def evaluate(self):
#         """Evaluate model and show metrics."""
#         logging.info("📊 Evaluating model performance...")

#         X_test_vec = self.vectorizer.transform(self.X_test)
#         y_pred = self.model.predict(X_test_vec)

#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             self.y_test,
#             y_pred,
#             average=None,
#             labels=["positive", "negative", "neutral"],
#             zero_division=0,
#         )

#         print("\n===== Classification Report =====")
#         print(classification_report(self.y_test, y_pred))

#         cm = confusion_matrix(self.y_test, y_pred, labels=["positive", "negative", "neutral"])
#         self.plot_confusion_matrix(cm, ["Positive", "Negative", "Neutral"])

#         logging.info("🎯 Evaluation complete")
#         logging.info(f"Accuracy: {accuracy:.2%}")

#         return {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         }

#     # ---------------------- Step 4: Confusion Matrix ----------------------
#     def plot_confusion_matrix(self, cm, labels):
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
#         plt.title("Confusion Matrix")
#         plt.xlabel("Predicted")
#         plt.ylabel("Actual")
#         plt.tight_layout()
#         plt.show()

#     # ---------------------- Step 5: Pipeline Runner ----------------------
#     def run_pipeline(self):
#         self.load_artifacts()
#         self.prepare_data()
#         metrics = self.evaluate()
#         logging.info(f"Final Results: {metrics}")


# # ---------------------- Execution ----------------------
# if __name__ == "__main__":
#     DATA_PATH = "artifacts_1_data_preprocessing/preprocessed_reviews_1.csv"
#     MODEL_PATH = "artifacts_2_model_training/sentiment_model.pkl"
#     VECTORIZER_PATH = "artifacts_1_data_preprocessing/vectorizer_1.joblib"

#     evaluator = ModelEvaluator(DATA_PATH, MODEL_PATH, VECTORIZER_PATH)
#     evaluator.run_pipeline()

"""
STEP 3: Model Evaluation
========================

Purpose:
- Load preprocessed text data and trained model from Step 2
- Evaluate model performance using metrics: accuracy, precision, recall, F1
- Generate confusion matrix and classification report
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

DATA_DIR = os.path.join(BASE_DIR, "artifacts_1_data_preprocessing")
MODEL_DIR = os.path.join(BASE_DIR, "artifacts_2_model_building")
EVAL_DIR = os.path.join(BASE_DIR, "artifacts_3_model_evaluation")

os.makedirs(EVAL_DIR, exist_ok=True)

# Artifacts paths
DTM_PATH = os.path.join(DATA_DIR, "dtm_1.npz")
CSV_PATH = os.path.join(DATA_DIR, "preprocessed_reviews_1.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")

EVAL_REPORT_TXT = os.path.join(EVAL_DIR, "evaluation_report.txt")
EVAL_SUMMARY_CSV = os.path.join(EVAL_DIR, "evaluation_summary.csv")
CONF_MATRIX_IMG = os.path.join(EVAL_DIR, "confusion_matrix.png")


# ---- Model Evaluator ----
class ModelEvaluator:
    def __init__(self, model_path, dtm_path, csv_path):
        self.model_path = model_path
        self.dtm_path = dtm_path
        self.csv_path = csv_path

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

        preds = self.model.predict(X_test)

        self.metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1_score": f1_score(y_test, preds, average="macro", zero_division=0),
        }

        logger.info(f"✅ Model evaluation complete | F1: {self.metrics['f1_score']:.3f}")
        self.report = classification_report(y_test, preds)
        self.cm = confusion_matrix(y_test, preds)

        # Save confusion matrix visualization
        plt.figure(figsize=(6, 5))
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(CONF_MATRIX_IMG)
        plt.close()

        logger.info(f"📊 Confusion matrix saved: {CONF_MATRIX_IMG}")

        return self.metrics

    def save_reports(self):
        logger.info("💾 Saving evaluation reports...")
        # Save text report
        with open(EVAL_REPORT_TXT, "w") as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("Metrics Summary:\n")
            for k, v in self.metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\nClassification Report:\n")
            f.write(self.report)

        # Save summary CSV
        pd.DataFrame([self.metrics]).to_csv(EVAL_SUMMARY_CSV, index=False)

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
