"""
STEP 2: Model Selection, Training, and Evaluation
=================================================

Purpose:
- Load preprocessed text data (Step 1 output)
- Train and compare ML models: Naive Bayes, Logistic Regression, SVM, RandomForest
- Evaluate using accuracy, precision, recall, F1
- Save best model and detailed performance report

Author: Massih Project | "NLP Automated Customers Reviews"
"""

import os
import joblib
import logging
import pandas as pd
from typing import Dict, Any
from scipy.sparse import load_npz
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Suppress future warnings from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
DATA_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts_1_data_preprocessing"
OUT_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts_2_model_building"

os.makedirs(OUT_DIR, exist_ok=True)

DTM_PATH = os.path.join(DATA_DIR, "dtm_1.npz")
CSV_PATH = os.path.join(DATA_DIR, "preprocessed_reviews_1.csv")

BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_model.joblib")
SUMMARY_CSV_PATH = os.path.join(OUT_DIR, "model_results_summary.csv")
SUMMARY_TXT_PATH = os.path.join(OUT_DIR, "model_summary_report.txt")


# ---- Model Trainer ----
class ModelTrainer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = self._init_models()
        self.results = {}
        self.best_model_info = None  # contains model object and metadata

    def _init_models(self) -> Dict[str, Dict[str, Any]]:
        return {
            "NaiveBayes": {
                "estimator": MultinomialNB(),
                "params": {"alpha": [0.5, 1.0]},
            },
            "LogisticRegression": {
                "estimator": LogisticRegression(max_iter=1000, solver="lbfgs"),
                "params": {"C": [0.1, 1, 5]},
            },
            "LinearSVM": {
                "estimator": LinearSVC(dual="auto"),
                "params": {"C": [0.5, 1, 5]},
            },
            "RandomForest": {
                "estimator": RandomForestClassifier(),
                "params": {
                    "n_estimators": [100, 300],
                    "max_depth": [None, 20],
                    "min_samples_split": [2, 5],
                },
            },
        }

    def run_grid_search(self, cv=3, scoring="f1_macro"):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info("🚀 Starting model selection via GridSearchCV...")

        for name, cfg in self.models.items():
            logger.info(f"🔍 Running Grid Search for {name}...")
            grid = GridSearchCV(
                cfg["estimator"],
                cfg["params"],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_

            preds = best_model.predict(X_test)
            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="macro", zero_division=0),
                "recall": recall_score(y_test, preds, average="macro", zero_division=0),
                "f1": f1_score(y_test, preds, average="macro", zero_division=0),
            }

            self.results[name] = {
                "best_params": grid.best_params_,
                "metrics": metrics,
                "report": classification_report(y_test, preds),
                "model": best_model,  # store actual estimator object
            }

            logger.info(
                f"✅ {name} done | F1: {metrics['f1']:.3f} | Params: {grid.best_params_}"
            )

        # Pick best model by F1
        best_name, best_info = max(
            self.results.items(), key=lambda kv: kv[1]["metrics"]["f1"]
        )
        self.best_model_info = {"name": best_name, **best_info}

        logger.info(f"🏆 Best model: {best_name}")

    def save_results(self):
        logger.info("💾 Saving model results...")
        # Save summary CSV
        rows = []
        for name, res in self.results.items():
            m = res["metrics"]
            rows.append({"Model": name, **m, "Best Params": res["best_params"]})

        pd.DataFrame(rows).to_csv(SUMMARY_CSV_PATH, index=False)

        # Save text report
        with open(SUMMARY_TXT_PATH, "w") as f:
            for name, res in self.results.items():
                f.write(f"{name}\nBest Params: {res['best_params']}\nMetrics: {res['metrics']}\n\n{res['report']}\n{'='*60}\n\n")

        # Save best model object
        joblib.dump(self.best_model_info["model"], BEST_MODEL_PATH)
        logger.info(f"✅ Best model saved: {BEST_MODEL_PATH}")
        logger.info(f"📑 Summary CSV: {SUMMARY_CSV_PATH}")
        logger.info(f"📜 Detailed report: {SUMMARY_TXT_PATH}")


# ---- MAIN ----
if __name__ == "__main__":
    logger.info("📦 Loading data...")
    X = load_npz(DTM_PATH)
    df = pd.read_csv(CSV_PATH)

    if "sentiment_label" not in df.columns:
        raise KeyError("Expected column 'sentiment_label' in preprocessed CSV.")
    y = df["sentiment_label"]

    trainer = ModelTrainer(X, y)
    trainer.run_grid_search(cv=3)
    trainer.save_results()

    logger.info("🎯 Model building complete.")
