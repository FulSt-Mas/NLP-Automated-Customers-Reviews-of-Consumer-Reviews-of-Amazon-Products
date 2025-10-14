"""
STEP 2.2: Transformer Fine-Tuning with Transfer Learning
---------------------------------------------------------
Purpose:
• Load preprocessed reviews (Step 1 output)
• Use a pre-trained Transformer (DistilBERT) from Hugging Face
• Apply transfer learning by fine-tuning on the review dataset
• Evaluate model performance (accuracy + report)
• Save model, tokenizer, and results as artifacts

Author: Massih Project | "NLP Automated Customer Reviews"
"""

import os
import logging
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm
import joblib
import pickle


# -------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Paths and constants
# -------------------------------------------------------------
BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
DATA_PATH = os.path.join(
    BASE_DIR,
    "artifacts/transformer_artifacts_1_data_preprocessing/preprocessed_reviews_transformer.csv",
)
OUT_DIR = os.path.join(BASE_DIR, "artifacts/transformer_artifacts_2_model_building")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------
# Class Definition
# -------------------------------------------------------------
class TransformerModelEvaluator:
    """Evaluate and fine-tune a pre-trained Transformer model for sentiment classification."""

    def __init__(self, model_name: str, data_path: str):
        self.model_name = model_name
        self.data_path = data_path
        self.tokenizer = None
        self.model = None
        logger.info(f"🧠 Using device: {DEVICE}")

    # ---------------------------------------------------------
    def load_model(self, num_labels=None):
        """Load tokenizer and (optionally) model head for classification."""
        logger.info(f"🔧 Loading tokenizer and model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if num_labels:
            # Load with classification head (transfer learning)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name, num_labels=num_labels
            )
        else:
            # Zero-shot inference model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(DEVICE)

    # ---------------------------------------------------------
    def load_data(self):
        """Load and validate the preprocessed dataset."""
        logger.info(f"📂 Loading preprocessed data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        required_cols = {"cleaned_text", "sentiment_label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise KeyError(f"Missing required columns: {missing}")
        df = df.dropna(subset=["cleaned_text", "sentiment_label"])
        return df[["cleaned_text", "sentiment_label"]]

    # ---------------------------------------------------------
    def predict_single(self, text: str) -> int:
        """Predict sentiment label for a single text sample."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).cpu().item()
        return predicted_class

    # ---------------------------------------------------------
    def evaluate(self, df: pd.DataFrame, fine_tune: bool = False):
        """Evaluate the model (zero-shot or fine-tuned)."""
        if fine_tune:
            self._fine_tune_and_evaluate(df)
        else:
            self._zero_shot_evaluate(df)

    # ---------------------------------------------------------
    def _zero_shot_evaluate(self, df: pd.DataFrame):
        """Evaluate pre-trained model without fine-tuning."""
        logger.info("🚀 Starting base model (zero-shot) evaluation...")
        y_true = df["sentiment_label"].astype("category").cat.codes.tolist()
        texts = df["cleaned_text"].tolist()
        predictions = []
        for text in tqdm(texts, desc="Predicting"):
            pred = self.predict_single(text)
            predictions.append(pred)
        acc = accuracy_score(y_true, predictions)
        report = classification_report(
            y_true, predictions, target_names=["negative", "neutral", "positive"], zero_division=0
        )
        logger.info(f"✅ Base model accuracy: {acc:.3f}")
        logger.info("\n" + report)
        self._save_results(acc, report, "base_model_results.txt")
        self._save_model_and_tokenizer("base_model_distilbert")

    # ---------------------------------------------------------
    def _fine_tune_and_evaluate(self, df: pd.DataFrame):
        """Fine-tune the model using transfer learning."""
        logger.info("🔧 Preparing data for fine-tuning (transfer learning)...")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"].tolist(),
            df["sentiment_label"].astype("category").cat.codes.tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df["sentiment_label"].astype("category").cat.codes,
        )

        # Tokenize
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=128)
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=128)

        # Dataset
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset = SentimentDataset(train_encodings, y_train)
        test_dataset = SentimentDataset(test_encodings, y_test)

        # ✅ Transfer learning training configuration
        training_args = TrainingArguments(
            output_dir=OUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{OUT_DIR}/logs",
            logging_steps=20,
            load_best_model_at_end=True,
            save_total_limit=2,
        )

        def compute_metrics(pred):
            preds = pred.predictions.argmax(-1)
            acc = accuracy_score(pred.label_ids, preds)
            return {"accuracy": acc}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("🚀 Starting fine-tuning (transfer learning)...")
        trainer.train()
        logger.info("✅ Fine-tuning complete")

        # Evaluation
        logger.info("📊 Evaluating fine-tuned model...")
        preds_output = trainer.predict(test_dataset)
        preds = torch.argmax(torch.tensor(preds_output.predictions), axis=1)
        acc = accuracy_score(y_test, preds)
        report = classification_report(
            y_test, preds, target_names=["negative", "neutral", "positive"], zero_division=0
        )

        logger.info(f"✅ Fine-tuned (transfer-learned) model accuracy: {acc:.3f}")
        logger.info("\n" + report)

        self._save_results(acc, report, "fine_tuned_model_results.txt")
        self._save_model_and_tokenizer("fine_tuned_model_distilbert_transfer_learning")
        self._save_model_as_joblib_and_pickle("fine_tuned_model_distilbert_transfer_learning")

    # ---------------------------------------------------------
    def _save_results(self, acc, report, filename):
        results_path = os.path.join(OUT_DIR, filename)
        with open(results_path, "w") as f:
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Accuracy: {acc:.3f}\n\n")
            f.write(report)
        logger.info(f"📄 Results saved to: {results_path}")

    # ---------------------------------------------------------
    def _save_model_and_tokenizer(self, model_dir):
        model_save_path = os.path.join(OUT_DIR, model_dir)
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        logger.info(f"💾 Model + tokenizer saved to: {model_save_path}")

    # ---------------------------------------------------------
    def _save_model_as_joblib_and_pickle(self, model_dir):
        model_save_path = os.path.join(OUT_DIR, model_dir)
        joblib_file = os.path.join(OUT_DIR, f"{model_dir}_model.joblib")
        pickle_file = os.path.join(OUT_DIR, f"{model_dir}_model.pkl")
        joblib.dump(self.model, joblib_file)
        with open(pickle_file, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"✅ Joblib model: {joblib_file}")
        logger.info(f"✅ Pickle model: {pickle_file}")


# -------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------
if __name__ == "__main__":
    evaluator = TransformerModelEvaluator(model_name=MODEL_NAME, data_path=DATA_PATH)
    evaluator.load_model(num_labels=3)        # 3 sentiment classes
    df = evaluator.load_data()

    # Choose one of the following:
    # --- Zero-shot baseline
    # evaluator.evaluate(df, fine_tune=False)

    # --- Transfer learning fine-tuning
    evaluator.evaluate(df, fine_tune=True)
