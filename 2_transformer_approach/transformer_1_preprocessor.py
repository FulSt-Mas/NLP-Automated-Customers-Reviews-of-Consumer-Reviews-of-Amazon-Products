"""
STEP 1: Transformer Preprocessor

Transformer preprocessing for "NLP | Automated Customers Reviews"

Performs:
 - Data loading
 - Text cleaning (lowercase, remove HTML/special chars, collapse whitespace)
 - Rating -> sentiment mapping (negative/neutral/positive) and numeric label
 - Tokenization + encoding using HuggingFace tokenizer (AutoTokenizer)
 - Saves artifacts:
     * tokenizer saved (tokenizer.save_pretrained)
     * encodings (input_ids, attention_mask, labels) as encodings.npz
     * preprocessed csv with cleaned_text, sentiment_label, sentiment_id

Usage:
  python transformer_preprocessor.py

Adjust the INPUT_PATH and OUTPUT_BASE variables as needed.

Requires: transformers, pandas, numpy
"""

import os
import re
import logging
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

# Transformers import (AutoTokenizer)
try:
    from transformers import AutoTokenizer
except Exception as e:
    raise ImportError(
        "The 'transformers' library is required for this script. "
        "Install with: pip install transformers\nOriginal error: " + str(e)
    )

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("transformer_preprocessor")


class TransformerPreprocessor:
    """
    Class to prepare text data for Transformer models (HuggingFace).
    """

    def __init__(
        self,
        input_csv_path: str,
        output_base_dir: str,
        text_column_candidates: Optional[List[str]] = None,
        rating_column_candidates: Optional[List[str]] = None,
        hf_model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 256,
        overwrite: bool = False,
    ):
        """
        Args:
            input_csv_path: path to raw CSV (your reviews CSV)
            output_base_dir: base directory where artifact folder will be created
            text_column_candidates: list of possible text column names to search for
            rating_column_candidates: list of possible rating column names to search for
            hf_model_name: HuggingFace tokenizer model id (default DistilBERT uncased)
            max_length: maximum token length (truncation)
            batch_size: tokenizer batching size (controls memory)
            overwrite: if True will overwrite existing artifacts
        """
        self.input_csv_path = input_csv_path
        self.output_base_dir = os.path.abspath(output_base_dir)
        self.artifact_dir = os.path.join(self.output_base_dir, "artifacts/transformer_artifacts_1_data_preprocessing")
        os.makedirs(self.artifact_dir, exist_ok=True)

        # default column candidates
        self.text_column_candidates = text_column_candidates or ["reviews.text", "reviewText", "text", "reviews_text"]
        self.rating_column_candidates = rating_column_candidates or ["reviews.rating", "rating", "stars"]

        self.hf_model_name = hf_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.overwrite = overwrite

        # artifacts paths
        self.tokenizer_dir = os.path.join(self.artifact_dir, "tokenizer")
        self.encodings_path = os.path.join(self.artifact_dir, "encodings.npz")
        self.output_csv_path = os.path.join(self.artifact_dir, "preprocessed_reviews_transformer.csv")

        # placeholders
        self.df = None
        self.text_col = None
        self.rating_col = None
        self.tokenizer = None

    # -------------------------
    # Utility helpers
    # -------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean a single review string:
          - convert to str, lowercase
          - remove HTML-like tags
          - replace non-alphanumeric (keep spaces) with space
          - collapse multiple whitespace to single space and strip
        """
        if pd.isna(text):
            return ""
        s = str(text)
        s = s.lower()
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"[^0-9a-z\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def map_rating_to_sentiment_label(rating: int) -> Optional[str]:
        """
        Map rating to sentiment label:
         - 1,2,3 -> 'negative'
         - 4     -> 'neutral'
         - 5     -> 'positive'
        Returns None for invalid/missing ratings.
        """
        try:
            r = int(rating)
        except Exception:
            return None
        if r in (1, 2, 3):
            return "negative"
        if r == 4:
            return "neutral"
        if r == 5:
            return "positive"
        return None

    # -------------------------
    # Step 1: load data
    # -------------------------
    def load_csv(self):
        logger.info("Loading CSV: %s", self.input_csv_path)
        if not os.path.exists(self.input_csv_path):
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv_path}")
        self.df = pd.read_csv(self.input_csv_path)
        logger.info("CSV loaded: %d rows, %d cols", len(self.df), len(self.df.columns))

        # auto-detect columns
        for c in self.text_column_candidates:
            if c in self.df.columns:
                self.text_col = c
                break
        if not self.text_col:
            raise KeyError(
                f"No text column found. Checked: {self.text_column_candidates}. Please specify the column name."
            )

        for c in self.rating_column_candidates:
            if c in self.df.columns:
                self.rating_col = c
                break
        if not self.rating_col:
            logger.warning(
                "No rating column found. Proceeding without rating->sentiment mapping. "
                "You can still run tokenization and save encodings without labels."
            )

        logger.info("Using text column: %s", self.text_col)
        if self.rating_col:
            logger.info("Using rating column: %s", self.rating_col)

    # -------------------------
    # Step 2: clean texts & map ratings
    # -------------------------
    def preprocess_texts_and_ratings(self):
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_csv() first.")
        logger.info("Cleaning text column and mapping ratings (if present)...")

        # cleaned text column
        self.df["cleaned_text"] = self.df[self.text_col].fillna("").astype(str).map(self.clean_text)

        # map sentiments
        if self.rating_col:
            self.df["sentiment_label"] = self.df[self.rating_col].map(self.map_rating_to_sentiment_label)
            # drop rows without label (optional)
            before = len(self.df)
            self.df = self.df.dropna(subset=["sentiment_label"]).reset_index(drop=True)
            after = len(self.df)
            logger.info("Dropped %d rows without valid sentiment label (left: %d)", before - after, after)
            # numeric ids for modelling (optional order: negative=0, neutral=1, positive=2)
            label2id = {"negative": 0, "neutral": 1, "positive": 2}
            self.df["sentiment_id"] = self.df["sentiment_label"].map(label2id)
        else:
            logger.info("No rating column -> skipping sentiment mapping.")

    # -------------------------
    # Step 3: load tokenizer
    # -------------------------
    def load_tokenizer(self):
        logger.info("Loading tokenizer: %s", self.hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        # create tokenizer dir if not exists
        os.makedirs(self.tokenizer_dir, exist_ok=True)

    # -------------------------
    # Step 4: tokenize & encode to ids
    # -------------------------
    def tokenize_and_encode(self):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")
        if self.df is None:
            raise RuntimeError("Data not loaded/preprocessed. Call preprocess_texts_and_ratings() first.")

        texts = self.df["cleaned_text"].tolist()
        n = len(texts)
        logger.info("Tokenizing %d texts (batch_size=%d, max_length=%d)...", n, self.batch_size, self.max_length)

        all_input_ids = []
        all_attention_masks = []

        # batch encode to avoid high mem usage
        for i in range(0, n, self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="np",  # returns numpy arrays
            )
            all_input_ids.append(enc["input_ids"])
            all_attention_masks.append(enc["attention_mask"])

        input_ids = np.vstack(all_input_ids)
        attention_mask = np.vstack(all_attention_masks)
        logger.info("Tokenization complete. input_ids shape: %s", input_ids.shape)

        # labels (if exist)
        if "sentiment_id" in self.df.columns:
            labels = self.df["sentiment_id"].values.astype(np.int64)
        else:
            labels = None

        # store in object for saving
        self.encodings = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # -------------------------
    # Step 5: save artifacts
    # -------------------------
    def save_artifacts(self):
        logger.info("Saving artifacts to: %s", self.artifact_dir)

        # tokenizer (save_pretrained)
        logger.info("Saving tokenizer to %s", self.tokenizer_dir)
        self.tokenizer.save_pretrained(self.tokenizer_dir)

        # encodings as compressed npz
        enc_path = self.encodings_path
        labels = self.encodings.get("labels")
        if labels is None:
            # save without labels
            np.savez_compressed(enc_path, input_ids=self.encodings["input_ids"], attention_mask=self.encodings["attention_mask"])
            logger.info("Encodings saved (no labels): %s", enc_path)
        else:
            np.savez_compressed(enc_path, input_ids=self.encodings["input_ids"], attention_mask=self.encodings["attention_mask"], labels=labels)
            logger.info("Encodings (with labels) saved: %s", enc_path)

        # preprocessed csv (append cleaned_text and sentiment columns)
        self.df.to_csv(self.output_csv_path, index=False)
        logger.info("Preprocessed CSV saved: %s", self.output_csv_path)

    # -------------------------
    # Run full pipeline
    # -------------------------
    def run(self):
        # guard: skip if artifacts exist unless overwrite True
        already = os.path.exists(self.encodings_path) and os.path.exists(self.tokenizer_dir) and os.path.exists(self.output_csv_path)
        if already and not self.overwrite:
            logger.info("Artifacts already exist in %s. Set overwrite=True to recreate.", self.artifact_dir)
            return

        self.load_csv()
        self.preprocess_texts_and_ratings()
        self.load_tokenizer()
        self.tokenize_and_encode()
        self.save_artifacts()
        logger.info("✅ Transformer preprocessing complete.")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    INPUT_PATH = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/Data/Reviews_of_Amazon_Products_sm.csv"
    OUTPUT_BASE = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts"

    pre = TransformerPreprocessor(
        input_csv_path=INPUT_PATH,
        output_base_dir=OUTPUT_BASE,
        hf_model_name="distilbert-base-uncased",
        max_length=128,
        batch_size=256,
        overwrite=True,  # set True on first run to ensure artifacts replaced
    )
    pre.run()

    # After running, artifacts are in:
    #   <OUTPUT_BASE>/transformer_artifacts_1_data_preprocessing/
    #       tokenizer/                # tokenizer files (vocab/config)
    #       encodings.npz             # compressed numpy arrays (input_ids, attention_mask, labels)
    #       preprocessed_reviews_transformer.csv  # original + cleaned_text + sentiment_label + sentiment_id
