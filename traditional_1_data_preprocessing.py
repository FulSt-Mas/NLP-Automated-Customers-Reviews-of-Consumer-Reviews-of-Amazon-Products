"""
STEP 1: Data Preprocessing
==========================

Purpose:
- Clean, tokenize, and lemmatize text reviews
- Convert them into a vectorized document-term matrix
- Transform review ratings (1–5) into sentiment labels: Negative, Neutral, Positive
- Save artifacts for next step (model training)

Output:
- dtm_1.npz
- vectorizer_1.joblib
- preprocessed_reviews_1.csv

Author: Massih Project | "NLP Automated Customers Reviews"
"""

import os
import re
import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple
from scipy import sparse
import joblib

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TreebankWordTokenizer

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# ---- Logging setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- NLTK paths ----
NLTK_DATA_DIR = os.getenv("NLTK_DATA", "/Users/massih/nltk_data")
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)
logger.info("NLTK data path: %s", nltk.data.path)


# ---- Ensure NLTK resources ----
def _ensure_nltk_resource(resource_name: str, find_path: Optional[str] = None):
    try:
        nltk.data.find(find_path or resource_name)
        return True
    except LookupError:
        logger.warning(f"Downloading missing package: {resource_name}")
        nltk.download(resource_name, download_dir=NLTK_DATA_DIR, quiet=True)
        return True


_REQUIRED_NLTK = [
    ("punkt", "tokenizers/punkt"),
    ("wordnet", "corpora/wordnet"),
    ("omw-1.4", "corpora/omw-1.4"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
]
for name, path in _REQUIRED_NLTK:
    _ensure_nltk_resource(name, path)


# ---- Helpers ----
def _nltk_pos_to_wordnet_pos(tag: str) -> str:
    t = tag[0].upper()
    return {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }.get(t, wordnet.NOUN)


# ---- Step 1.1 – Clean text ----
class TextCleaner(TransformerMixin, BaseEstimator):
    def __init__(self, text_column: str = "reviews.text"):
        self.text_column = text_column

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.Series:
        if self.text_column not in X.columns:
            raise KeyError(f"Missing column: {self.text_column}")

        def clean(t: str):
            t = str(t).lower()
            t = re.sub(r"<[^>]+>", " ", t)
            t = re.sub(r"[^a-z0-9\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        return X[self.text_column].fillna("").apply(clean)


# ---- Step 1.2 – Tokenize + Lemmatize ----
class TokenizerLemmatizer(TransformerMixin, BaseEstimator):
    def __init__(self, return_list=False):
        self.lemmatizer = WordNetLemmatizer()
        self.return_list = return_list
        self.fallback_tokenizer = TreebankWordTokenizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.Series, list]) -> pd.Series:
        def process(text):
            if not text:
                return ""
            try:
                tokens = word_tokenize(text)
            except LookupError:
                tokens = self.fallback_tokenizer.tokenize(text)

            try:
                tags = pos_tag(tokens)
            except LookupError:
                tags = [(t, "NN") for t in tokens]

            lemmas = [
                self.lemmatizer.lemmatize(t, _nltk_pos_to_wordnet_pos(p))
                for t, p in tags
                if len(t) > 1 and not t.isnumeric()
            ]
            return lemmas if self.return_list else " ".join(lemmas)

        return pd.Series([process(t) for t in X])


# ---- Step 1.3 – NLP Preprocessor Class ----
class NLPPreprocessor:
    def __init__(self, text_column="reviews.text", vectorizer_type="tfidf"):
        self.text_column = text_column
        self.cleaner = TextCleaner(text_column)
        self.tokenizer = TokenizerLemmatizer()
        self.vectorizer = (
            TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, stop_words="english")
            if vectorizer_type == "tfidf"
            else CountVectorizer(max_features=10000)
        )
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame):
        logger.info("Cleaning and tokenizing text...")
        cleaned = self.cleaner.transform(df)
        tokenized = self.tokenizer.transform(cleaned)
        X = self.vectorizer.fit_transform(tokenized)
        self.fitted = True
        return X

    def save_vectorizer(self, path: str):
        joblib.dump(self.vectorizer, path)
        logger.info(f"Vectorizer saved: {path}")

    def save_matrix(self, X, path: str):
        sparse.save_npz(path, X)
        logger.info(f"DTM saved: {path}")

    def save_preprocessed_csv(self, df: pd.DataFrame, out_path: str):
        df.to_csv(out_path, index=False)
        logger.info(f"Preprocessed CSV saved: {out_path}")


# ---- Sentiment Mapping Function ----
def map_ratings_to_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Map numeric review ratings to sentiment labels."""
    if "reviews.rating" not in df.columns:
        raise KeyError("Expected 'reviews.rating' column in dataset.")

    def _map(score):
        try:
            s = int(score)
        except Exception:
            return None
        if s in [1, 2, 3]:
            return "negative"
        elif s == 4:
            return "neutral"
        elif s == 5:
            return "positive"
        return None

    df = df.copy()
    df["sentiment_label"] = df["reviews.rating"].apply(_map)
    df = df.dropna(subset=["sentiment_label"])
    return df


# ---- MAIN EXECUTION ----
if __name__ == "__main__":
    input_path = "Data/Reviews_of_Amazon_Products_sm.csv"
    out_dir = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products/artifacts/artifacts_1_data_preprocessing"
    os.makedirs(out_dir, exist_ok=True)

    vectorizer_path = os.path.join(out_dir, "vectorizer_1.joblib")
    dtm_path = os.path.join(out_dir, "dtm_1.npz")
    csv_path = os.path.join(out_dir, "preprocessed_reviews_1.csv")

    df = pd.read_csv(input_path)
    df = map_ratings_to_sentiment(df)

    pre = NLPPreprocessor(text_column="reviews.text")
    X = pre.fit_transform(df)

    pre.save_vectorizer(vectorizer_path)
    pre.save_matrix(X, dtm_path)
    pre.save_preprocessed_csv(df, csv_path)

    logger.info("✅ Step 1 complete: preprocessing artifacts saved successfully.")
