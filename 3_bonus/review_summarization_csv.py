"""
Summarize Reviews by Score and Category
Purpose:
• Use Generative AI to summarize reviews, broken down by review score (0-5) and product category.
• Save the summarized results to a CSV file.
Author: Massih | "NLP Automated Customer Reviews"
"""

import os
import logging
import pandas as pd
from transformers import pipeline

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
DATA_PATH = os.path.join(BASE_DIR, "artifacts", "transformer_artifacts_1_data_preprocessing", "preprocessed_reviews_transformer.csv")
SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "artifacts", "summarized_reviews.csv")

# ---- Load Data ----
logger.info(f"Loading preprocessed data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Adjust column names based on your data
required_cols = {"cleaned_text", "sentiment_label", "primaryCategories", "reviews.rating"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Rename columns for clarity
df = df.rename(columns={
    "primaryCategories": "product_category",
    "reviews.rating": "review_score"
})

# ---- Select Top-K Categories ----
K = 10  # Number of top categories to visualize
top_categories = df["product_category"].value_counts().nlargest(K).index
df = df[df["product_category"].isin(top_categories)]

# ---- Initialize Generative Summarizer ----
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ---- Summarize Reviews by Score and Category ----
def summarize_reviews_by_group(df, group_cols, text_col, max_reviews_per_group=5):
    summaries = []
    for name, group in df.groupby(group_cols):
        reviews = group[text_col].tolist()[:max_reviews_per_group]
        if reviews:
            summary = summarizer(" ".join(reviews), max_length=60, min_length=30, do_sample=False)
            summaries.append({
                **dict(zip(group_cols, name)),
                "summary": summary[0]["summary_text"]
            })
    return pd.DataFrame(summaries)

summary_df = summarize_reviews_by_group(
    df,
    group_cols=["product_category", "review_score"],
    text_col="cleaned_text"
)

# Save the summarized data to a CSV file
summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
logger.info(f"Summarized reviews saved to {SUMMARY_CSV_PATH}")


# """
# Summarize Reviews by Score and Category
# Purpose:
# • Use Generative AI to summarize reviews, broken down by review score (0-5) and product category.
# • Save the summarized results to a CSV file.
# Author: Massih | "NLP Automated Customer Reviews"
# """

# import os
# import logging
# import pandas as pd
# from transformers import pipeline

# # ---- Logging ----
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # ---- Paths ----
# BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
# DATA_PATH = os.path.join(BASE_DIR, "artifacts", "transformer_artifacts_1_data_preprocessing", "preprocessed_reviews_transformer.csv")
# SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "artifacts", "summarized_reviews.csv")

# # ---- Load Data ----
# logger.info(f"Loading preprocessed data from {DATA_PATH}")
# df = pd.read_csv(DATA_PATH)

# # Adjust column names based on your data
# required_cols = {"cleaned_text", "sentiment_label", "primaryCategories", "reviews.rating"}
# missing = required_cols - set(df.columns)
# if missing:
#     raise KeyError(f"Missing required columns: {missing}")

# # Rename columns for clarity
# df = df.rename(columns={
#     "primaryCategories": "product_category",
#     "reviews.rating": "review_score"
# })

# # Clean and standardize product categories
# def clean_category(category):
#     # Replace commas with underscores and strip any whitespace
#     return '_'.join(sorted(set(c.strip() for c in category.split(','))))

# df['product_category'] = df['product_category'].apply(clean_category)

# # Define the desired categories
# desired_categories = {
#     'Electronics_Hardware_Media',
#     'Office_Supplies'
# }

# # Filter the DataFrame to only include desired categories
# df = df[df['product_category'].isin(desired_categories)]

# # ---- Initialize Generative Summarizer ----
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # ---- Summarize Reviews by Score and Category ----
# def summarize_reviews_by_group(df, group_cols, text_col, max_reviews_per_group=5):
#     summaries = []
#     for name, group in df.groupby(group_cols):
#         reviews = group[text_col].tolist()[:max_reviews_per_group]
#         if reviews:
#             summary = summarizer(" ".join(reviews), max_length=60, min_length=30, do_sample=False)
#             summaries.append({
#                 **dict(zip(group_cols, name)),
#                 "summary": summary[0]["summary_text"]
#             })
#     return pd.DataFrame(summaries)

# summary_df = summarize_reviews_by_group(
#     df,
#     group_cols=["product_category", "review_score"],
#     text_col="cleaned_text"
# )

# # Save the summarized data to a CSV file
# summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
# logger.info(f"Summarized reviews saved to {SUMMARY_CSV_PATH}")
