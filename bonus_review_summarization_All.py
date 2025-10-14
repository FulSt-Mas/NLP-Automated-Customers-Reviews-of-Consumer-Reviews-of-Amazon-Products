# """
# BONUS: Generative AI Review Summarization & Dynamic Dashboard
# Purpose:
# • Use Generative AI to summarize reviews, broken down by review score (0-5) and product category.
# • Create a dynamic, clickable visualization dashboard using Plotly Dash.
# • If categories are too many, select top-K categories by review count.
# Author: Massih Project | "NLP Automated Customer Reviews"
# """

# import os
# import logging
# import pandas as pd
# import plotly.express as px
# from dash import Dash, dcc, html, Input, Output
# from transformers import pipeline

# # ---- Logging ----
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # # ---- Paths ----
# BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
# DATA_PATH = os.path.join(BASE_DIR, "artifacts", "transformer_artifacts_1_data_preprocessing", "preprocessed_reviews_transformer.csv")
# DASHBOARD_PORT = 8050

# # ---- Load Data ----
# logger.info(f"📂 Loading preprocessed data from {DATA_PATH}")
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

# # ---- Select Top-K Categories ----
# K = 10  # Number of top categories to visualize
# top_categories = df["product_category"].value_counts().nlargest(K).index
# df = df[df["product_category"].isin(top_categories)]

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

# # ---- Create Dynamic Dashboard ----
# app = Dash(__name__)

# app.layout = html.Div([
#     html.H1("Review Summarization Dashboard", style={"textAlign": "center"}),
#     dcc.Dropdown(
#         id="category-dropdown",
#         options=[{"label": cat, "value": cat} for cat in summary_df["product_category"].unique()],
#         value=summary_df["product_category"].iloc[0],
#         multi=False,
#         placeholder="Select a product category..."
#     ),
#     dcc.Graph(id="summary-scatter"),
#     html.Div(id="summary-text", style={"margin": "20px"})
# ])

# @app.callback(
#     [Output("summary-scatter", "figure"), Output("summary-text", "children")],
#     [Input("category-dropdown", "value")]
# )
# def update_dashboard(selected_category):
#     filtered_df = summary_df[summary_df["product_category"] == selected_category]
#     fig = px.scatter(
#         filtered_df,
#         x="review_score",
#         y="summary",
#         title=f"Review Summaries for {selected_category}",
#         labels={"review_score": "Review Score (0-5)", "summary": "AI-Generated Summary"},
#         hover_data=["review_score"]
#     )
#     fig.update_layout(yaxis={"type": "category"})
#     summary_text = f"### Summaries for {selected_category}\n\n{', '.join(filtered_df['summary'].tolist())}"
#     return fig, summary_text

# # ---- Run Dashboard ----
# if __name__ == "__main__":
#     logger.info(f"🚀 Starting dashboard on http://127.0.0.1:{DASHBOARD_PORT}")
#     app.run_server(debug=True, port=DASHBOARD_PORT)

"""
BONUS: Generative AI Review Summarization & Dynamic Dashboard
Purpose:
• Use Generative AI to summarize reviews, broken down by review score (0-5) and product category.
• Create a dynamic, clickable visualization dashboard using Plotly Dash.
• If categories are too many, select top-K categories by review count.
Author: Massihlo Luca | "NLP Automated Customer Reviews"
"""

import os
import logging
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from transformers import pipeline

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
DATA_PATH = os.path.join(BASE_DIR, "artifacts", "transformer_artifacts_1_data_preprocessing", "preprocessed_reviews_transformer.csv")
DASHBOARD_PORT = 8050

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

# ---- Create Dynamic Dashboard ----
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Review Summarization Dashboard", style={"textAlign": "center", "fontFamily": "Arial", "color": "#333"}),

    html.Div([
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": cat, "value": cat} for cat in summary_df["product_category"].unique()],
            value=summary_df["product_category"].iloc[0],
            multi=False,
            placeholder="Select a product category...",
            style={"width": "80%", "margin": "0 auto"}
        ),
    ], style={"padding": "20px", "backgroundColor": "#f9f9f9", "borderRadius": "5px"}),

    html.Div([
        dcc.Graph(id="summary-bar-chart", style={"height": "60vh"})
    ], style={"padding": "20px"}),

    html.Div(id="summary-text", style={
        "margin": "20px",
        "padding": "20px",
        "backgroundColor": "#f0f8ff",
        "borderRadius": "5px",
        "fontFamily": "Arial",
        "fontSize": "16px"
    })
])

@app.callback(
    [Output("summary-bar-chart", "figure"), Output("summary-text", "children")],
    [Input("category-dropdown", "value")]
)
def update_dashboard(selected_category):
    filtered_df = summary_df[summary_df["product_category"] == selected_category]

    # Create a bar chart for better visualization
    fig = px.bar(
        filtered_df,
        x="review_score",
        y="summary",
        title=f"Review Summaries for {selected_category}",
        labels={"review_score": "Review Score (0-5)", "summary": "AI-Generated Summary"},
        color="review_score",
        orientation='v',
        barmode='group'
    )
    fig.update_layout(
        yaxis={"type": "category"},
        xaxis_title="Review Score",
        yaxis_title="Summary",
        plot_bgcolor="white",
        title_font_size=24,
        title_x=0.5
    )

    # Display summaries in a readable format
    summaries = filtered_df.groupby("review_score")["summary"].apply(list).to_dict()
    summary_text = []
    for score, texts in summaries.items():
        summary_text.append(f"**Review Score: {score}**")
        summary_text.append(", ".join(texts))
        summary_text.append("---")

    summary_text = html.Div([html.P(text) for text in summary_text])

    return fig, summary_text

# ---- Run Dashboard ----
if __name__ == "__main__":
    logger.info(f"Starting dashboard on http://127.0.0.1:{DASHBOARD_PORT}")
    app.run_server(debug=True, port=DASHBOARD_PORT)

