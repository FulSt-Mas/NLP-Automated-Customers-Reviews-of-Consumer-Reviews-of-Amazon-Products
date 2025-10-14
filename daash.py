"""
Create Dynamic Dashboard
Purpose:
• Create a dynamic, clickable visualization dashboard using Plotly Dash.
• Display summarized reviews by product category and review score.
Author: Massih | "NLP Automated Customer Reviews"
"""

import os
import logging
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Paths ----
BASE_DIR = "/Users/massih/Code/Ironhack/1_Project/5_Project NLP | Automated Customers Reviews/NLP Automated Customers Reviews of Consumer Reviews of Amazon Products"
SUMMARY_CSV_PATH = os.path.join(BASE_DIR, "artifacts", "summarized_reviews.csv")
DASHBOARD_PORT = 8050

# ---- Load Summarized Data ----
logger.info(f"Loading summarized data from {SUMMARY_CSV_PATH}")
summary_df = pd.read_csv(SUMMARY_CSV_PATH)

# ---- Create Dynamic Dashboard ----
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Review Summarization Dashboard", style={
        "textAlign": "center",
        "fontFamily": "Arial",
        "color": "#333",
        "marginBottom": "20px"
    }),

    html.Div([
        dcc.Dropdown(
            id="category-dropdown",
            options=[{"label": cat, "value": cat} for cat in summary_df["product_category"].unique()],
            value=summary_df["product_category"].iloc[0],
            multi=False,
            placeholder="Select a product category...",
            style={"width": "80%", "margin": "0 auto"}
        ),
    ], style={
        "padding": "20px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "5px",
        "marginBottom": "20px"
    }),

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
