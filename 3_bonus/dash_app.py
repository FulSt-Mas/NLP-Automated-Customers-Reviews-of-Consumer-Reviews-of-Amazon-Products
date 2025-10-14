"""
Create Dynamic Dashboard
Purpose:
• Create a dynamic, clickable visualization dashboard using Plotly Dash.
• Display summarized reviews by product category in a clean, readable format.
Author: Massihlo Luca | "NLP Automated Customer Reviews"
"""

import os
import logging
import pandas as pd
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
    # Header
    html.Div([
        html.H1("Review Summarization Dashboard",
                style={"textAlign": "center", "color": "#fff", "fontFamily": "'Segoe UI', sans-serif", "margin": "20px 0"})
    ], style={
        "backgroundColor": "#2c3e50",
        "padding": "20px 0",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.2)"
    }),

    # Main Content Container
    html.Div([
        # Dropdown Section
        html.Div([
            dcc.Dropdown(
                id="category-dropdown",
                options=[{"label": cat.replace("_", " "), "value": cat} for cat in summary_df["product_category"].unique()],
                value=summary_df["product_category"].iloc[0],
                multi=False,
                placeholder="Select a product category...",
                style={
                    "width": "100%",
                    "color": "#333",
                    "backgroundColor": "#fff",
                    "border": "1px solid #ddd",
                    "borderRadius": "4px",
                    "padding": "10px"
                }
            ),
        ], style={
            "width": "80%",
            "margin": "20px auto",
            "padding": "15px",
            "backgroundColor": "#fff",
            "borderRadius": "8px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
        }),

        # Summary Text Section
        html.Div(id="summary-container", style={
            "width": "90%",
            "margin": "0 auto 20px",
            "padding": "20px",
            "backgroundColor": "#fff",
            "borderRadius": "8px",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
            "maxHeight": "70vh",
            "overflowY": "auto"
        })
    ], style={
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
        "padding": "20px 0"
    })
])

@app.callback(
    Output("summary-container", "children"),
    [Input("category-dropdown", "value")]
)
def update_dashboard(selected_category):
    filtered_df = summary_df[summary_df["product_category"] == selected_category]

    # Create a clean, organized layout for the summaries
    summary_content = []

    # Add a title for the selected category
    summary_content.append(
        html.H2(f"Review Summaries for {selected_category.replace('_', ' ')}",
                style={"textAlign": "center", "color": "#2c3e50", "marginBottom": "30px"})
    )

    # Group summaries by review score
    for score in sorted(filtered_df["review_score"].unique()):
        score_summaries = filtered_df[filtered_df["review_score"] == score]

        # Add a section header for each score
        summary_content.append(
            html.Div(
                f"Review Score: {score}",
                style={
                    "fontWeight": "bold",
                    "fontSize": "18px",
                    "color": "#2c3e50",
                    "borderBottom": "2px solid #2c3e50",
                    "paddingBottom": "5px",
                    "marginTop": "25px",
                    "marginBottom": "15px"
                }
            )
        )

        # Add each summary for this score
        for _, row in score_summaries.iterrows():
            summary_content.append(
                html.Div(
                    row['summary'],
                    style={
                        "marginBottom": "20px",
                        "lineHeight": "1.6",
                        "fontSize": "15px",
                        "color": "#333",
                        "textAlign": "justify",
                        "paddingLeft": "10px",
                        "borderLeft": "3px solid #e0e0e0"
                    }
                )
            )

    return summary_content

# ---- Run Dashboard ----
if __name__ == "__main__":
    logger.info(f"Starting dashboard on http://127.0.0.1:{DASHBOARD_PORT}")
    app.run_server(debug=True, port=DASHBOARD_PORT)
