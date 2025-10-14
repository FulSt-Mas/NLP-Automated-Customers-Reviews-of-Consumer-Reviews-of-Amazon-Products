# NLP Automated Customer Reviews of Amazon Products

**Leveraging NLP and Generative AI to Automate Customer Review Analysis**

---

## 📌 Overview

This project automates the analysis of customer reviews using NLP and Generative AI. It includes data preprocessing, model building, evaluation, and a dynamic dashboard for visualizing review summaries by product category and review score.

---

## 🛠️ Setup Instructions

### Prerequisites
- **Python Version:** 3.10
- **Dependencies:** Listed in `requirements.txt`

### Installation
1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt

🚀 Running the Application
To run the application, navigate to the root of the folder and execute:
 Kopierenstreamlit run app.py
For the Review Summarization Dashboard, run:

python bonus_review_summarization.py


Then, open the dashboard in your browser at http://127.0.0.1:8050.

📂 Project Structure

NLP_AUTOMATED_CUSTOMERS_REVIEWS_OF_CONSUMER_REVIEWS_OF_AMAZON_PRODUCTS/
│
├── artifacts/
│   ├── artifacts_1_data_preprocessing/
│   ├── artifacts_2_model_building/
│   └── artifacts_3_model_evaluation/
│
├── Data/
│   └── Reviews_of_Amazon_Products_sm.csv
│
├── Experiments/
│   ├── EDA_transformer_artifacts_1_data_preprocessing.ipynb
│   └── EDA.ipynb
│
├── transformer_artifacts_1_data_preprocessing/
├── transformer_artifacts_2_model_building/
├── transformer_artifacts_3_evaluation/
│
├── bonus_review_summarization.py
├── readme.md
├── requirements.txt
├── traditional_1_data_preprocessing.py
├── traditional_2_model_building.py
├── traditional_3_model_evaluation.py
├── transformer_1_preprocessor.py
├── transformer_2_model_building.py
└── transformer_3_model_evaluation.py

🎯 Key Features

Data Preprocessing: Clean and prepare customer review data.
Model Building: Build and fine-tune NLP models for sentiment analysis.
Model Evaluation: Evaluate model performance with accuracy and classification reports.
Generative AI Summarization: Summarize reviews by product category and review score.
Dynamic Dashboard: Interactive visualization of review summaries using Plotly Dash.


📝 License
This project is licensed under the Massih Chopan License.

📧 Contact
For any inquiries, please contact chopanmassih@gmail.com
