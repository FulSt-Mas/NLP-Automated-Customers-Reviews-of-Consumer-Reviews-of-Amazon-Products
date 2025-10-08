

1. Dependency roadmap for your entire NLP | Automated Customer Reviews project

✅ Step 1. Compatible Python Version

For this mix of:
traditional ML (scikit-learn, pandas, numpy, matplotlib, seaborn)
transformer models (HuggingFace Transformers, PyTorch)
dashboard visualization (Plotly / Tableau export)
→ the most stable, widely compatible base version is:
Python 3.10.14

It ensures full support for the latest compatible versions of transformers, torch, and scikit-learn, without breaking dependencies (some newer packages have issues with 3.12).

✅ Step 2. Library Groups and Compatibility Roadmap
Purpose	Library	Recommended Version	Dependency Notes
Core environment management	pip	24.2	Default package manager
setuptools	75.1.0	Required for smooth installs
wheel	0.44.0	Needed for building wheels
🧹 1. Data Preprocessing and Utilities
Library	Version	Reason / Notes
pandas	2.2.2	Compatible with Python 3.10 & NumPy 1.26
numpy	1.26.4	Stable with pandas 2.x and scikit-learn
nltk	3.9.1	For tokenization, lemmatization, stopwords
spacy	3.7.5	For lemmatization; works with Python 3.10
beautifulsoup4	4.12.3	Optional: text cleaning (HTML stripping)
regex	2024.5.15	Efficient regex for preprocessing

🧮 2. Traditional ML Models
Library	Version	Reason / Notes
scikit-learn	1.4.2	Stable and compatible with NumPy 1.26
scipy	1.13.1	Required by scikit-learn
joblib	1.4.2	For model saving/loading
xgboost	2.1.1	Optional: additional boosting baseline

🤖 3. Deep Learning (Transformers)
Library	Version	Reason / Notes
torch	2.3.1	Compatible with transformers 4.44
transformers	4.44.2	Stable for BERT/RoBERTa/DistilBERT
datasets	2.21.0	For loading and managing review datasets
evaluate	0.4.3	For metrics (accuracy, F1, etc.)
accelerate	0.31.0	For fine-tuning models efficiently
sentencepiece	0.2.0	Tokenizer dependency for some models
protobuf	4.25.3	Required by transformers + PyTorch

🧠 4. Generative AI (Summarization)
Library	Version	Reason / Notes
transformers	same as above	Used for summarization models
torch	same as above	Backend for generation
tqdm	4.66.5	Progress bars for generation/fine-tuning

📊 5. Visualization & Dashboard
Library	Version	Reason / Notes
plotly	5.24.0	For interactive charts/dashboards
dash	2.18.2	If you choose a web-based dashboard
matplotlib	3.9.2	For static visualizations
seaborn	0.13.2	For confusion matrices and plots
tableau-api-lib	0.1.15	(Optional) To connect to Tableau Server

🧾 6. System & Dev Utilities
Library	Version	Reason / Notes
python-dotenv	1.0.1	Manage environment variables
notebook	7.2.2	For Jupyter notebooks
ipykernel	6.29.5	Kernel support
rich	13.8.1	Beautiful terminal outputs
pyyaml	6.0.2	Configuration handling

✅ Step 3. Example requirements.txt
You can copy-paste this directly and install via:
pip install -r requirements.txt

# Core
pip==24.2
setuptools==75.1.0
wheel==0.44.0

# Data
pandas==2.2.2
numpy==1.26.4
nltk==3.9.1
spacy==3.7.5
beautifulsoup4==4.12.3
regex==2024.5.15

# Traditional ML
scikit-learn==1.4.2
scipy==1.13.1
joblib==1.4.2
xgboost==2.1.1

# Deep Learning / Transformers
torch==2.2.2
transformers==4.44.2
datasets==2.21.0
evaluate==0.4.3
accelerate==0.31.0
sentencepiece==0.2.0
protobuf==4.25.3
tqdm==4.66.5

# Visualization
plotly==5.24.0
dash==2.18.2
matplotlib==3.9.2
seaborn==0.13.2
tableau-api-lib==0.1.15

# Dev utilities
python-dotenv==1.0.1
notebook==7.2.2
ipykernel==6.29.5
rich==13.8.1
pyyaml==6.0.2


✅ Step 4. Virtual Environment Setup
1. Create environment
python3.10 -m venv nlp_reviews_env
2. Activate
macOS / Linux:
source nlp_reviews_env/bin/activate

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


