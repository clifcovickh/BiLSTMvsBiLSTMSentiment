# BiLSTMvsBiLSTMSentiment
# End-to-End AI Stock Analyst: BBCA Stock Prediction
Welcome to the repository for my Thesis Project. This project demonstrates a fully automated, end-to-end Machine Learning Operations (MLOps) pipeline designed to predict the daily stock price movement of Bank Central Asia (BBCA). This repository was created to provide data supporting the claim that adding sentiment scoring can improve a model's performance relative to its base version.

This system does not just look at historical numbers; it reads the news. By combining Natural Language Processing (NLP) to gauge market psychology with a Deep Learning Time-Series model, this project acts as an automated daily financial analyst.

## System Architecture & Workflow Logic
This project relies on a "Two-Engine" architecture:
1. Engine 1 (The Sentiment Engine): An automated pipeline that scrapes Indonesian news via Google RSS, uses Deep Translation to convert it to English, and feeds it into FinBERT (a Hugging Face NLP model trained explicitly on financial text) to extract an objective psychological sentiment score (-1 to 1).
2. Engine 2 (The Forecasting Engine): A Bidirectional Long Short-Term Memory (BiLSTM) Neural Network that acts as the "Brain". It ingests 10-day sliding windows of both historical stock prices and the daily FinBERT sentiment scores to forecast the next day's price trend.

## Repository Structure
To replicate this project or deploy the live analyst, the workflow is split into three distinct Jupyter/Colab Notebooks. They must be run in numerical order.

### Notebook 1: `1_DataEngineering_&_FinbertSentiment.ipynb`
**Purpose: The Factory (Data Preparation & Engineering)**
This notebook creates the "Master Textbook" that the AI will study. 
* **What it does:** * Loads  raw historical Indonesian news headlines.
  * Translates them and extracts individual FinBERT sentiment scores.
  * Aggregates the scores into a daily sentiment average.
  * Cleans human-readable volume data (e.g., "193M" to 193,000,000) from historical stock files.
  * Fuses the stock prices and daily sentiment into a single, clean timeline.
* **Input Files Required:** `GoogleNews_BBCA_2025_Lengkap.csv`, `Stock_History_BBCA.csv`
* **Output Generated:** `BBCA_Master_Dataset_BiLSTM.csv` (The Master Database)

### Notebook 2: `2_BiLSTM_Training.ipynb`
**Purpose: The Laboratory (Model Training)**
This notebook is where the Deep Learning happens.

* **What it does:**
  * Loads the `BBCA_Master_Dataset_LSTM.csv`. 
  * Scales large financial numbers down to 0-1 decimals using `MinMaxScaler`.
  * Formats the chronological data into 10-day "Sliding Windows" (Time Steps = 10).
  * Trains two separate BiLSTM models (Baseline vs. Enhanced.
  * Plot model comparison
* **Input File Required:** `BBCA_Master_Dataset_BiLSTM.csv`
* **Outputs Generated (The "Brain" & "Translators"):** * `BBCA_BiLSTM_Sentiment_Model.keras` 
  * `scaler_features.pkl` & `scaler_target.pkl` (The mathematical scaling rules)

### Notebook 3: `3_Live_Model_BiLSTM.ipynb`
**Purpose: The Deployment (Live Daily Inference)**
This is the grand finale—a lightweight deployment script meant to be run daily after the market closes.
* **What it does:**
  * Loads the saved `.keras` brain and `.pkl` scalers.
  * Scrape today's Google News RSS for BBCA.
  * Translates and scores today's news in real-time.
  * Uses the `yfinance` API to fetch today's closing stock price directly from Yahoo Finance.
  * **Upserts** (Updates/Inserts) today's data into the local Master CSV to perfectly maintain the 10-day historical memory without duplicating rows.
  * Prints a live financial forecast predicting tomorrow's market trend.
* **Input Files Required:** The 3 output files from Notebook 2, plus the Master CSV.

### Notebook 4: `4_Database_Recovery_Tool.ipynb`
**Purpose: The Emergency Backup (Cold Start Resolution)**
Because the LSTM requires a continuous 10-day rolling window, pausing the daily script for weeks will break the timeline. This tool fixes that data gap.
* **What it does:** Scrapes the past 14 days of news and stock prices, translates/scores the missing sentiment, and performs a surgical "Upsert" to heal the `BBCA_Master_Dataset_BiLSTM.csv` without duplicating rows. 
* **When to use it:** Only required to run this if the live script (Notebook 3) has not been executed for more than 2 weeks ish.

### EXTRAS 
I will provide the initial Stock_hisotry_BBCA.csv I used, which was later formatted by the `1_DataEngineering_&_FinbertSentiment.ipynb`. I will also provide the scraper script I used to obtain the Google RSS news (gooogle_news_scrapper).

### NOTES
DO Note that the dataset's sentiment column used was shifted by 1, meaning that we are using yesterday's news to then predict today's news.

## How to Use
1. Clone this repository.
2. Upload the files in the `Assets` folder (raw data) to your Google Colab or local Jupyter environment.
3. Run **Notebook 1** to generate your Master Dataset.
4. Run **Notebook 2** to train your AI and generate the `.keras` and `.pkl` files.
5. Run **Notebook 3** anytime you want a live prediction for the next trading day!

## About This Project
This repository was built as the practical codebase for an undergraduate thesis focusing on the intersection of Behavioral Finance, Natural Language Processing, and Deep Learning Time-Series Forecasting.
