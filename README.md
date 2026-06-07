# End-to-End AI Stock Analyst: Sentiment-Assisted BBCA Prediction

Welcome to the repository for my undergraduate thesis project. This repository establishes a production-grade, end-to-end Machine Learning Operations (MLOps) pipeline designed to predict the daily stock price movement of Bank Central Asia (BBCA.JK). 

The core objective of this research is to prove whether or not incorporating deep-learning-derived news sentiment scoring is able to increase a model's prediction accuracy and directional win-rate relative to traditional numerical-only baselines.

## 📊 BILSTM Sentiment-Assisted Price Prediction
<img width="1376" height="907" alt="image" src="https://github.com/user-attachments/assets/2e1cf74d-d0e0-4b9c-84a4-35c9f4ec4019" />

---

## 🧠 System Architecture & Core Workflow Logic

The pipeline operates as a synchronized "Two-Engine" architecture with explicit temporal safety guards:

1. **The Sentiment Engine (Continuous NLP Optimization):** The pipeline pulls live Indonesian news via Google RSS queries. Headlines are translated from Indonesian to English and passed to **FinBERT**. Instead of rounding predictions into coarse categorical integers ($1, 0, -1$), this system extracts raw, continuous probability distributions directly from the model's **Softmax** layer:
   $$\text{Sentiment Score} = \text{Probability(Positive)} - \text{Probability(Negative)}$$
   This preserves the high-fidelity nuances of financial text, yielding a smooth decimal scalar between $-1.0$ and $+1.0$.

2. **The Forecasting Engine (Deep Learning Time-Series):** A 2-layer Bidirectional Long Short-Term Memory (BiLSTM) network ingests a 10-day sliding temporal window containing historical stock features (Open, High, Low, Close, Volume) matched with our continuous sentiment features to forecast the next business day's trend.

3. **Temporal Integrity Constraints:** * **The Lag Rule:** To prevent **look-ahead data leakage**, all sentiment scores are strictly lagged by one operational cycle ($T-1$). Today’s market close ($T$) is mapped directly to yesterday's news sentiment.
   * **The Weekend Cushion:** Because the Jakarta Stock Exchange (IDX) closes on weekends but the media loop does not, the pipeline dynamically scales its collection look-back window on Mondays to **3 days (`when:3d`)** to aggregate Friday night, Saturday, and Sunday news into a single weekend data payload.

---

## 🗂️ Repository & Notebook Structure

To execute the pipeline or replicate our thesis benchmarks, run the notebooks in strict numerical order (**1 → 2 → 3**). Use **Notebook 4** for data maintenance when needed, and refer to `Models_Comparison.ipynb` for the baseline architectural experiments.

Click any badge below to instantly launch the notebook in a live Google Colab environment:

### 1. `1_DataEngineering_&_FinbertSentiment.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/BiLSTMvsBiLSTMSentiment/blob/main/1_DataEngineering_&_FinbertSentiment.ipynb)
* **Purpose:** The Data Refinery Pipeline.
* **What it does:** Processes raw historical Indonesian news and structural stock CSV files. It cleans numerical volume string notations (e.g., matching "193M" to `193,000,000`), translates text headlines, computes daily continuous FinBERT scores, applies the $T-1$ lag mapping, and outputs a uniform historical master database.
* **Inputs:** `GoogleNews_BBCA.csv`, `BBCA_Stock_History.csv`
* **Outputs:** `BBCA_Master_Dataset_BiLSTM.csv`

### 2. `2_BiLSTM_Training.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/BiLSTMvsBiLSTMSentiment/blob/main/2_BiLSTM_Training.ipynb)
* **Purpose:** The Laboratory (Production Model Export).
* **What it does:** Fits the winning, enhanced BiLSTM architecture against the master database. It normalizes values via `MinMaxScaler` arrays, structures the 15-day chronological sequence windows, trains the deep learning network layers, and exports the serialized model binaries.
* **Inputs:** `BBCA_Master_Dataset_BiLSTM.csv`
* **Outputs:** `BBCA_BiLSTM_Sentiment_Model.keras`, `scaler_features.pkl`, `scaler_target.pkl`

### 3. `3_Live_Model_BiLSTM.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/BiLSTMvsBiLSTMSentiment/blob/main/3_Live_Model_BiLSTM.ipynb)
* **Purpose:** Production Deployment (Daily Live Inference & Live XAI Dashboard).
* **What it does:** Runs daily after market close. It uses `yfinance` to grab the fresh daily stock row, checks the calendar day to adjust the news collection query look-back parameter dynamically (`when:3d` on Mondays to prevent weekend data loss), scores current headlines, runs inference, and renders an **Explainable AI Dashboard** mapping micro-trends and peak confidence points.
* **Inputs:** Saved `.keras` model, both `.pkl` scalers, and `BBCA_Master_Dataset_BiLSTM.csv`.
* **Outputs:** Next-business-day trend verdict (`📈 UPTREND` / `📉 DOWNTREND`), price projection, and a live matplotlib dual-axis trend chart.

### 4. `4_Dataset_Recovery_Tool.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/BiLSTMvsBiLSTMSentiment/blob/main/4_Dataset_Recovery_Tool.ipynb)
* **Purpose:** Automated Maintenance and Cold-Start Gap Resolution.
* **What it does:** Used as an operational backup tool if the daily inference engine hasn't been run for several days. It checks the max date in the master database against the system clock, automatically calculates the gap sequence (`days_diff + 2` days to satisfy the lag constraint), downloads missing Yahoo Finance stock rows, scrapes backlogged news, parses continuous scores, and performs a surgical row upsert to heal the timeline without row duplication.
* **Inputs:** `BBCA_Master_Dataset_BiLSTM.csv`
* **Outputs:** A completely repaired, up-to-date master CSV file.

### `Models_Comparison.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/BiLSTMvsBiLSTMSentiment/blob/main/Models_Comparison.ipynb)
* **Purpose:** Controlled Hyperparameter Benchmarking & Architecture Validation.
* **What it does:** Implements a strict, reproducible environment to compare 5 distinct structural time-series models. Fairness is mathematically guaranteed by binding all architectures to an identical universal seed rule, uniform 2-layer structures, matching optimizer parameters, and a 15-day sequence window. This experiment provides empirical justification for selecting the BiLSTM as our primary forecasting model.
* **Inputs:** `BBCA_Master_Dataset_BiLSTM.csv`
* **Outputs:** Validation loss tracking curves and structural selection proof metrics.

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Local Environment Preparation
Clone this repository and ensure your environment has your dependencies ready:
```bash
git clone [https://github.com/yourusername/BiLSTMvsBiLSTMSentiment.git](https://github.com/yourusername/BiLSTMvsBiLSTMSentiment.git)
pip install feedparser deep_translator yfinance transformers torch pandas numpy matplotlib
