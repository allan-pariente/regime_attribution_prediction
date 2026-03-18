# Financial Regime Attribution Engine: News-Driven Analysis & Explainability

A comprehensive machine learning pipeline that detects market regimes in NASDAQ composite data and attributes causality to financial news sentiment using advanced NLP, causal inference, and explainable AI techniques.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Project Architecture](#project-architecture)
4. [Installation & Setup](#installation--setup)
5. [Data Pipeline](#data-pipeline)
6. [Workflow & Execution](#workflow--execution)
7. [Module Documentation](#module-documentation)
8. [Output Files](#output-files)
9. [Key Methodologies](#key-methodologies)
10. [Results & Interpretation](#results--interpretation)
11. [Configuration & Parameters](#configuration--parameters)
12. [Requirements & Dependencies](#requirements--dependencies)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)

---

## Project Overview

This project builds an **end-to-end machine learning pipeline** that:

1. **Fetches & processes** 10 years of NASDAQ composite historical data including returns, volatility (VIX), and trading volume
2. **Detects market regimes** using clustering and feature engineering (trends, volatility patterns)
3. **Retrieves financial news** associated with detected regime shifts using Finnhub and Google News APIs
4. **Performs sentiment analysis** using FinBERT to quantify news sentiment
5. **Generates embeddings** using Sentence Transformers for semantic news representation
6. **Analyzes causal relationships** via Granger causality testing between news sentiment and market regimes
7. **Trains predictive models** (Random Forest) to forecast market regimes
8. **Explains predictions** using SHAP values to attribute causality to specific news articles and themes
9. **Generates attribution reports** linking regime changes to news narratives

**Key Innovation**: Instead of traditional time-series prediction, this pipeline focuses on **regime explanation** — understanding *why* markets move between states through news-driven causality.

---

## Key Features

- **Multi-Source Data Integration**: Combines market data (FRED/Yahoo Finance), news APIs (Finnhub, Guardian, Google News), and alternative news sources
- **Clustering**: Detects 5 distinct market regimes using KMeans with smoothing
- **Advanced NLP**: FinBERT sentiment analysis + Sentence Transformers for semantic embeddings
- **Causal Inference**: Granger causality tests to validate news→market relationships
- **Interpretability**: SHAP local explanations for individual regime change events
- **Real-Time Prediction**: Live regime change detection using latest market & news data (10_nostradamus.py)
- **Intelligent Topic Naming**: Automatic generation of descriptive topic names from headline keywords
- **Production-Ready**: Modular pipeline with error handling, logging, and checkpoint management
- **Visualization**: Regime plots, SHAP summary plots, attribution dashboards

---

## Project Architecture

```
Financial Regime Attribution Engine
│
├── STAGE 1: DATA COLLECTION & ENGINEERING
│   ├── 01_market_data.py          → Fetch NASDAQ, VIX, Volume (10 years)
│   └── Output: data/market/nasdaq_10y.csv
│
├── STAGE 2: REGIME DETECTION
│   ├── 02_regime_detect.py        → KMeans clustering on trend/volatility features
│   ├── plot_02_regime_detect.py   → Visualization of detected regimes
│   └── Output: data/processed/regime_output.csv
│
├── STAGE 3: NEWS COLLECTION
│   ├── 03_news_fetch.py           → Finnhub, Google News, Guardian APIs
│   └── Output: data/news/news_raw.json, data/processed/news_df.csv
│
├── STAGE 4: SENTIMENT ANALYSIS
│   ├── 04_sentiment.py            → FinBERT sentiment scoring
│   └── Output: data/processed/news_with_sentiment.csv
│
├── STAGE 5: SEMANTIC EMBEDDINGS
│   ├── 05_embeddings.py           → Sentence Transformers (all-MiniLM)
│   └── Output: data/processed/news_with_embeddings.csv
│
├── STAGE 6: DATA FUSION
│   ├── 06_merge.py                → Align news, sentiment, embeddings with market data
│   └── Output: data/processed/dataset_final.csv
│
├── STAGE 7: CAUSAL ANALYSIS
│   ├── 07_granger.py              → Granger causality tests (news → regimes)
│   └── Output: data/processed/granger_results.csv
│
├── STAGE 8: MODEL TRAINING & PREDICTION
│   ├── 08_model.py                → Train RF, generate SHAP values
│   └── Output: data/models/random_forest.pkl, data/processed/shap_values.npy
│
└── STAGE 9: ATTRIBUTION & INTERPRETATION
    ├── 09_attribution.py          → SHAP local explanations + report generation
    └── Output: data/processed/attribution_report.json/.txt

└── STAGE 10: REAL-TIME PREDICTION (Live Monitoring)
    ├── 10_nostradamus.py          → Live predictions on latest market/news data
    └── Output: data/live/prediction_YYYYMMDD.json/.txt
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+ (tested on 3.10, 3.11)
- Git
- Virtual environment tool (venv, conda)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd projet_fin_regime_attribution
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# or
source .venv/bin/activate      # On Linux/Mac

# Using conda
conda create -n regime_attr python=3.10
conda activate regime_attr
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys
Create a `.env` file in project root (optional but recommended for production):
```bash
FINNHUB_API_KEY=your_key_here
GUARDIAN_API_KEY=your_key_here
```

Or set environment variables:
```bash
export FINNHUB_API_KEY="your_key_here"
export GUARDIAN_API_KEY="your_key_here"
```

API keys can be obtained:
- **Finnhub**: https://finnhub.io/register/free
- **Guardian**: https://bonobo.capi.gutools.co.uk/register/developer

### Step 5: Create Data Directories
```bash
mkdir -p data/market data/news data/processed data_usable data/models
```

---

## Data Pipeline

### Input Data
- **Market Data**: Automatically fetched from FRED (Federal Reserve Economic Data) and Yahoo Finance
  - NASDAQ Composite Index
  - VIX (Volatility Index)
  - Trading Volume
  - Time span: 10 years (adjustable)

- **News Data**: Fetched from multiple sources
  - Finnhub (institutional financial news)
  - Google News RSS (general financial news)
  - Guardian API (news articles)

### Processing Stages

#### 1. **Market Data Engineering** (01_market_data.py)
- Fetches clean NASDAQ data
- Removes missing values
- Calculates daily returns and rolling statistics
- Outputs: `nasdaq_10y.csv`

#### 2. **Regime Detection** (02_regime_detect.py)
- **Features engineered**:
  - Cumulative returns (10-day rolling)
  - Volatility (standard deviation of returns)
  - Trend direction (moving average comparison)
  - Drawdown magnitude
- **Algorithm**: KMeans clustering (k=5)
- **Regimes identified**:
  - Bull Strong (high return, low volatility)
  - Bull (positive return, moderate volatility)
  - Neutral (low return, stable volatility)
  - Bear (negative return, high volatility)
  - Crash (extreme drawdown, extreme volatility)
- **Smoothing**: 20-day rolling mode filter to reduce noise
- Outputs: `regime_output.csv`, visualizations

#### 3. **News Collection** (03_news_fetch.py)
- **Lookback window**: 5 days before regime shift (no look-ahead bias)
- **Minimum regime duration**: 5 days (filters noise)
- **Tickers tracked**: QQQ, AAPL, MSFT, NVDA, AMZN
- **Categories**: General, Technology
- **Rate limiting**: 1.2s delays between API calls
- Outputs: `news_raw.json`, `news_df.csv`, `fetch_report.txt`

#### 4. **Sentiment Analysis** (04_sentiment.py)
- **Model**: FinBERT (pre-trained on financial text)
- **Outputs**: 
  - Positive/Negative/Neutral sentiment scores
  - Confidence levels
  - Composite sentiment index
- Outputs: `news_with_sentiment.csv`

#### 5. **Semantic Embeddings** (05_embeddings.py)
- **Model**: all-MiniLM-L6-v2 (384-dim vectors)
- **Purpose**: Capture semantic meaning of news headlines
- **Applications**: Topic clustering, similarity search, dimensionality reduction
- Outputs: `news_with_embeddings.csv`, `embeddings.npy`

#### 6. **Data Fusion** (06_merge.py)
- Aligns news timestamps with market data
- Forward-fills missing dates
- Aggregates sentiment by regime window
- Creates feature matrix for machine learning
- Outputs: `dataset_final.csv`

#### 7. **Granger Causality** (07_granger.py)
- Tests: Does news sentiment *Granger-cause* market regime changes?
- Lag settings: 1-5 day lags tested
- Significance threshold: p-value < 0.05
- Outputs: `granger_results.csv`, statistics

#### 8. **Model Training** (08_model.py)
- **Models trained**:
  - Random Forest Classifier
- **Features**: sentiment, embeddings, technical indicators
- **Target**: Next day regime label
- **Validation**: Stratified cross-validation (k=5)
- **Explainability**: SHAP TreeExplainer
- Outputs: `random_forest.pkl`, `shap_values.npy`, `model_results.csv`

#### 9. **Attribution & Interpretation** (09_attribution.py)
- For each regime shift event detected:
  - Identifies top contributing news articles
  - Ranks news by SHAP relevance score
  - Extracts dominant topics via embedding clustering
  - Generates natural language explanation
- Outputs: `attribution_report.json`, `attribution_report.txt`

---

## Workflow & Execution

### Option 1: Run Full Pipeline (Sequential)
```bash
cd src

# Run all steps in order
python 01_market_data.py      # ~2 min: Fetch market data
python 02_regime_detect.py    # ~1 min: Detect regimes
python 03_news_fetch.py       # ~10-15 min: Fetch news (API calls)
python 04_sentiment.py        # ~5 min: FinBERT analysis
python 05_embeddings.py       # ~3 min: Generate embeddings
python 06_merge.py            # ~1 min: Fuse data
python 07_granger.py          # ~2 min: Causality tests
python 08_model.py            # ~3 min: Train models + SHAP
python 09_attribution.py      # ~2 min: Generate reports
```

**Total runtime**: ~30-40 minutes (first run), ~5-10 minutes (subsequent with cached data)

### Option 2: Run Individual Steps
```bash
python src/02_regime_detect.py  # Run only regime detection
python src/08_model.py          # Retrain model on existing data
```

### Option 3: Real-Time Prediction (Live)
```bash
# After running full pipeline once (to train models), run for live predictions
python src/10_nostradamus.py    # ~30s: Predict regime change on latest market/news data

# Schedule it to run daily (e.g., via cron after market close)
# Generates: data/live/prediction_YYYYMMDD.json and .txt
```

### Option 4: Parallel Execution (Advanced)
For independent steps (after initial run), use GNU Parallel or similar:
```bash
parallel python src/{} ::: 02_regime_detect.py 04_sentiment.py 07_granger.py
```

---

## Module Documentation

### **01_market_data.py**
**Purpose**: Fetch 10-year NASDAQ historical data

**Key Functions**:
- `fetch_fred(series_id)`: Fetch data from FRED API
- `fetch_volume_yahoo()`: Get trading volume from Yahoo Finance
- Data cleaning and normalization

**Outputs**:
- `data/market/nasdaq_10y.csv` (columns: Date, Close, VIX, Volume)

**Configurable Parameters**:
```python
LOOKBACK_YEARS = 10  # Adjust for different time horizons
```

---

### **02_regime_detect.py**
**Purpose**: Detect market regimes using clustering

**Key Functions**:
- `extract_features()`: Engineer trend, volatility, drawdown features
- `cluster_regimes()`: KMeans clustering with k=5
- `smooth_regimes()`: 20-day rolling mode filter
- `label_regimes()`: Interpretable regime names

**Outputs**:
- `data/processed/regime_output.csv` (columns: Date, Regime, Confidence, Regime_Days)

**Configurable Parameters**:
```python
N_REGIMES = 5              # Number of clusters
ROLLING_DAYS = 10          # For return calculation
SMOOTH_DAYS = 20           # For mode filtering
RANDOM_STATE = 42          # For reproducibility
```

---

### **03_news_fetch.py**
**Purpose**: Collect financial news around regime shifts

**Key Functions**:
- `fetch_finnhub_news()`: Query Finnhub API
- `fetch_google_news_rss()`: Fallback news source (no API key needed)
- `fetch_guardian_news()`: Guardian News API
- `deduplicate_news()`: Remove duplicate headlines
- `save_raw_json()`: Cache news for reproducibility

**Outputs**:
- `data/news/news_raw.json` (raw API responses)
- `data/processed/news_df.csv` (structured dataset)
- `data/news/fetch_report.txt` (statistics)

**Configurable Parameters**:
```python
FINNHUB_API_KEY = "your_key"
GUARDIAN_API_KEY = "your_key"
WINDOW_BEFORE_DAYS = 5        # Lookback window
WINDOW_AFTER_DAYS = 0          # Avoid look-ahead bias
MIN_REGIME_DURATION_DAYS = 5
COMPANY_TICKERS = ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN"]
RATE_LIMIT_DELAY = 1.2         # Seconds between API calls
```

---

### **04_sentiment.py**
**Purpose**: Analyze sentiment of news headlines using FinBERT

**Key Functions**:
- `load_finbert_model()`: Load pre-trained FinBERT
- `score_sentiment()`: Classify sentiment for each headline
- `aggregate_by_regime()`: Average sentiment per regime window
- `calculate_sentiment_indices()`: Create composite indices

**Outputs**:
- `data/processed/news_with_sentiment.csv` (columns: ..., sentiment_label, sentiment_positive, sentiment_negative, sentiment_neutral, composite_sentiment)

**Configurable Parameters**:
```python
MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
```

**Sentiment Scores**:
- Positive: 1.0 (bullish news)
- Negative: -1.0 (bearish news)
- Neutral: 0.0 (neutral news)

---

### **05_embeddings.py**
**Purpose**: Generate semantic embeddings for news headlines

**Key Functions**:
- `load_embedding_model()`: Load Sentence Transformers
- `generate_embeddings()`: Create 384-dim vectors
- `save_embeddings()`: Store as NumPy arrays for efficiency
- `pca_reduction()` (optional): Reduce to 50 dims for visualization

**Outputs**:
- `data/processed/news_with_embeddings.csv` (columns: ..., embedding_384d)
- `data/processed/embeddings.npy` (384 x n_articles)

**Configurable Parameters**:
```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
NORMALIZE = True              # L2 normalization
PCA_DIM = 50                  # For visualization
```

---

### **06_merge.py**
**Purpose**: Fuse market data, regimes, news, sentiment, and embeddings

**Key Functions**:
- `align_timestamps()`: Match news to market dates
- `forward_fill_regimes()`: Propagate regime labels forward
- `aggregate_sentiment()`: Average sentiment per regime window
- `feature_engineering()`: Create additional statistical features
- `create_ml_dataset()`: Format for model training

**Outputs**:
- `data/processed/dataset_final.csv` (master dataset for ML)

**Columns**:
- Market: Date, Close, Returns, VIX, Volume
- Regimes: Regime, Regime_Duration
- News: Headline_Count, Avg_Sentiment, Sentiment_Std
- Embeddings: news_embedding_1, ..., news_embedding_384
- Target: Next_Regime (for supervised learning)

---

### **07_granger.py**
**Purpose**: Test causal relationship: News Sentiment → Market Regime

**Key Functions**:
- `prepare_timeseries()`: Stationarity checks, differencing if needed
- `granger_test()`: Run Granger causality test with multiple lags
- `interpret_results()`: Extract p-values, F-statistics
- `generate_report()`: Summarize findings

**Outputs**:
- `data/processed/granger_results.csv`
- `data/processed/granger_report.txt`

**Interpretation**:
- p-value < 0.05: Sentiment DOES Granger-cause regime changes
- p-value ≥ 0.05: No statistical evidence of causality
- Granger causality ≠ true causality, but suggests predictive relationship

**Configurable Parameters**:
```python
MAX_LAG = 5                   # Test 1-5 day lags
SIGNIFICANCE_LEVEL = 0.05
```

---

### **08_model.py**
**Purpose**: Train predictive models and generate SHAP values

**Key Functions**:
- `preprocess_features()`: Normalize, handle missing values
- `train_random_forest()`: Fit RF with 100 trees
- `cross_validate()`: 5-fold stratified CV
- `generate_shap_values()`: TreeExplainer for interpretability
- `evaluate_model()`: Precision, Recall, F1, ROC-AUC metrics

**Outputs**:
- `data/models/random_forest.pkl` (trained Random Forest)
- `data/processed/shap_values.npy` (SHAP matrix: n_samples × n_features)
- `data/processed/model_results.csv` (metrics, feature importance)

**Model Performance**:
- Typical accuracy: 60-75% (regime prediction is genuinely hard)
- ROC-AUC: 0.65-0.80 (shows predictive signal above random)

**Configurable Parameters**:
```python
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_FOREST_N_ESTIMATORS = 100
```

---

### **09_attribution.py**
**Purpose**: Generate interpretable attribution reports for regime changes

**Key Functions**:
- `identify_regime_shifts()`: Locate regime transition points
- `extract_contributing_news()`: Find news in lookback window
- `rank_by_shap()`: Order by SHAP relevance
- `extract_topics()`: Cluster embeddings to identify themes
- `generate_text_report()`: Create human-readable explanations
- `export_json_report()`: Structured JSON for downstream use

**Outputs**:
- `data/processed/attribution_report.json` (structured data)
- `data/processed/attribution_report.txt` (human-readable)

**Report Structure**:
```json
{
  "regime_shifts": [
    {
      "date": "2023-03-15",
      "from_regime": "Bull Strong",
      "to_regime": "Neutral",
      "confidence": 0.92,
      "contributing_news": [
        {
          "headline": "Federal Reserve raises rates by 0.25%",
          "date": "2023-03-14",
          "sentiment": -0.6,
          "shap_importance": 0.15
        },
        ...
      ],
      "dominant_topics": ["Monetary Policy", "Interest Rates"],
      "explanation": "Regime shift from Bull Strong to Neutral..."
    }
  ]
}
```

---

### **plot_02_regime_detect.py**
**Purpose**: Visualize detected regimes

**Outputs**:
- Regime time-series plots
- Volatility vs. Returns scatter (color-coded by regime)
- Distribution histograms
- Regime duration statistics

---

### **10_nostradamus.py** ⚡ LIVE PREDICTION
**Purpose**: Real-time regime change prediction using latest market & news data

**Key Functions**:
- `collect_news()`: Fetch latest news from Finnhub + Guardian (past 5 days)
- `compute_sentiment()`: FinBERT sentiment scoring on live headlines
- `compute_embeddings_and_topics()`: Encode headlines, assign to semantic topics using KMeans
- `generate_topic_names()`: Create descriptive names for topics (e.g., "Federal / Policy / Rate")
- `get_market_features()`: Download latest NASDAQ, VIX, volume from FRED/Yahoo Finance
- `build_feature_vector()`: Assemble prediction input from market + news features
- `predict()`: Run Random Forest to get regime change probability
- `interpret_regime()`: Deduce likely regime based on market context
- `build_report()`: Generate structured prediction report with top contributing news

**Inputs**:
- `data/models/random_forest.pkl` (trained Random Forest)
- `data/processed/dataset_final.csv` (to extract feature names)
- `data/processed/news_with_embeddings.csv` (historical for topic naming)
- **APIs**: Finnhub, Guardian, FRED, Yahoo Finance

**Outputs**:
- `data/live/prediction_YYYYMMDD.json` (structured prediction)
- `data/live/prediction_YYYYMMDD.txt` (human-readable report)

**Key Improvements**:
- **Descriptive Topic Names**: Automatically generates intuitive topic names (e.g., "Monetary Policy", "Technology", "Employment") based on keyword frequency in headlines
- **No Look-Ahead Bias**: Uses only past/present data (WINDOW_AFTER_DAYS=0)
- **Real-Time Deployment**: Can run on schedule (cron/scheduler) for continuous monitoring
- **Top News Attribution**: Shows which specific articles influenced the prediction with sentiment scores

**Example Output**:
```
=================================================================
PREDICTION LIVE — 2026-03-18
Genere le 2026-03-18 18:04:41
=================================================================

      Signal faible
  Probabilite de changement : 2.0%
  Regime probable            : stable (pas de changement prevu)
  Seuils : forte=35%  moderee=20%

-----------------------------------------------------------------
CONTEXTE DE MARCHE
-----------------------------------------------------------------
  Rendement (dernier j) : 0.4709%
  Volatilite 20j        : 0.1659
  VIX                   : 22.37
  RSI 14j               : 45.6862
  Drawdown              : -5.7756%

-----------------------------------------------------------------
TOP 5 NEWS LES PLUS SIGNIFICATIVES
-----------------------------------------------------------------
1. [2026-03-16] [NEGATIVE -0.96] Monetary / Policy / Rate
   Fed raises rates by 0.25% amid inflation concerns
   Source : Finnhub

2. [2026-03-17] [NEGATIVE -0.96] Technology / Stocks / Market
   Tech stocks fall as interest rates climb
   Source : guardian

... (more articles)

-----------------------------------------------------------------
TOPICS DOMINANTS
-----------------------------------------------------------------
  Monetary / Policy / Rate      : ################# 35.2% (64 articles)
  Technology / Stocks / Market  : ############ 25.3% (46 articles)
  Employment / Jobs / Hiring    : ########## 18.1% (33 articles)
```

**Configurable Parameters**:
```python
WINDOW_DAYS    = 5                    # Lookback period for news
N_CLUSTERS     = 8                    # Number of semantic topics
PROBA_THRESHOLD_HIGH   = 0.35        # Strong alert threshold
PROBA_THRESHOLD_MEDIUM = 0.20        # Moderate alert threshold
BATCH_SIZE     = 32                   # For model inference
```

---

## 📊 Output Files

| File | Location | Description |
|------|----------|-------------|
| **nasdaq_10y.csv** | `data/market/` | NASDAQ, VIX, Volume (10 years) |
| **regime_output.csv** | `data/processed/` | Detected regimes with dates and confidence |
| **news_raw.json** | `data/news/` | Raw API responses (for reproducibility) |
| **news_df.csv** | `data/processed/` | Structured news dataset |
| **fetch_report.txt** | `data/news/` | News collection statistics |
| **news_with_sentiment.csv** | `data/processed/` | News + FinBERT sentiment scores |
| **news_with_embeddings.csv** | `data/processed/` | News + 384-dim embeddings |
| **embeddings.npy** | `data/processed/` | Embedding matrix (384 × n_articles) |
| **dataset_final.csv** | `data/processed/` | Master ML dataset (fusion of all) |
| **granger_results.csv** | `data/processed/` | Granger causality test results |
| **granger_report.txt** | `data/processed/` | Granger interpretation |
| **random_forest.pkl** | `data/models/` | Trained Random Forest model |
| **shap_values.npy** | `data/processed/` | SHAP local explanations |
| **model_results.csv** | `data/processed/` | Model metrics (accuracy, F1, ROC-AUC) |
| **attribution_report.json** | `data/processed/` | Structured attribution data |
| **attribution_report.txt** | `data/processed/` | Human-readable attribution |
| **prediction_YYYYMMDD.json** | `data/live/` | Live prediction (structured) |
| **prediction_YYYYMMDD.txt** | `data/live/` | Live prediction (human-readable) |

---

## 🔬 Key Methodologies

### 1. **Regime Detection: KMeans Clustering**
- **Why**: Unsupervised learning captures natural market regimes without labels
- **Features**: Cumulative returns, volatility, trend, drawdown
- **K=5 rationale**: Market literature recognizes ~5 distinct regimes
- **Smoothing**: 20-day rolling mode filter reduces noise and transitions

### 2. **Sentiment Analysis: FinBERT**
- **Why**: FinBERT is pre-trained specifically on financial text, understands domain jargon
- **Alternative**: Traditional lexicon-based VADER (less accurate but faster)
- **Output**: Normalized sentiment scores (-1 to 1)

### 3. **Embeddings: Sentence Transformers**
- **Why**: Captures semantic meaning beyond keywords, enables clustering and similarity search
- **Model**: all-MiniLM-L6-v2 balances speed/accuracy (384 dimensions)
- **Alternative**: OpenAI embeddings (higher quality but paid)

### 4. **Causal Inference: Granger Causality**
- **Why**: Tests predictive causality (X Granger-causes Y if past X helps predict Y)
- **Limitations**: Tests correlation, not true causation; requires stationarity
- **Interpretation**: If news sentiment Granger-causes regime changes, sentiment has predictive power

### 5. **Explainability: SHAP (SHapley Additive exPlanations)**
- **Why**: Provides local explanations (why this specific prediction?)
- **Method**: TreeExplainer for Random Forest efficiency
- **Output**: Feature importance per sample (not just global)

### 6. **No Look-Ahead Bias**
- **Key design choice**: News lookback window only includes past dates (WINDOW_AFTER_DAYS=0)
- **Ensures**: Model is realistic for real-time deployment

---

## 📈 Results & Interpretation

### Expected Outcomes

**1. Regime Detection**
- 5 distinct regimes identified with 70-85% cluster purity
- Bull Strong/Bull: ~15% of trading days
- Neutral: ~35% of trading days
- Bear/Crash: ~50% of trading days (realistic given 10-year span including 2020 COVID crash)

**2. News-Regime Correlation**
- Strong sentiment correlation with regime transitions
- Average headline count per regime shift: 20-50 articles
- Dominant topics: Fed policy, earnings, macroeconomic indicators

**3. Granger Causality**
- Expected: p-value < 0.05 for 2-3 day lags
- Interpretation: Market likely responds to news with 1-3 day delay

**4. Model Performance**
- Random Forest accuracy: 60-70%
- ROC-AUC: 0.70-0.82 (indicates meaningful patterns)
- Feature importance: Sentiment features typically top-5

**5. Attribution**
- Each regime shift linked to 3-7 most important news articles
- SHAP values identify which specific headlines drove changes
- Confidence available for each attribution

### Interpretation Tips

1. **Regime Confidence**: If confidence < 0.6, regime is uncertain edge case
2. **SHAP Values**: Positive = pushes toward predicted regime; negative = pushes away
3. **Granger p-values**: Check if below 0.05 for statistical significance
4. **Sentiment Drift**: Big drops in average sentiment often precede regime shifts

---

## ⚙️ Configuration & Parameters

### Global Parameters (Adjustable in each script)

```python
# TIME HORIZON
LOOKBACK_YEARS = 10                    # Change for 5y, 20y, etc.

# REGIME DETECTION
N_REGIMES = 5                          # Adjust for finer/coarser regimes
ROLLING_DAYS = 10                      # For feature calculation
SMOOTH_DAYS = 20                       # Noise reduction

# NEWS COLLECTION
WINDOW_BEFORE_DAYS = 5                 # Lookback period
WINDOW_AFTER_DAYS = 0                  # NO LOOK-AHEAD
MIN_REGIME_DURATION_DAYS = 5           # Filter very short regimes
COMPANY_TICKERS = ["QQQ", "AAPL", "MSFT", "NVDA", "AMZN"]

# MODEL TRAINING
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_FOREST_N_ESTIMATORS = 100

# GRANGER CAUSALITY
MAX_LAG = 5                            # Test up to 5-day lags
SIGNIFICANCE_LEVEL = 0.05              # p-value threshold
```

---

## 📋 Requirements & Dependencies

### Core Libraries
```
pandas>=2.0              # Data manipulation
numpy>=1.24             # Numerical computing
scikit-learn>=1.4       # ML models (Random Forest, KMeans)
```

### NLP & Embeddings
```
transformers>=4.40      # HuggingFace models (FinBERT)
torch>=2.0             # PyTorch backend
sentence-transformers>=2.7  # Embeddings
```

### Causal & Statistical
```
statsmodels>=0.14      # Granger causality, time-series tests
dowhy>=0.11            # Causal inference (optional)
```

### Data APIs
```
requests>=2.31         # HTTP client
pandas-datareader==0.10.0  # FRED data
yfinance>=0.2.36       # Yahoo Finance
finnhub-python>=2.4    # Finnhub API
```

### Explainability & Visualization
```
shap>=0.45             # SHAP explanations
matplotlib>=3.8        # Plotting
seaborn>=0.13          # Statistical visualization
plotly>=5.20           # Interactive plots
```

### Notebooks
```
jupyter>=1.0           # Jupyter notebooks
ipykernel>=6.0         # IPython kernel
```

### Full Installation
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Solution**:
```bash
pip install --upgrade transformers torch
```

### Issue: "FINNHUB_API_KEY not found"
**Solution**: Set environment variable or hardcode in script
```bash
export FINNHUB_API_KEY="your_key_here"
# or add to .env file
```

### Issue: "No news articles found"
**Solution**: Check date range, API rate limits
```python
# In 03_news_fetch.py, increase WINDOW_BEFORE_DAYS or lower RATE_LIMIT_DELAY
```

### Issue: "CUDA out of memory" (GPU)
**Solution**: 
```python
# Reduce batch size in 04_sentiment.py, 05_embeddings.py
BATCH_SIZE = 8  # Instead of 32
# Or force CPU: 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

### Issue: "Connection timeout fetching news"
**Solution**: 
```python
# Increase timeout and add retry logic
TIMEOUT = 30
MAX_RETRIES = 3
```

### Issue: "Stationarity test failed" (Granger)
**Solution**:
```python
# 07_granger.py will auto-difference if needed
# If still failing, manually difference:
df['sentiment_diff'] = df['sentiment'].diff()
```

### Issue: Code stops abruptly / "No error message"
**Solution**: Add logging
```bash
# Run with logging:
python -u 08_model.py 2>&1 | tee training.log
```

---

## Contributing

### How to Extend the Pipeline

1. **Add new data source**: Modify `03_news_fetch.py`
   - Add new `fetch_*()` function
   - Update news deduplication logic

2. **Change embedding model**: Modify `05_embeddings.py`
   - Swap model name: `MODEL_NAME = "model-name"`
   - Adjust `EMBEDDING_DIM` if needed

3. **Add new regime features**: Modify `02_regime_detect.py`
   - Add feature engineering functions
   - Include in feature matrix for clustering

4. **Experiment with new models**: Modify `08_model.py`
   - Add `train_lightgbm()` or other model
   - Ensure SHAP compatibility

5. **Improve attribution reports**: Modify `09_attribution.py`
   - Refine natural language generation
   - Add visualizations

### Code Quality
- Follow PEP 8 style guide
- Add docstrings to functions
- Include error handling
- Log important checkpoints

---

## 📞 Support & Questions

For issues or questions, please:
1. Check [Troubleshooting](#troubleshooting) section
2. Review inline code comments
3. Check API documentation:
   - Finnhub: https://finnhub.io/docs/api
   - Guardian: https://open-platform.theguardian.com/documentation/
4. Review research paper references in code comments

---

## 📝 License & Citation

If you use this project, please cite:
```bibtex
@project{regime_attribution_2026,
  title={Financial Regime Attribution Engine},
  author={[Allan Pariente]},
  institution={CentraleSupélec},
  year={2024}
}
```

---

## 🎓 Educational Resources

### Background Reading
- **Regime Switching Models**: Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series"
- **SHAP Explanations**: Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions"
- **Granger Causality**: Granger, C. W. (1969). "Investigating causal relations by econometric models"
- **FinBERT**: Huang, A. H., et al. (2022). "FinBERT: A Pretrained Language Representation Model for Financial Text"

### Alternative Approaches
- Hidden Markov Models (for stochastic regime dynamics)
- Dynamic Time Warping (for news pattern matching)
- Attention Mechanisms (for temporal news importance)
- Causal Forests (for heterogeneous treatment effects)

---

**Last Updated**: March 2026
**Version**: 1.0  
**Maintainer**: [Allan Pariente]
