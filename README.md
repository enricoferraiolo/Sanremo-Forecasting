# 🎤 Sanremo Forecasting: Predicting the Sanremo 2025 Winner (for home viewers) through Sentiment Analysis
**Predicting the Sanremo 2025 Winner for Home Viewers Using Real-Time Twitter Sentiment Analysis**

## 📊 Project Overview
This repository contains the code and documentation for **Sanremo Forecasting**, a machine learning project aimed at **predicting the televote winner** of Italy's
**Sanremo Music Festival 2025** by performing **real-time sentiment analysis** on Twitter (X.com) posts. The core idea is to scrape tweets about each artist during the live broadcast, classify their sentiment (positive or negative), and use an LSTM-based model to forecast which artist will top the public televote.

## 📁 Repository Structure
```
.
├── .gitignore
├── README.md
├── report/                  # Contains the final project report (main.tex, main.pdf, bib.bib, media/)
├── requirements_macos.txt   # Python dependencies for macOS
├── requirements_ubuntu.txt  # Python dependencies for Ubuntu (CPU)
├── requirements_ubuntu_cuda.txt # Python dependencies for Ubuntu (NVIDIA GPU with CUDA)
└── src/
    └── notebooks/
        ├── constants.py     # Defines artists, festival nights, and other constants
        ├── utils.py         # Utility functions for data loading, processing, and prediction
        ├── tweet_scraper.ipynb # Jupyter notebook for scraping tweets
        ├── tweet_analysis.ipynb # Jupyter notebook for sentiment analysis of scraped tweets
        ├── early_prediction.ipynb # Jupyter notebook for training and evaluating the prediction model
        ├── scraped_data/    # Directory where raw scraped tweets are stored
        │   └── {Night_Name}/
        │       └── {Artist_Name}/
        │           └── YYYY-MM-DD_HH-HH.csv
        ├── analyzed_data/   # Directory where sentiment-analyzed tweets are stored
        │   └── {Night_Name}/
        │       └── {Artist_Name}/
        │           └── YYYY-MM-DD_HH-HH.csv
        └── predictions/     # Directory for saving prediction plots
```

## ⚙️ Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sanremo-Forecasting
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**
   Choose the appropriate requirements file based on your operating system and hardware:
   - For macOS:
     ```bash
     pip install -r requirements_macos.txt
     ```
   - For Ubuntu (CPU):
     ```bash
     pip install -r requirements_ubuntu.txt
     ```
   - For Ubuntu (NVIDIA GPU with CUDA):
     ```bash
     pip install -r requirements_ubuntu_cuda.txt
     ```
4. **Set up environment variables**
   - Create a `.env` file in the root directory (`Sanremo-Forecasting/.env`) and add your Twitter credentials:
     ```env
     TWITTER_USERNAME="your_twitter_username"
     TWITTER_PASSWORD="your_twitter_password"
     ```
   This is used by the [`TwitterScraper`](src/notebooks/tweet_scraper.ipynb) for logging into Twitter.

## 🗃️ Data Collection
Tweet collection is performed by the [`TwitterScraper`](src/notebooks/tweet_scraper.ipynb) class within the [src/notebooks/tweet_scraper.ipynb](src/notebooks/tweet_scraper.ipynb) notebook.

- **Scraper Implementation**:
    - Uses `selenium` with `webdriver_manager` to control a Chrome browser instance.
    - Simulates human-like behavior to navigate Twitter and search for tweets.
    - Handles login using credentials from the `.env` file via the [`TwitterScraper.login`](src/notebooks/tweet_scraper.ipynb) method.
- **Scraping Process** ([`TwitterScraper.scrape_tweets`](src/notebooks/tweet_scraper.ipynb) and [`TwitterScraper.scrape_every_night`](src/notebooks/tweet_scraper.ipynb)):
    - Iterates through artists and festival nights defined in [`SANREMO`](src/notebooks/constants.py) from [src/notebooks/constants.py](src/notebooks/constants.py).
    - For each artist and night, it scrapes tweets within specified time windows (e.g., 21:00 to 01:00 the next day, Rome time).
    - Tweets are collected in hourly intervals.
- **Data Storage**:
    - Raw scraped tweets are saved in CSV files.
    - The directory structure is: `src/notebooks/scraped_data/{Night_Name}/{Artist_Name}/YYYY-MM-DD_HH-HH.csv`.
    - Each CSV file contains `artist`, `datetime`, and `content` columns.
- **Query Optimization**:
    - Ambiguous artist names (e.g., "Clara", "Gaia") are disambiguated by appending "Sanremo" to the search term (e.g., "Clara Sanremo") as defined in [`SANREMO["ARTISTS"]`](src/notebooks/constants.py).

## 🧹 Data Cleaning
- **Initial Cleaning**: The scraper primarily focuses on collecting text content.
- **Analysis Time**: During sentiment analysis in [src/notebooks/tweet_analysis.ipynb](src/notebooks/tweet_analysis.ipynb), tweets with empty content after basic preprocessing are marked as "NOT_AVAILABLE".
- **Prediction Time**:
    - The [`load_data_night`](src/notebooks/utils.py) function in [src/notebooks/utils.py](src/notebooks/utils.py) loads data up to a specified prediction time, filtering out future data.
    - The [`clean_data`](src/notebooks/utils.py) function further refines the dataset by removing irrelevant nights and time slots based on the prediction point.

## 🤖 Sentiment Analysis
Sentiment analysis is performed in the [src/notebooks/tweet_analysis.ipynb](src/notebooks/tweet_analysis.ipynb) notebook.

- **Model**:
    - Utilizes the `cardiffnlp/twitter-roberta-base-sentiment-latest` model from Hugging Face, which is fine-tuned on the TweetEval benchmark.
- **Process**:
    - For each tweet, the model predicts probabilities for 'negative', 'neutral', and 'positive' sentiments.
    - **Binary Conversion**: Neutral probabilities are evenly redistributed to positive and negative scores. The sentiment with the higher resulting score is chosen.
- **Output**:
    - The sentiment analysis script updates the CSV files in `src/notebooks/analyzed_data/` (mirroring the `scraped_data` structure) by adding a `sentiment` column (`positive` or `negative`).

## 📈 Prediction Model
The prediction model is an LSTM (Long Short-Term Memory) network, implemented and trained in [src/notebooks/early_prediction.ipynb](src/notebooks/early_prediction.ipynb).

- **Data Preparation** ([`utils.py`](src/notebooks/utils.py)):
    - [`aggregate_data`](src/notebooks/utils.py): Aggregates tweet counts (positive and negative) per artist for each hourly interval.
    - [`prepare_lstm_data_with_labels`](src/notebooks/utils.py):
        - Creates sequences of historical positive and negative tweet counts for each artist.
        - Applies `MinMaxScaler` to scale features to a [0,1] range for each artist independently.
        - Uses a `lookback` window to define the length of input sequences.
        - Splits data into training and validation sets (20% for validation, without shuffling to maintain temporal order).
- **Model Architecture** ([`build_improved_lstm_model`](src/notebooks/early_prediction.ipynb)):
    - A sequential Keras model:
        - `LSTM(128, return_sequences=True)`
        - `Dropout(0.3)`
        - `LSTM(64, return_sequences=False)`
        - `Dropout(0.3)`
        - `Dense(64, activation="relu")`
        - `BatchNormalization()`
        - `Dropout(0.3)`
        - `Dense(2)` (outputting scaled positive and negative counts)
- **Training**:
    - Optimizer: `adam`
    - Loss Function: `mse` (Mean Squared Error)
    - Callbacks: `EarlyStopping` (monitors `val_loss` with a patience of 5).
- **Prediction**:
    - The [`predict_next_for_artist`](src/notebooks/utils.py) function uses the trained model to predict the next hour's positive and negative tweet counts for each artist.
    - Predictions are inverse-transformed using the artist-specific scaler.

## 📊 Results & Evaluation
The project aims to predict the artist with the highest positive sentiment momentum, correlating this with potential televote success.

- Detailed results, including model performance, prediction accuracy for the final night, and comparisons with actual televote rankings (Top 5, Bottom 5), are documented in the project report: [report/main.pdf](report/main.pdf).
- The model demonstrated strong performance, successfully predicting the winner and several other artists in the Top 5 of the televote.
- Plots for individual artist predictions and combined predictions are saved in the `src/notebooks/predictions/` directory.

---
This README provides a comprehensive guide to the Sanremo Forecasting project. For more in-depth information, please refer to the source code and the [project report](report/main.pdf).