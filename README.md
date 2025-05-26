# 🎤 Sanremo Forecasting: Predicting the Sanremo 2025 Winner (for home viewers) through Sentiment Analysis
**Predicting the Sanremo 2025 Winner for Home Viewers Using Real-Time Twitter Sentiment Analysis**

## 📊 Project Overview
This repository contains the code and documentation for **Sanremo Forecasting**, a machine learning project aimed at **predicting the televote winner** of Italy's 
**Sanremo Music Festival 2025** by performing **real-time sentiment analysis** on Twitter (X.com) posts. The core idea is to scrape tweets about each artist during the live broadcast, classify their sentiment (positive or negative), and use an LSTM-based model to forecast which artist will top the public televote.

## 📁 Repository Structure
```

```

## ⚙️ Installation
1. **Clone the repository**
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables**
   - Create a `.env` file in the root directory and add your Twitter API keys:
     ```
     TWITTER_USERNAME=your_username
     TWITTER_PASSWORD=your_password
     ```

## 🗃️ Data Collection
- **Scraper**: implemented with Selenium, simulating a human-like behavior
- **Data Storage**: collected tweets are stored in:
  ```
  {NightNumber}/
    {Artist}/
        YYYY-MM-DD_HH-HH.csv
  ```
- **Query optimization**: Ambiguous names (e.g., Clara, Gaia, Modà) are disambiguated by appending "Sanremo" to search terms (e.g., "Clara Sanremo").

## 🧹 Data Cleaning
- Removal of tweets without text content

## 🤖 Sentiment Analysis
- **Model**: *cardiffnlp/twitter-roberta-base-sentiment* (fine-tuned on TweetEval) classifies tweets into Negative, Neutral, or Positive.

- **Binary Conversion**: Neutral probabilities are evenly redistributed to positive and negative, yielding a binary sentiment label.

- **Output CSV**: Contains columns `[Artist, Datetime, Content, Sentiment]`.


## 📈 Prediction Model
**Problem Definition**: Predict the winner based on sentiment scores.
1. Input: Time series of counts `(p_t, n_t)` for positive and negative tweets in 1-hour windows (21–22, 22–23, 23–00, 00–01).
   - Task: Forecast `(p_{t+1}, n_{t+1})` given previous windows.

2. Time-to-Predict Function:
```python
TIME_TO_PREDICT = get_time_to_predict(SERATE[-1], "23:53:00")
```
In this case we are predicting the winner at 23:53, so we use the last hour of data (23:00–00:00) to predict the next hour (00:00–01:00).

3. **Model**: LSTM-based model which produces two outputs: `p_{t+1}` and `n_{t+1}`.
4. **Training Setup**:
   - Feature scaling via MinMaxScaler on [0,1]
   - Lookback window for sequence data
   - Train/validation split (20% validation, no shuffle)
   - EarlyStopping callback (patience=5)

# 📊 Results & Evaluation
You can check the results in the `report/main.pdf` which contains all the detail about this project. Anyways, given the ground truth Top 5, our model achieved incredible results by correctly predicting the winner and all the other artists in the Top 5  but the fifth one.