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
     TWITTER_API_KEY=your_api_key
     TWITTER_API_SECRET=your_api_secret
     TWITTER_ACCESS_TOKEN=your_access_token
     TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
     ```