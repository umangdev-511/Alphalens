# 🔬 AlphaLens
### AI-Powered Stock Analyst for Indian Markets

AlphaLens takes any NSE ticker, pulls live price data, computes technical indicators, and generates a full AI analyst report using Claude — in seconds.

---

## Features
- 📈 Live NSE stock data via Yahoo Finance
- 📊 Technical indicators: RSI, SMA 20/50, EMA 20, MACD
- 🤖 AI-generated analyst report with BUY / HOLD / SELL verdict
- 🕯️ Interactive candlestick charts with Plotly
- 🇮🇳 Built specifically for Indian retail investors

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/alphalens
cd alphalens
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
Go to `http://localhost:8501`

---

## Deploy on Streamlit Cloud (Free — takes 2 minutes)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set main file as `app.py`
4. Click **Deploy**
5. You'll get a live URL like `https://alphalens.streamlit.app`

---

## Usage
1. Enter your [Anthropic API key](https://console.anthropic.com) in the sidebar
2. Type any NSE ticker: `RELIANCE`, `TCS`, `INFY`, `HDFCBANK`, etc.
3. Click **Analyse**
4. Get a full AI analyst report instantly

---

## Tech Stack
- **Streamlit** — UI and deployment
- **yfinance** — NSE stock data
- **Pandas / NumPy** — technical indicator calculations
- **Plotly** — interactive charts
- **Anthropic Claude API** — AI analyst report generation

---

## Disclaimer
AlphaLens is for educational purposes only. Not financial advice.

---

*Built by Umang Pandey*
