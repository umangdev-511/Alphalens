import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaLens",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextInput>div>div>input { background-color: #1e2130; color: white; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .buy { color: #00c853; font-weight: bold; font-size: 1.4rem; }
    .sell { color: #ff1744; font-weight: bold; font-size: 1.4rem; }
    .hold { color: #ffd600; font-weight: bold; font-size: 1.4rem; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Pull OHLCV data from Yahoo Finance for NSE stocks."""
    nse_ticker = ticker.upper() + ".NS"
    df = yf.download(nse_ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data found for {ticker}. Please check the ticker symbol.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate three key technical indicators:
    - SMA 20/50: Simple Moving Average — average price over last 20/50 days
    - EMA 20: Exponential Moving Average — recent prices weighted more heavily
    - RSI 14: Relative Strength Index — measures if stock is overbought (>70) or oversold (<30)
    """
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    return df

def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Build an interactive price + RSI chart using Plotly."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(f"{ticker} Price & Moving Averages", "RSI (14)", "MACD")
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color='#00c853', decreasing_line_color='#ff1744'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20',
        line=dict(color='#2979ff', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50',
        line=dict(color='#ff9100', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA 20',
        line=dict(color='#e040fb', width=1.5, dash='dot')), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#00e5ff', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff1744", opacity=0.6, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00c853", opacity=0.6, row=2, col=1)

    # MACD
    colors = ['#00c853' if v >= 0 else '#ff1744' for v in (df['MACD'] - df['Signal']).fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'],
        name='MACD Hist', marker_color=colors, opacity=0.7), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='#2979ff', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal',
        line=dict(color='#ff9100', width=1.5)), row=3, col=1)

    fig.update_layout(
        height=620,
        paper_bgcolor='#0e1117',
        plot_bgcolor='#1e2130',
        font=dict(color='white'),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig.update_xaxes(gridcolor='#2e3450', showgrid=True)
    fig.update_yaxes(gridcolor='#2e3450', showgrid=True)
    return fig

def summarise_for_ai(df: pd.DataFrame, ticker: str, info: dict) -> str:
    """
    Create a structured data summary to send to Claude.
    The AI will use this to write the analyst report.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    week_ago = df.iloc[-5]['Close'] if len(df) >= 5 else prev['Close']
    month_ago = df.iloc[-22]['Close'] if len(df) >= 22 else df.iloc[0]['Close']
    week_change = ((latest['Close'] - week_ago) / week_ago) * 100
    month_change = ((latest['Close'] - month_ago) / month_ago) * 100
    high_52w = df['High'].tail(252).max()
    low_52w = df['Low'].tail(252).min()

    summary = f"""
STOCK: {ticker}.NS (NSE India)
Company Name: {info.get('longName', ticker)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

PRICE DATA:
- Current Price: ₹{latest['Close']:.2f}
- Day Change: {price_change:+.2f}%
- 1-Week Change: {week_change:+.2f}%
- 1-Month Change: {month_change:+.2f}%
- 52-Week High: ₹{high_52w:.2f}
- 52-Week Low: ₹{low_52w:.2f}
- Current vs 52W High: {((latest['Close'] - high_52w) / high_52w) * 100:.1f}%
- Volume (today): {int(latest['Volume']):,}

TECHNICAL INDICATORS:
- RSI (14): {latest['RSI']:.1f} {'[OVERBOUGHT - above 70]' if latest['RSI'] > 70 else '[OVERSOLD - below 30]' if latest['RSI'] < 30 else '[NEUTRAL]'}
- SMA 20: ₹{latest['SMA20']:.2f} | Price is {'ABOVE' if latest['Close'] > latest['SMA20'] else 'BELOW'} SMA20
- SMA 50: ₹{latest['SMA50']:.2f} | Price is {'ABOVE' if latest['Close'] > latest['SMA50'] else 'BELOW'} SMA50
- EMA 20: ₹{latest['EMA20']:.2f}
- MACD: {latest['MACD']:.3f} | Signal: {latest['Signal']:.3f} | {'BULLISH crossover' if latest['MACD'] > latest['Signal'] else 'BEARISH crossover'}
- SMA20 vs SMA50: {'GOLDEN CROSS (bullish)' if latest['SMA20'] > latest['SMA50'] else 'DEATH CROSS (bearish)'}

FUNDAMENTALS:
- Market Cap: {info.get('marketCap', 'N/A')}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- P/B Ratio: {info.get('priceToBook', 'N/A')}
- Dividend Yield: {info.get('dividendYield', 'N/A')}
- 52W Beta: {info.get('beta', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 'N/A')}
- Profit Margins: {info.get('profitMargins', 'N/A')}
"""
    return summary

def generate_ai_report(summary: str, ticker: str, api_key: str) -> str:
    """
    Send data summary to Claude and get back a full analyst report.
    This is the AI brain of AlphaLens.
    """
    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are a senior equity analyst at a top Indian brokerage. 
A retail investor has asked for a professional analysis of {ticker} listed on NSE India.

Here is the latest data:

{summary}

Write a concise but professional analyst report with these sections:

1. **VERDICT** — Start with a clear BUY / HOLD / SELL call in bold, followed by one sentence rationale.

2. **TECHNICAL PICTURE** — 3-4 sentences analysing the price action, moving averages, RSI and MACD. What is the chart telling us?

3. **KEY RISKS** — 2-3 bullet points of the main risks an investor should be aware of right now.

4. **WHAT TO WATCH** — 2 specific price levels or events to monitor (e.g. support/resistance levels, earnings, macro triggers).

5. **BOTTOM LINE** — One final paragraph summarising your overall view in plain English for a retail investor.

Keep the tone professional but accessible. No fluff. Be direct."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.markdown("## 🔬 AlphaLens")
st.markdown("##### AI-Powered Stock Analyst for Indian Markets")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Anthropic API Key", type="password",
        help="Get your key at console.anthropic.com")
    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
1. Enter your Anthropic API key
2. Type any NSE ticker (e.g. **RELIANCE**, **TCS**, **INFY**, **HDFCBANK**)
3. Click **Analyse**
4. Get a full AI analyst report in seconds
    """)
    st.markdown("---")
    st.markdown("### 💡 Popular Tickers")
    st.markdown("RELIANCE · TCS · INFY · HDFCBANK · ICICIBANK · WIPRO · SBIN · TATAMOTORS · ADANIENT · BAJFINANCE")

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input("Enter NSE Ticker Symbol", placeholder="e.g. RELIANCE, TCS, INFY, HDFCBANK",
        label_visibility="collapsed")
with col2:
    analyse = st.button("🔍 Analyse", use_container_width=True, type="primary")

if analyse:
    if not api_key:
        st.error("Please enter your Anthropic API key in the sidebar.")
    elif not ticker:
        st.warning("Please enter a ticker symbol.")
    else:
        ticker = ticker.strip().upper()

        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                df = fetch_stock_data(ticker)
                df = compute_indicators(df)
                stock_info = yf.Ticker(ticker + ".NS").info
            except Exception as e:
                st.error(f"Could not fetch data: {str(e)}")
                st.stop()

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100

        # ── METRICS ROW ──
        st.markdown("---")
        company_name = stock_info.get('longName', ticker)
        st.markdown(f"### {company_name} ({ticker}.NS)")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Price", f"₹{latest['Close']:.2f}", f"{price_change:+.2f}%")
        m2.metric("RSI (14)", f"{latest['RSI']:.1f}",
            "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Neutral")
        m3.metric("SMA 20", f"₹{latest['SMA20']:.2f}",
            "Price Above ↑" if latest['Close'] > latest['SMA20'] else "Price Below ↓")
        m4.metric("SMA 50", f"₹{latest['SMA50']:.2f}",
            "Price Above ↑" if latest['Close'] > latest['SMA50'] else "Price Below ↓")
        m5.metric("MACD Signal",
            "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish",
            f"{latest['MACD']:.3f}")

        # ── CHART ──
        st.plotly_chart(build_chart(df, ticker), use_container_width=True)

        # ── AI REPORT ──
        st.markdown("### 🤖 AI Analyst Report")
        with st.spinner("Generating AI analysis..."):
            try:
                summary = summarise_for_ai(df, ticker, stock_info)
                report = generate_ai_report(summary, ticker, api_key)
                st.markdown(report)
            except Exception as e:
                st.error(f"AI report generation failed: {str(e)}")

        st.markdown("---")
        st.caption("⚠️ AlphaLens is for educational purposes only. Not financial advice. Always do your own research before investing.")
