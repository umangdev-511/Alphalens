import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

st.set_page_config(
    page_title="AlphaLens — AI Stock Analyst",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: #050810;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,200,120,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0,120,255,0.06) 0%, transparent 50%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

[data-testid="stSidebar"] {
    background: #080d14 !important;
    border-right: 1px solid rgba(0,200,120,0.12);
}
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stTextInput"] input {
    background: #0d1520 !important;
    border: 1px solid rgba(0,200,120,0.25) !important;
    border-radius: 4px !important;
    color: #e8f5e9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(0,200,120,0.6) !important;
    box-shadow: 0 0 0 2px rgba(0,200,120,0.1) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c878, #00a060) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,200,120,0.3) !important;
}

[data-testid="stMetric"] {
    background: #0d1520;
    border: 1px solid rgba(0,200,120,0.12);
    border-radius: 6px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: #4a7a6a !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #e8f5e9 !important;
}

hr { border-color: rgba(0,200,120,0.1) !important; }
.stSpinner > div { border-top-color: #00c878 !important; }
.stMarkdown p, .stMarkdown li { color: #a8c8b8; line-height: 1.7; }
.stMarkdown strong { color: #e8f5e9; }
.stCaption { font-family: 'JetBrains Mono', monospace !important; color: #2a4a3a !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def fetch_stock_data(ticker: str, retries: int = 3) -> pd.DataFrame:
    nse_ticker = ticker.upper() + ".NS"
    for attempt in range(retries):
        try:
            df = yf.download(nse_ticker, period="6mo", auto_adjust=True, progress=False)
            if not df.empty:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                return df
            time.sleep(2)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise ValueError(f"Could not fetch data for {ticker}. Please check the ticker and try again.")
    raise ValueError(f"No data found for {ticker}. Please verify the NSE ticker symbol.")

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def build_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name="Price",
        increasing_line_color='#00c878', decreasing_line_color='#ff4560',
        increasing_fillcolor='rgba(0,200,120,0.7)',
        decreasing_fillcolor='rgba(255,69,96,0.7)'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20',
        line=dict(color='#2979ff', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50',
        line=dict(color='#ff9100', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA 20',
        line=dict(color='#e040fb', width=1.2, dash='dot')), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
        line=dict(color='#00e5ff', width=2), fill='tozeroy',
        fillcolor='rgba(0,229,255,0.05)'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,69,96,0.5)", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,200,120,0.5)", row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.1)", row=2, col=1)

    hist_colors = ['rgba(0,200,120,0.7)' if v >= 0 else 'rgba(255,69,96,0.7)'
                   for v in (df['MACD'] - df['Signal']).fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df['MACD'] - df['Signal'],
        name='Histogram', marker_color=hist_colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
        line=dict(color='#2979ff', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal',
        line=dict(color='#ff9100', width=1.5)), row=3, col=1)

    fig.update_layout(
        height=640,
        paper_bgcolor='#050810',
        plot_bgcolor='#080d14',
        font=dict(color='#8ab0a0', family='JetBrains Mono'),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            bgcolor='rgba(0,0,0,0)', font=dict(size=11)
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0d1520', font_color='#e8f5e9', font_family='JetBrains Mono')
    )
    fig.update_xaxes(gridcolor='rgba(0,200,120,0.06)', showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor='rgba(0,200,120,0.06)', showgrid=True, zeroline=False)
    return fig

def summarise_for_ai(df, ticker, info):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
    week_ago = df.iloc[-5]['Close'] if len(df) >= 5 else prev['Close']
    month_ago = df.iloc[-22]['Close'] if len(df) >= 22 else df.iloc[0]['Close']
    week_change = ((latest['Close'] - week_ago) / week_ago) * 100
    month_change = ((latest['Close'] - month_ago) / month_ago) * 100
    high_52w = float(df['High'].tail(252).max())
    low_52w = float(df['Low'].tail(252).min())
    return f"""
STOCK: {ticker}.NS (NSE India)
Company: {info.get('longName', ticker)}
Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}

PRICE:
- Current: ₹{float(latest['Close']):.2f} | Day: {price_change:+.2f}% | Week: {week_change:+.2f}% | Month: {month_change:+.2f}%
- 52W High: ₹{high_52w:.2f} | 52W Low: ₹{low_52w:.2f}
- Distance from 52W High: {((float(latest['Close']) - high_52w) / high_52w) * 100:.1f}%
- Volume: {int(float(latest['Volume'])):,}

TECHNICALS:
- RSI(14): {float(latest['RSI']):.1f} {'[OVERBOUGHT]' if float(latest['RSI']) > 70 else '[OVERSOLD]' if float(latest['RSI']) < 30 else '[NEUTRAL]'}
- SMA20: ₹{float(latest['SMA20']):.2f} — price {'ABOVE' if float(latest['Close']) > float(latest['SMA20']) else 'BELOW'}
- SMA50: ₹{float(latest['SMA50']):.2f} — price {'ABOVE' if float(latest['Close']) > float(latest['SMA50']) else 'BELOW'}
- MACD vs Signal: {'BULLISH' if float(latest['MACD']) > float(latest['Signal']) else 'BEARISH'}
- SMA20 vs SMA50: {'GOLDEN CROSS (bullish)' if float(latest['SMA20']) > float(latest['SMA50']) else 'DEATH CROSS (bearish)'}

FUNDAMENTALS:
- Market Cap: {info.get('marketCap', 'N/A')}
- P/E: {info.get('trailingPE', 'N/A')} | P/B: {info.get('priceToBook', 'N/A')}
- Dividend Yield: {info.get('dividendYield', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 'N/A')}
- Profit Margins: {info.get('profitMargins', 'N/A')}
"""

def generate_ai_report(summary, ticker, api_key):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""You are a senior equity analyst at a top Indian brokerage.
Analyse {ticker} (NSE India) using this data:

{summary}

Write a sharp, professional analyst report with these exact sections:

**VERDICT** — BUY / HOLD / SELL in bold + one sentence rationale.

**TECHNICAL PICTURE** — 3-4 sentences on price action, moving averages, RSI and MACD.

**KEY RISKS** — 3 bullet points of main risks right now.

**PRICE LEVELS TO WATCH** — Key support and resistance levels with specific prices.

**BOTTOM LINE** — One paragraph in plain English for a retail investor.

Be direct. No fluff. Use ₹ for prices."""}]
    )
    return message.content[0].text

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 ALPHALENS")
    st.markdown("---")
    api_key = st.text_input("ANTHROPIC API KEY", type="password", placeholder="sk-ant-...")
    st.markdown("---")
    st.markdown("**HOW TO USE**")
    st.markdown("1. Enter API key above\n2. Type NSE ticker below\n3. Hit Analyse\n4. Get AI report instantly")
    st.markdown("---")
    st.markdown("**POPULAR TICKERS**")
    tickers_display = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "WIPRO", "SBIN", "TATAMOTORS", "BAJFINANCE", "ADANIENT"]
    st.markdown(" · ".join(tickers_display))
    st.markdown("---")
    st.caption("⚠ Educational use only. Not financial advice.")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom: 2rem;'>
    <div style='font-family: Syne, sans-serif; font-size: 2.6rem; font-weight: 800; color: #e8f5e9; letter-spacing: -0.02em; line-height: 1;'>
        📡 AlphaLens
    </div>
    <div style='font-family: JetBrains Mono, monospace; font-size: 0.85rem; color: #4a7a6a; margin-top: 0.4rem; letter-spacing: 0.08em;'>
        AI-POWERED STOCK ANALYST · NSE INDIA · REAL-TIME TECHNICALS
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    ticker = st.text_input("", placeholder="Enter NSE ticker — e.g. RELIANCE, TCS, INFY, HDFCBANK",
        label_visibility="collapsed")
with col2:
    analyse = st.button("ANALYSE →", use_container_width=True, type="primary")

st.markdown("---")

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if analyse:
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar to continue.")
    elif not ticker:
        st.warning("Please enter an NSE ticker symbol.")
    else:
        ticker = ticker.strip().upper()

        with st.spinner(f"Fetching market data for {ticker}..."):
            try:
                df = fetch_stock_data(ticker)
                df = compute_indicators(df)
                stock_info = yf.Ticker(ticker + ".NS").info
            except Exception as e:
                st.error(f"⚠ {str(e)}")
                st.info("💡 Yahoo Finance may be rate-limiting. Wait 30 seconds and try again.")
                st.stop()

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = ((float(latest['Close']) - float(prev['Close'])) / float(prev['Close'])) * 100
        company_name = stock_info.get('longName', ticker)
        sector = stock_info.get('sector', '')

        # Company header
        st.markdown(f"""
        <div style='margin-bottom: 1.5rem;'>
            <div style='font-family: Syne, sans-serif; font-size: 1.5rem; font-weight: 700; color: #e8f5e9;'>{company_name}</div>
            <div style='font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: #4a7a6a; letter-spacing: 0.1em;'>{ticker}.NS · NSE INDIA{' · ' + sector.upper() if sector else ''}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        rsi_val = float(latest['RSI'])
        rsi_label = "⚠ Overbought" if rsi_val > 70 else "⚠ Oversold" if rsi_val < 30 else "Neutral"
        macd_val = float(latest['MACD'])
        sig_val = float(latest['Signal'])

        m1.metric("PRICE", f"₹{float(latest['Close']):.2f}", f"{price_change:+.2f}%")
        m2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_label)
        m3.metric("SMA 20", f"₹{float(latest['SMA20']):.2f}", "Above ↑" if float(latest['Close']) > float(latest['SMA20']) else "Below ↓")
        m4.metric("SMA 50", f"₹{float(latest['SMA50']):.2f}", "Above ↑" if float(latest['Close']) > float(latest['SMA50']) else "Below ↓")
        m5.metric("MACD", "Bullish" if macd_val > sig_val else "Bearish", f"{macd_val:.3f}")

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Chart
        st.plotly_chart(build_chart(df, ticker), use_container_width=True)

        # AI Report
        st.markdown("""
        <div style='font-family: Syne, sans-serif; font-size: 1rem; font-weight: 700; color: #00c878;
             letter-spacing: 0.08em; text-transform: uppercase; margin: 1.5rem 0 0.8rem;'>
            🤖 AI Analyst Report
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Generating analysis..."):
            try:
                summary = summarise_for_ai(df, ticker, stock_info)
                report = generate_ai_report(summary, ticker, api_key)
                st.markdown(f"""
                <div style='background:#080d14; border:1px solid rgba(0,200,120,0.15);
                     border-left: 3px solid #00c878; border-radius: 6px;
                     padding: 1.6rem 2rem; font-family: JetBrains Mono, monospace;
                     font-size: 0.87rem; line-height: 1.85; color: #b8d8c8;'>
                {report.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"AI report failed: {str(e)}")

        st.markdown("---")
        st.caption(f"Data sourced from Yahoo Finance · Analysis generated by Claude AI · {datetime.now().strftime('%d %b %Y, %H:%M IST')} · Not financial advice")

else:
    # Empty state
    st.markdown("""
    <div style='text-align: center; padding: 5rem 2rem; opacity: 0.4;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
        <div style='font-family: JetBrains Mono, monospace; font-size: 0.85rem; color: #4a7a6a; letter-spacing: 0.1em;'>
            ENTER A TICKER ABOVE TO BEGIN ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)
