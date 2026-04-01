import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET
import time
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="AlphaLens — AI Stock Analyst",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #06111a;
    --panel: rgba(8, 18, 28, 0.82);
    --panel-strong: rgba(11, 23, 35, 0.96);
    --border: rgba(94, 229, 172, 0.16);
    --text: #e6fff4;
    --muted: #88a99b;
    --green: #41e39a;
    --amber: #ffcb6b;
    --red: #ff6b6b;
    --cyan: #4cc9f0;
}

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 15% -10%, rgba(65,227,154,0.14) 0%, transparent 55%),
        radial-gradient(ellipse 60% 40% at 85% 10%, rgba(76,201,240,0.12) 0%, transparent 50%),
        linear-gradient(180deg, rgba(255,255,255,0.015), rgba(255,255,255,0));
}

#MainMenu, footer { visibility: hidden; }
header {
    visibility: visible !important;
    background: transparent !important;
}
.stApp header [data-testid="stHeaderActionElements"] {
    display: none;
}
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1240px; }

[data-testid="stSidebar"] {
    background: rgba(4, 12, 19, 0.95) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'JetBrains Mono', monospace !important; }

[data-testid="stTextInput"] input {
    background: rgba(13, 21, 32, 0.95) !important;
    border: 1px solid rgba(65,227,154,0.22) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    min-height: 3rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(65,227,154,0.55) !important;
    box-shadow: 0 0 0 2px rgba(65,227,154,0.08) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #41e39a, #14b8a6) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
    min-height: 3rem !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) scale(1.01) !important;
    box-shadow: 0 10px 30px rgba(65,227,154,0.22) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(12, 23, 35, 0.92), rgba(8, 16, 24, 0.9));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1.2rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: #6d9586 !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    background: var(--panel-strong);
}

hr { border-color: rgba(65,227,154,0.1) !important; }
.stSpinner > div { border-top-color: var(--green) !important; }
.stMarkdown p, .stMarkdown li { color: #a8c8b8; line-height: 1.7; }
.stMarkdown strong { color: var(--text); }
.stCaption { font-family: 'JetBrains Mono', monospace !important; color: #2a4a3a !important; font-size: 0.72rem !important; }

.hero-shell {
    position: relative;
    overflow: hidden;
    padding: 1.6rem 1.7rem;
    margin-bottom: 1.4rem;
    border-radius: 28px;
    border: 1px solid var(--border);
    background:
        radial-gradient(circle at top right, rgba(76,201,240,0.18), transparent 30%),
        radial-gradient(circle at left center, rgba(65,227,154,0.12), transparent 35%),
        linear-gradient(180deg, rgba(12, 23, 35, 0.95), rgba(8, 16, 24, 0.92));
    box-shadow: 0 24px 80px rgba(0,0,0,0.35);
}
.hero-kicker, .section-kicker, .chip-row {
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero-kicker { color: #7bc7aa; font-size: 0.74rem; }
.hero-title {
    color: var(--text);
    font-size: clamp(2.2rem, 5vw, 4rem);
    line-height: 0.94;
    letter-spacing: -0.04em;
    margin: 0.5rem 0 0.8rem;
    font-weight: 800;
    max-width: 720px;
}
.hero-copy {
    color: #9fc4b5;
    font-size: 1rem;
    max-width: 720px;
    margin-bottom: 1rem;
}
.chip-row {
    color: #82aa99;
    font-size: 0.72rem;
}
.glass-card {
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.15rem 1.2rem;
    background: linear-gradient(180deg, rgba(12, 23, 35, 0.9), rgba(8, 16, 24, 0.88));
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
}
.section-kicker {
    color: #7bc7aa;
    font-size: 0.72rem;
    margin-bottom: 0.35rem;
}
.section-title {
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.section-copy {
    color: #91b6a6;
    font-size: 0.9rem;
    line-height: 1.65;
}
.verdict-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: rgba(255,255,255,0.06);
    color: var(--text);
    margin-bottom: 0.65rem;
}
.brand-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.brand-badge {
    width: 62px;
    height: 62px;
    border-radius: 18px;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 16px 40px rgba(0,0,0,0.28);
}
.brand-lockup {
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
}
.brand-name {
    color: var(--text);
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.brand-subtitle {
    color: #7bc7aa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

NEWS_LOOKBACK_DAYS = 10
MAX_HEADLINES = 12
VADER_ANALYSER = SentimentIntensityAnalyzer()
POSITIVE_FINANCE_TERMS = {
    "beat": 0.9, "beats": 0.9, "upgrade": 0.8, "upgrades": 0.8, "surge": 0.8,
    "surges": 0.8, "growth": 0.5, "record": 0.6, "profit": 0.7, "profits": 0.7,
    "strong": 0.5, "bullish": 0.9, "outperform": 0.9, "expands": 0.5,
    "expansion": 0.5, "wins": 0.7, "order": 0.4, "orders": 0.4, "rally": 0.8,
    "rallies": 0.8, "dividend": 0.5, "approval": 0.6, "accretive": 0.8,
}
NEGATIVE_FINANCE_TERMS = {
    "miss": -0.9, "misses": -0.9, "downgrade": -0.8, "downgrades": -0.8,
    "fall": -0.7, "falls": -0.7, "slump": -0.9, "slumps": -0.9, "loss": -0.8,
    "losses": -0.8, "weak": -0.5, "bearish": -0.9, "underperform": -0.9,
    "probe": -0.6, "fraud": -1.0, "penalty": -0.7, "default": -1.0,
    "lawsuit": -0.8, "crash": -1.0, "plunge": -1.0, "decline": -0.6,
    "declines": -0.6, "warning": -0.5, "cuts": -0.6, "cut": -0.6,
}


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip().upper() == "N/A":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_large_number(value) -> str:
    number = safe_float(value)
    if np.isnan(number):
        return "N/A"
    if abs(number) >= 1e12:
        return f"₹{number / 1e12:.2f}T"
    if abs(number) >= 1e9:
        return f"₹{number / 1e9:.2f}B"
    if abs(number) >= 1e7:
        return f"₹{number / 1e7:.2f}Cr"
    if abs(number) >= 1e5:
        return f"₹{number / 1e5:.2f}L"
    return f"₹{number:,.0f}"


def format_percent(value) -> str:
    number = safe_float(value)
    return "N/A" if np.isnan(number) else f"{number * 100:.1f}%"

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


def normalise_news_item(item, ticker: str):
    title = item.get("title") or item.get("content", {}).get("title")
    link = item.get("link") or item.get("canonicalUrl", {}).get("url")
    publisher = item.get("publisher")
    if isinstance(publisher, dict):
        publisher = publisher.get("name")
    published = item.get("providerPublishTime")
    published_dt = pd.to_datetime(published, unit="s", errors="coerce") if published else pd.NaT
    if not title or not link:
        return None
    return {
        "ticker": ticker,
        "title": title.strip(),
        "publisher": publisher or "Yahoo Finance",
        "link": link,
        "published_at": published_dt,
        "source": "Yahoo Finance",
    }


def fetch_yahoo_news(ticker: str, company_name: str) -> list[dict]:
    try:
        news_items = yf.Ticker(f"{ticker}.NS").news or []
    except Exception:
        return []

    articles = []
    for item in news_items[:MAX_HEADLINES]:
        parsed = normalise_news_item(item, ticker)
        if parsed:
            articles.append(parsed)
    return articles


def fetch_google_news_rss(ticker: str, company_name: str) -> list[dict]:
    search_query = quote_plus(f'"{ticker}" "{company_name}" NSE stock')
    url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []

    articles = []
    for item in root.findall(".//item")[:MAX_HEADLINES]:
        title = item.findtext("title")
        link = item.findtext("link")
        publisher = item.findtext("source") or "Google News"
        pub_date = item.findtext("pubDate")
        if not title or not link:
            continue
        try:
            published_at = parsedate_to_datetime(pub_date) if pub_date else pd.NaT
        except Exception:
            published_at = pd.NaT
        articles.append({
            "ticker": ticker,
            "title": title.strip(),
            "publisher": publisher,
            "link": link,
            "published_at": pd.to_datetime(published_at, errors="coerce"),
            "source": "Google News",
        })
    return articles


def fetch_stock_news(ticker: str, company_name: str) -> pd.DataFrame:
    raw_articles = fetch_yahoo_news(ticker, company_name)
    if len(raw_articles) < 5:
        raw_articles.extend(fetch_google_news_rss(ticker, company_name))

    if not raw_articles:
        return pd.DataFrame(columns=["ticker", "title", "publisher", "link", "published_at", "source"])

    news_df = pd.DataFrame(raw_articles).drop_duplicates(subset=["title"]).copy()
    news_df["published_at"] = pd.to_datetime(news_df["published_at"], errors="coerce")
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    cutoff = now - pd.Timedelta(days=NEWS_LOOKBACK_DAYS)
    recent_mask = news_df["published_at"].isna() | (news_df["published_at"] >= cutoff)
    news_df = news_df.loc[recent_mask].sort_values("published_at", ascending=False, na_position="last")
    return news_df.head(MAX_HEADLINES).reset_index(drop=True)


def score_headline_sentiment(headline: str) -> dict:
    headline_lower = headline.lower()
    vader_score = VADER_ANALYSER.polarity_scores(headline)["compound"]
    lexicon_score = 0.0

    for word, weight in POSITIVE_FINANCE_TERMS.items():
        if word in headline_lower:
            lexicon_score += weight
    for word, weight in NEGATIVE_FINANCE_TERMS.items():
        if word in headline_lower:
            lexicon_score += weight

    lexicon_score = float(np.tanh(lexicon_score / 2.5))
    combined_score = float(np.clip((vader_score * 0.55) + (lexicon_score * 0.45), -1, 1))

    if combined_score >= 0.25:
        label = "Positive"
    elif combined_score <= -0.25:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "vader_score": vader_score,
        "lexicon_score": lexicon_score,
        "sentiment_score": combined_score,
        "sentiment_label": label,
    }


def analyse_news_sentiment(news_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if news_df.empty:
        return news_df.copy(), {
            "headline_count": 0,
            "sentiment_score": 0.0,
            "sentiment_percent": 50,
            "sentiment_label": "Neutral",
            "confidence": 0.0,
            "positive_headlines": 0,
            "negative_headlines": 0,
            "neutral_headlines": 0,
        }

    scored_df = news_df.copy()
    scores = scored_df["title"].apply(score_headline_sentiment).apply(pd.Series)
    scored_df = pd.concat([scored_df, scores], axis=1)

    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    age_days = ((now - scored_df["published_at"]).dt.total_seconds() / 86400).clip(lower=0)
    scored_df["recency_weight"] = np.where(scored_df["published_at"].notna(), np.exp(-age_days / 4), 0.65)
    weighted_score = np.average(scored_df["sentiment_score"], weights=scored_df["recency_weight"])
    confidence = min(1.0, 0.35 + (0.08 * len(scored_df)) + (0.25 * abs(weighted_score)))

    if weighted_score >= 0.2:
        label = "Bullish"
    elif weighted_score <= -0.2:
        label = "Bearish"
    else:
        label = "Mixed"

    summary = {
        "headline_count": int(len(scored_df)),
        "sentiment_score": float(weighted_score),
        "sentiment_percent": int(round((weighted_score + 1) * 50)),
        "sentiment_label": label,
        "confidence": float(confidence),
        "positive_headlines": int((scored_df["sentiment_label"] == "Positive").sum()),
        "negative_headlines": int((scored_df["sentiment_label"] == "Negative").sum()),
        "neutral_headlines": int((scored_df["sentiment_label"] == "Neutral").sum()),
    }
    return scored_df, summary


def generate_recommendation(df: pd.DataFrame, sentiment_summary: dict) -> dict:
    latest = df.iloc[-1]
    score = 0.0
    reasons = []

    rsi = safe_float(latest["RSI"])
    close = safe_float(latest["Close"])
    sma20 = safe_float(latest["SMA20"])
    sma50 = safe_float(latest["SMA50"])
    macd = safe_float(latest["MACD"])
    signal = safe_float(latest["Signal"])

    if not np.isnan(close) and not np.isnan(sma20):
        if close > sma20:
            score += 1.0
            reasons.append("Price is trading above the 20-day trend.")
        else:
            score -= 1.0
            reasons.append("Price is sitting below the 20-day trend.")

    if not np.isnan(close) and not np.isnan(sma50):
        if close > sma50:
            score += 1.2
            reasons.append("Price is holding above the 50-day base.")
        else:
            score -= 1.2
            reasons.append("Price has slipped under the 50-day base.")

    if not np.isnan(sma20) and not np.isnan(sma50):
        if sma20 > sma50:
            score += 1.0
            reasons.append("Short-term moving averages remain in bullish alignment.")
        else:
            score -= 1.0
            reasons.append("Short-term moving averages remain in bearish alignment.")

    if not np.isnan(macd) and not np.isnan(signal):
        if macd > signal:
            score += 0.8
            reasons.append("MACD momentum is positive.")
        else:
            score -= 0.8
            reasons.append("MACD momentum is negative.")

    if not np.isnan(rsi):
        if rsi < 35:
            score += 0.6
            reasons.append("RSI suggests the stock is closer to oversold than overbought.")
        elif rsi > 70:
            score -= 0.9
            reasons.append("RSI is stretched into overbought territory.")

    sentiment_boost = sentiment_summary["sentiment_score"] * 2.2
    score += sentiment_boost
    reasons.append(
        f"News flow is {sentiment_summary['sentiment_label'].lower()} with a sentiment score of "
        f"{sentiment_summary['sentiment_percent']}/100."
    )

    conviction = min(100, max(0, int(round(((score + 6) / 12) * 100))))
    if score >= 2.2:
        verdict = "BUY"
    elif score <= -2.0:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    return {
        "verdict": verdict,
        "composite_score": round(score, 2),
        "conviction": conviction,
        "reasons": reasons[:5],
    }


def build_sentiment_gauge(sentiment_summary: dict, recommendation: dict) -> go.Figure:
    score = sentiment_summary["sentiment_percent"]
    verdict = recommendation["verdict"]
    color = "#41e39a" if verdict == "BUY" else "#ff6b6b" if verdict == "SELL" else "#ffcb6b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"color": "#e6fff4", "size": 34, "family": "Syne"}},
        title={"text": f"<span style='font-family:JetBrains Mono; font-size:12px; letter-spacing:1px;'>NEWS SENTIMENT</span>"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.2)", "tickfont": {"color": "#6d9586"}},
            "bar": {"color": color, "thickness": 0.34},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(255,107,107,0.18)"},
                {"range": [35, 65], "color": "rgba(255,203,107,0.16)"},
                {"range": [65, 100], "color": "rgba(65,227,154,0.18)"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6fff4", family="JetBrains Mono"),
    )
    return fig


def build_factor_chart(df: pd.DataFrame, sentiment_summary: dict) -> go.Figure:
    latest = df.iloc[-1]
    factors = {
        "Trend 20D": 1 if safe_float(latest["Close"]) > safe_float(latest["SMA20"]) else -1,
        "Trend 50D": 1.2 if safe_float(latest["Close"]) > safe_float(latest["SMA50"]) else -1.2,
        "MACD": 0.8 if safe_float(latest["MACD"]) > safe_float(latest["Signal"]) else -0.8,
        "RSI": 0.6 if safe_float(latest["RSI"]) < 35 else -0.9 if safe_float(latest["RSI"]) > 70 else 0.0,
        "News": sentiment_summary["sentiment_score"] * 2.2,
    }
    factor_df = pd.DataFrame({"factor": list(factors.keys()), "value": list(factors.values())})
    factor_df["color"] = factor_df["value"].apply(lambda x: "#41e39a" if x >= 0 else "#ff6b6b")
    fig = go.Figure(go.Bar(
        x=factor_df["value"],
        y=factor_df["factor"],
        orientation="h",
        marker_color=factor_df["color"],
        text=factor_df["value"].map(lambda x: f"{x:+.2f}"),
        textposition="outside",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=30, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d7efe4", family="JetBrains Mono"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def lion_mark_svg(size: int = 40) -> str:
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M32 5 18 11 10 24l3 17 12 15h14l12-15 3-17-8-13L32 5Z" fill="#000000"/>
        <path d="M32 12c-5.6 0-10.8 2.7-13.8 7.3 1.7-.2 3.4.2 4.8 1.1 2.5-1.7 5.7-2.6 9-2.6s6.5.9 9 2.6c1.4-.9 3.1-1.3 4.8-1.1C42.8 14.7 37.6 12 32 12Z" fill="#ffffff"/>
        <path d="M22.5 22.5c1.6 1 2.9 2.3 3.9 3.9-2.2.6-4.2 1.8-5.8 3.5-1.4-.7-2.9-1-4.4-.9.7-2.8 3-5.4 6.3-7.5Zm19 0c3.3 2.1 5.6 4.7 6.3 7.5-1.5-.1-3 .2-4.4.9-1.6-1.7-3.6-2.9-5.8-3.5 1-1.6 2.3-2.9 3.9-3.9Z" fill="#ffffff"/>
        <path d="M24.5 31.5c2.2 0 4 1.8 4 4 0 2-1.5 3.7-3.5 4l-2.4 7.5-4.1-2.2 1.9-7.5c-.7-.6-1.1-1.5-1.1-2.5 0-2.2 1.8-4 4.2-4Zm15 0c2.4 0 4.2 1.8 4.2 4 0 1-.4 1.9-1.1 2.5l1.9 7.5-4.1 2.2-2.4-7.5c-2-.3-3.5-2-3.5-4 0-2.2 1.8-4 4-4Z" fill="#ffffff"/>
        <path d="M32 26.5c3.2 0 5.8 2.4 5.8 5.2 0 1.7-.9 3.2-2.3 4.2l2.1 8.7H26.4l2.1-8.7c-1.4-1-2.3-2.5-2.3-4.2 0-2.8 2.6-5.2 5.8-5.2Z" fill="#ffffff"/>
        <path d="M29.2 32.6c.6 0 1 .4 1 1 0 .9.8 1.6 1.8 1.6s1.8-.7 1.8-1.6c0-.6.4-1 1-1s1 .4 1 1c0 2-1.7 3.6-3.8 3.6s-3.8-1.6-3.8-3.6c0-.6.4-1 1-1Z" fill="#000000"/>
        <path d="M27 29.8c.8 0 1.4.6 1.4 1.4S27.8 32.6 27 32.6s-1.4-.6-1.4-1.4.6-1.4 1.4-1.4Zm10 0c.8 0 1.4.6 1.4 1.4s-.6 1.4-1.4 1.4-1.4-.6-1.4-1.4.6-1.4 1.4-1.4Z" fill="#000000"/>
    </svg>
    """

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


def summarise_for_ai(df, ticker, info, sentiment_summary, recommendation):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    pe_ratio = safe_float(info.get('trailingPE'))
    pb_ratio = safe_float(info.get('priceToBook'))
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
- Market Cap: {format_large_number(info.get('marketCap'))}
- P/E: {f"{pe_ratio:.2f}" if not np.isnan(pe_ratio) else 'N/A'} | P/B: {f"{pb_ratio:.2f}" if not np.isnan(pb_ratio) else 'N/A'}
- Dividend Yield: {format_percent(info.get('dividendYield'))}
- Revenue Growth: {format_percent(info.get('revenueGrowth'))}
- Profit Margins: {format_percent(info.get('profitMargins'))}

SENTIMENT ENGINE:
- Headline Count: {sentiment_summary['headline_count']}
- Headline Sentiment: {sentiment_summary['sentiment_label']} ({sentiment_summary['sentiment_percent']}/100)
- Positive / Neutral / Negative: {sentiment_summary['positive_headlines']} / {sentiment_summary['neutral_headlines']} / {sentiment_summary['negative_headlines']}
- Confidence: {sentiment_summary['confidence']:.2f}

DECISION ENGINE:
- Quant Verdict: {recommendation['verdict']}
- Composite Score: {recommendation['composite_score']}
- Conviction: {recommendation['conviction']}/100
"""


def generate_ai_report(summary, ticker, api_key, recommendation):
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""You are a senior equity analyst at a top Indian brokerage.
Analyse {ticker} (NSE India) using this data:

{summary}

Write a sharp, professional analyst report with these exact sections:

**VERDICT** — The first word must be **{recommendation['verdict']}** and the rationale must explicitly reference both technicals and news sentiment.

**TECHNICAL PICTURE** — 3-4 sentences on price action, moving averages, RSI and MACD.

**SENTIMENT SIGNAL** — 2-3 sentences on headline tone, what is driving it, and whether it strengthens or weakens the technical case.

**KEY RISKS** — 3 bullet points of main risks right now.

**PRICE LEVELS TO WATCH** — Key support and resistance levels with specific prices.

**BOTTOM LINE** — One paragraph in plain English for a retail investor.

Be direct. No fluff. Keep the quant verdict unchanged. Use ₹ for prices."""}]
    )
    return message.content[0].text

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class='brand-row' style='margin-bottom:0.2rem;'>
        <div class='brand-badge'>{lion_mark_svg(38)}</div>
        <div class='brand-lockup'>
            <div class='brand-name'>ALPHALENS</div>
            <div class='brand-subtitle'>Lion Signal Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
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
<div class='hero-shell'>
    <div class='brand-row' style='margin-bottom:1rem;'>
        <div class='brand-badge'>""" + lion_mark_svg(42) + """</div>
        <div class='brand-lockup'>
            <div class='brand-name' style='font-size:1.5rem;'>AlphaLens</div>
            <div class='brand-subtitle'>Indian Equities Intelligence Layer</div>
        </div>
    </div>
    <div class='hero-title'>AI stock analyst with a live news-sentiment decision engine for NSE names.</div>
    <div class='hero-copy'>
        Technical momentum, fresh headlines, and an explainable recommendation model merged into one builder-grade dashboard.
    </div>
    <div class='chip-row'>Real-time technicals · NLP headline scoring · Quant verdict · Claude narrative</div>
</div>
<div class='glass-card' style='margin-bottom: 1.25rem;'>
    <div class='section-kicker'>Input</div>
    <div class='section-title'>Run an institutional-style stock check</div>
    <div class='section-copy'>
        Enter an NSE ticker, pull current market structure, score the latest news flow, and generate an explainable Buy/Hold/Sell view.
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
                news_df = fetch_stock_news(ticker, stock_info.get("longName", ticker))
                scored_news_df, sentiment_summary = analyse_news_sentiment(news_df)
                recommendation = generate_recommendation(df, sentiment_summary)
            except Exception as e:
                st.error(f"⚠ {str(e)}")
                st.info("💡 Yahoo Finance may be rate-limiting. Wait 30 seconds and try again.")
                st.stop()

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        price_change = ((float(latest['Close']) - float(prev['Close'])) / float(prev['Close'])) * 100
        company_name = stock_info.get('longName', ticker)
        sector = stock_info.get('sector', '')

        verdict_color = "#41e39a" if recommendation["verdict"] == "BUY" else "#ff6b6b" if recommendation["verdict"] == "SELL" else "#ffcb6b"

        # Company header
        st.markdown(f"""
        <div class='glass-card' style='margin-bottom: 1.2rem;'>
            <div class='verdict-pill' style='border:1px solid {verdict_color}; color:{verdict_color};'>Quant Verdict · {recommendation["verdict"]}</div>
            <div style='font-family: Syne, sans-serif; font-size: 2rem; font-weight: 800; color: #e8f5e9; line-height:1.05;'>{company_name}</div>
            <div style='font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: #6d9586; letter-spacing: 0.1em; margin-top:0.35rem;'>{ticker}.NS · NSE INDIA{' · ' + sector.upper() if sector else ''}</div>
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

        engine_cols = st.columns([1.2, 1, 1, 1])
        engine_cols[0].markdown(f"""
        <div class='glass-card' style='height:100%;'>
            <div class='section-kicker'>Decision Engine</div>
            <div style='font-family: Syne, sans-serif; font-size:2.3rem; font-weight:800; color:{verdict_color}; line-height:0.95;'>
                {recommendation["verdict"]}
            </div>
            <div style='font-family: JetBrains Mono, monospace; font-size:0.8rem; color:#a8c8b8; margin-top:0.55rem;'>Composite score {recommendation["composite_score"]} · Conviction {recommendation["conviction"]}/100</div>
            <div style='margin-top:0.8rem; height:8px; border-radius:999px; background:rgba(255,255,255,0.06); overflow:hidden;'>
                <div style='width:{recommendation["conviction"]}%; height:100%; background:linear-gradient(90deg, {verdict_color}, #4cc9f0);'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        engine_cols[1].metric("NEWS SENTIMENT", f"{sentiment_summary['sentiment_percent']}/100", sentiment_summary["sentiment_label"])
        engine_cols[2].metric("HEADLINES", sentiment_summary["headline_count"], f"{sentiment_summary['positive_headlines']} positive")
        engine_cols[3].metric("MODEL CONFIDENCE", f"{sentiment_summary['confidence']:.2f}", "News-weighted")

        st.markdown("<div style='height: 0.6rem'></div>", unsafe_allow_html=True)
        top_left, top_right = st.columns([1.1, 1])
        with top_left:
            st.markdown("""
            <div class='glass-card'>
                <div class='section-kicker'>What Drove The Call</div>
                <div class='section-title'>Signal breakdown</div>
                <div class='section-copy'>A readable explanation of the current model output, combining chart structure and news tone.</div>
            </div>
            """, unsafe_allow_html=True)
            for reason in recommendation["reasons"]:
                st.markdown(f"- {reason}")
        with top_right:
            st.markdown("""
            <div class='glass-card' style='padding-bottom:0.4rem;'>
                <div class='section-kicker'>Sentiment Gauge</div>
                <div class='section-title'>Headline mood map</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_sentiment_gauge(sentiment_summary, recommendation), use_container_width=True)

        # Chart
        chart_col, factor_col = st.columns([1.55, 1])
        with chart_col:
            st.markdown("""
            <div class='glass-card' style='padding-bottom:0.4rem; margin-bottom:0.7rem;'>
                <div class='section-kicker'>Price Structure</div>
                <div class='section-title'>Multi-layer technical view</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_chart(df, ticker), use_container_width=True)
        with factor_col:
            st.markdown("""
            <div class='glass-card' style='padding-bottom:0.4rem; margin-bottom:0.7rem;'>
                <div class='section-kicker'>Factor Weights</div>
                <div class='section-title'>Contribution to verdict</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_factor_chart(df, sentiment_summary), use_container_width=True)

        st.markdown("""
        <div class='glass-card' style='padding-bottom:0.85rem; margin: 1rem 0 0.8rem;'>
            <div class='section-kicker'>Headline Tape</div>
            <div class='section-title'>News sentiment radar</div>
            <div class='section-copy'>Recent headlines scored individually so the recommendation stays explainable.</div>
        </div>
        """, unsafe_allow_html=True)

        if scored_news_df.empty:
            st.info("No recent headlines were available, so the sentiment engine defaulted to neutral.")
        else:
            display_news = scored_news_df.copy()
            display_news["published_at"] = display_news["published_at"].dt.strftime("%d %b %H:%M").fillna("N/A")
            display_news["sentiment"] = display_news["sentiment_score"].map(lambda x: f"{x:+.2f}")
            st.dataframe(
                display_news[["published_at", "publisher", "title", "link", "sentiment_label", "sentiment", "source"]],
                column_config={
                    "published_at": "Published",
                    "publisher": "Publisher",
                    "title": "Headline",
                    "link": st.column_config.LinkColumn("Link", display_text="Open"),
                    "sentiment_label": "Tone",
                    "sentiment": "Score",
                    "source": "Source",
                },
                use_container_width=True,
                hide_index=True,
            )

        # AI Report
        st.markdown("""
        <div class='glass-card' style='padding-bottom:0.85rem; margin: 1rem 0 0.8rem;'>
            <div class='section-kicker'>AI Layer</div>
            <div class='section-title'>Claude analyst report</div>
            <div class='section-copy'>Narrative commentary generated from the same technical and sentiment inputs driving the quant verdict.</div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Generating analysis..."):
            try:
                summary = summarise_for_ai(df, ticker, stock_info, sentiment_summary, recommendation)
                report = generate_ai_report(summary, ticker, api_key, recommendation)
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
    <div class='glass-card' style='text-align:center; padding:4.5rem 2rem; margin-top:0.75rem;'>
        <div style='font-size: 3.5rem; margin-bottom: 1rem;'>📊</div>
        <div class='section-kicker'>Awaiting Input</div>
        <div style='font-family: Syne, sans-serif; font-size: 1.35rem; font-weight: 700; color: #e8f5e9; margin-bottom:0.45rem;'>
            Enter an NSE ticker to activate the AlphaLens decision engine
        </div>
        <div style='font-family: JetBrains Mono, monospace; font-size: 0.8rem; color: #6d9586; letter-spacing: 0.08em;'>
            EXAMPLES · RELIANCE · TCS · INFY · HDFCBANK · SBIN
        </div>
    </div>
    """, unsafe_allow_html=True)
