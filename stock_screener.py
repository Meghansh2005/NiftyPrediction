"""
Stock Screener — NSE Top 50 stocks
====================================
Port: 8003
Endpoints: /top-movers, /stock/{symbol}, /search

RUN:
    uvicorn stock_screener:app --port 8003
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Stock Screener", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFOSYS.NS", "SBIN.NS", "HINDUNILVR.NS", "ITC.NS", "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "ASIANPAINT.NS",
    "TITAN.NS", "SUNPHARMA.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS", "BAJAJFINSV.NS",
    "ONGC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "CIPLA.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "GRASIM.NS", "INDUSINDBK.NS",
    "M&M.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "HINDALCO.NS", "VEDL.NS",
    "BPCL.NS", "IOC.NS", "SHREECEM.NS", "PIDILITIND.NS", "SIEMENS.NS",
]

BULL_KW = ["rally", "surge", "jump", "gain", "bullish", "rate cut", "stimulus",
           "fii buying", "growth", "profit", "beat estimates", "upgrade"]
BEAR_KW = ["fall", "drop", "crash", "decline", "bearish", "rate hike", "war",
           "tension", "fii selling", "recession", "inflation", "miss estimates", "downgrade"]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def symbol_to_name(symbol: str) -> str:
    """Derive a readable name from the ticker symbol."""
    base = symbol.replace(".NS", "").replace("-", " ").replace("&", " & ")
    names = {
        "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
        "HDFCBANK": "HDFC Bank", "BHARTIARTL": "Bharti Airtel",
        "ICICIBANK": "ICICI Bank", "INFOSYS": "Infosys",
        "SBIN": "State Bank of India", "HINDUNILVR": "Hindustan Unilever",
        "ITC": "ITC Limited", "LT": "Larsen & Toubro",
        "KOTAKBANK": "Kotak Mahindra Bank", "AXISBANK": "Axis Bank",
        "BAJFINANCE": "Bajaj Finance", "MARUTI": "Maruti Suzuki",
        "ASIANPAINT": "Asian Paints", "TITAN": "Titan Company",
        "SUNPHARMA": "Sun Pharmaceutical", "WIPRO": "Wipro",
        "ULTRACEMCO": "UltraTech Cement", "NESTLEIND": "Nestle India",
        "POWERGRID": "Power Grid Corp", "NTPC": "NTPC Limited",
        "TECHM": "Tech Mahindra", "HCLTECH": "HCL Technologies",
        "BAJAJFINSV": "Bajaj Finserv", "ONGC": "ONGC",
        "TATAMOTORS": "Tata Motors", "ADANIENT": "Adani Enterprises",
        "JSWSTEEL": "JSW Steel", "TATASTEEL": "Tata Steel",
        "COALINDIA": "Coal India", "DIVISLAB": "Divi's Laboratories",
        "DRREDDY": "Dr. Reddy's", "CIPLA": "Cipla",
        "EICHERMOT": "Eicher Motors", "HEROMOTOCO": "Hero MotoCorp",
        "APOLLOHOSP": "Apollo Hospitals", "BRITANNIA": "Britannia Industries",
        "GRASIM": "Grasim Industries", "INDUSINDBK": "IndusInd Bank",
        "M&M": "Mahindra & Mahindra", "BAJAJ-AUTO": "Bajaj Auto",
        "TATACONSUM": "Tata Consumer", "HINDALCO": "Hindalco Industries",
        "VEDL": "Vedanta", "BPCL": "BPCL", "IOC": "Indian Oil Corp",
        "SHREECEM": "Shree Cement", "PIDILITIND": "Pidilite Industries",
        "SIEMENS": "Siemens India",
    }
    return names.get(base, base.title())


def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def compute_simple_signal(df: pd.DataFrame) -> tuple[str, str]:
    close = df["Close"]
    if len(close) < 20:
        return "NEUTRAL", "Insufficient data"

    ma20 = float(close.rolling(20).mean().iloc[-1])
    curr = float(close.iloc[-1])
    rsi = compute_rsi(close)

    if curr > ma20 and rsi < 65:
        return "BULLISH", f"Price above 20-day MA ({ma20:,.0f}) and RSI {rsi:.0f} < 65"
    elif curr < ma20 and rsi > 35:
        return "BEARISH", f"Price below 20-day MA ({ma20:,.0f}) and RSI {rsi:.0f} > 35"
    else:
        return "NEUTRAL", f"Price vs MA20: {((curr-ma20)/ma20*100):+.1f}%, RSI: {rsi:.0f}"


def fetch_stock_news(company_name: str) -> list:
    if not NEWS_API_KEY:
        return []
    try:
        since = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        url = (f"https://newsapi.org/v2/everything?q={company_name}&from={since}"
               f"&sortBy=publishedAt&language=en&pageSize=10&apiKey={NEWS_API_KEY}")
        arts = requests.get(url, timeout=5).json().get("articles", [])
        out = []
        for a in arts[:5]:
            t = ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
            b = sum(1 for k in BULL_KW if k in t)
            r = sum(1 for k in BEAR_KW if k in t)
            out.append({
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", ""),
                "published": (a.get("publishedAt", "")[:16]).replace("T", " "),
                "sentiment": "bullish" if b > r else "bearish" if r > b else "neutral",
            })
        return out
    except Exception:
        return []


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "engine": "stock_screener", "port": 8003}


@app.get("/top-movers")
def top_movers():
    results = []
    tickers_str = " ".join(NIFTY50_TICKERS)

    try:
        data = yf.download(tickers_str, period="2d", interval="1d",
                           progress=False, auto_adjust=True, group_by="ticker")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    for ticker in NIFTY50_TICKERS:
        try:
            if len(NIFTY50_TICKERS) == 1:
                df = data
            else:
                df = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()

            if len(df) < 2:
                continue

            curr = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            change_pct = round((curr - prev) / prev * 100, 3)

            # 52w high/low
            try:
                df_1y = yf.download(ticker, period="1y", interval="1d",
                                    progress=False, auto_adjust=True)
                df_1y = _flatten(df_1y)
                w52_high = round(float(df_1y["High"].max()), 2)
                w52_low = round(float(df_1y["Low"].min()), 2)
            except Exception:
                w52_high = w52_low = curr

            symbol_base = ticker.replace(".NS", "")
            results.append({
                "symbol": symbol_base,
                "name": symbol_to_name(ticker),
                "price": round(curr, 2),
                "change_pct": change_pct,
                "week52_high": w52_high,
                "week52_low": w52_low,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    gainers = results[:10]
    losers = results[-10:][::-1]

    return {
        "gainers": gainers,
        "losers": losers,
        "total_fetched": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/stock/{symbol}")
def get_stock(symbol: str):
    symbol_upper = symbol.upper()
    ticker_str = f"{symbol_upper}.NS"

    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info or {}

        df = yf.download(ticker_str, period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        df = _flatten(df)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol_upper}")

        curr = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else curr
        change_pct = round((curr - prev) / prev * 100, 3)

        # 52w
        try:
            df_1y = yf.download(ticker_str, period="1y", interval="1d",
                                 progress=False, auto_adjust=True)
            df_1y = _flatten(df_1y)
            w52_high = round(float(df_1y["High"].max()), 2)
            w52_low = round(float(df_1y["Low"].min()), 2)
        except Exception:
            w52_high = w52_low = curr

        signal, signal_reason = compute_simple_signal(df)

        pe = info.get("trailingPE") or info.get("forwardPE")
        market_cap = info.get("marketCap")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        company_name = info.get("longName") or symbol_to_name(ticker_str)

        news = fetch_stock_news(company_name)

        return {
            "symbol": symbol_upper,
            "name": company_name,
            "price": round(curr, 2),
            "change_pct": change_pct,
            "change": round(curr - prev, 2),
            "week52_high": w52_high,
            "week52_low": w52_low,
            "pe_ratio": round(pe, 2) if pe else None,
            "market_cap": market_cap,
            "sector": sector,
            "industry": industry,
            "signal": signal,
            "signal_reason": signal_reason,
            "news": news,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    q_lower = q.lower()
    matches = []

    for ticker in NIFTY50_TICKERS:
        base = ticker.replace(".NS", "")
        name = symbol_to_name(ticker)
        if q_lower in base.lower() or q_lower in name.lower():
            try:
                df = yf.download(ticker, period="2d", interval="1d",
                                  progress=False, auto_adjust=True)
                df = _flatten(df)
                curr = float(df["Close"].iloc[-1]) if not df.empty else 0.0
                prev = float(df["Close"].iloc[-2]) if len(df) > 1 else curr
                change_pct = round((curr - prev) / prev * 100, 3) if prev else 0.0
            except Exception:
                curr = 0.0
                change_pct = 0.0

            matches.append({
                "symbol": base,
                "name": name,
                "ticker": ticker,
                "price": round(curr, 2),
                "change_pct": change_pct,
            })

    return {"results": matches, "query": q, "count": len(matches)}
