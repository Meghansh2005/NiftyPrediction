"""
Simple Signal Engine — Rule-based, no ML
=========================================
Port: 8002
Features: volatility (VIX), price_momentum, expiry_proximity, news_score
Chart patterns on last 20 bars of 15m data
BankNifty correlation for Nifty impact

RUN:
    uvicorn simple_signal_engine:app --port 8002
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Simple Signal Engine", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TICKERS = {"nifty": "^NSEI", "banknifty": "^NSEBANK", "sensex": "^BSESN"}
VIX_TICKER = "^INDIAVIX"
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

BULL_KW = ["rally", "surge", "jump", "gain", "bullish", "rate cut", "stimulus",
           "fii buying", "growth", "profit", "beat estimates", "upgrade", "infrastructure"]
BEAR_KW = ["fall", "drop", "crash", "decline", "bearish", "rate hike", "war",
           "tension", "fii selling", "recession", "inflation", "miss estimates",
           "downgrade", "geopolitical"]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def fetch_ohlcv(ticker: str, period: str = "30d", interval: str = "15m") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return _flatten(df)


def fetch_vix() -> float:
    try:
        df = yf.download(VIX_TICKER, period="5d", interval="1d",
                         progress=False, auto_adjust=True)
        df = _flatten(df)
        return float(df["Close"].iloc[-1])
    except Exception:
        return 15.0


def days_to_next_thursday() -> int:
    today = datetime.now().date()
    days_ahead = 3 - today.weekday()  # Thursday = 3
    if days_ahead < 0:
        days_ahead += 7
    return days_ahead


def fetch_news_sentiment(query: str = "Nifty NSE India stock market") -> float:
    if not NEWS_API_KEY:
        return 0.0
    try:
        since = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = (f"https://newsapi.org/v2/everything?q={query}&from={since}"
               f"&sortBy=publishedAt&language=en&pageSize=20&apiKey={NEWS_API_KEY}")
        arts = requests.get(url, timeout=5).json().get("articles", [])
        if not arts:
            return 0.0
        bt = br = 0
        for a in arts:
            t = ((a.get("title") or "") + " " + (a.get("description") or "")).lower()
            bt += sum(1 for k in BULL_KW if k in t)
            br += sum(1 for k in BEAR_KW if k in t)
        tot = bt + br
        return round(max(-1.0, min(1.0, (bt - br) / tot if tot else 0.0)), 3)
    except Exception:
        return 0.0


def fetch_news_headlines(query: str = "Nifty NSE India stock market") -> list:
    if not NEWS_API_KEY:
        return []
    try:
        since = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = (f"https://newsapi.org/v2/everything?q={query}&from={since}"
               f"&sortBy=publishedAt&language=en&pageSize=20&apiKey={NEWS_API_KEY}")
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


# ── PATTERN DETECTION ─────────────────────────────────────────────────────────

def detect_pattern(df: pd.DataFrame) -> str:
    bars = df.tail(20)
    if len(bars) < 3:
        return "NONE"

    o = bars["Open"].values
    h = bars["High"].values
    l = bars["Low"].values
    c = bars["Close"].values

    # Last two bars
    prev_o, prev_c = o[-2], c[-2]
    curr_o, curr_c = o[-1], c[-1]
    curr_h, curr_l = h[-1], l[-1]

    prev_green = prev_c > prev_o
    curr_green = curr_c > curr_o

    prev_body = abs(prev_c - prev_o)
    curr_body = abs(curr_c - curr_o)
    curr_range = curr_h - curr_l if curr_h != curr_l else 1e-9

    upper_wick = curr_h - max(curr_o, curr_c)
    lower_wick = min(curr_o, curr_c) - curr_l

    # BULLISH_ENGULFING
    if curr_green and not prev_green and curr_o < prev_c and curr_c > prev_o:
        return "BULLISH_ENGULFING"

    # BEARISH_ENGULFING
    if not curr_green and prev_green and curr_o > prev_c and curr_c < prev_o:
        return "BEARISH_ENGULFING"

    # DOJI
    if curr_body < 0.1 * curr_range:
        return "DOJI"

    # HAMMER (bullish reversal): lower wick > 2x body, small upper wick
    if curr_body > 0 and lower_wick > 2 * curr_body and upper_wick < curr_body:
        return "HAMMER"

    # SHOOTING_STAR (bearish reversal): upper wick > 2x body, small lower wick
    if curr_body > 0 and upper_wick > 2 * curr_body and lower_wick < curr_body:
        return "SHOOTING_STAR"

    # THREE_GREEN
    if all(c[i] > o[i] for i in [-3, -2, -1]):
        return "THREE_GREEN"

    # THREE_RED
    if all(c[i] < o[i] for i in [-3, -2, -1]):
        return "THREE_RED"

    return "NONE"


# ── BANKNIFTY CORRELATION ─────────────────────────────────────────────────────

def fetch_banknifty_correlation() -> dict:
    try:
        df_bn = fetch_ohlcv("^NSEBANK", period="5d", interval="15m")
        last_close = float(df_bn["Close"].iloc[-1])
        prev_close = float(df_bn["Close"].iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 3)

        if change_pct > 0.2:
            direction = "UP"
            impact = "70-80% chance Nifty follows"
        elif change_pct < -0.2:
            direction = "DOWN"
            impact = "70-80% chance Nifty follows"
        else:
            direction = "FLAT"
            impact = "No strong correlation signal"

        return {
            "direction": direction,
            "magnitude_pct": abs(change_pct),
            "nifty_impact": impact,
        }
    except Exception:
        return {"direction": "FLAT", "magnitude_pct": 0.0, "nifty_impact": "Data unavailable"}


# ── SIGNAL LOGIC ──────────────────────────────────────────────────────────────

def compute_signal(vix: float, price_momentum: float, news_score: float,
                   expiry_proximity: int, pattern: str, banknifty: dict) -> tuple[int, str]:
    score = 0

    if vix < 15:
        score += 1
    if vix > 25:
        score -= 2

    if price_momentum > 0.3:
        score += 2
    elif price_momentum < -0.3:
        score -= 2

    if news_score > 0.2:
        score += 1
    elif news_score < -0.2:
        score -= 1

    if expiry_proximity <= 1:
        score -= 1

    if pattern in ("BULLISH_ENGULFING", "HAMMER", "THREE_GREEN"):
        score += 2
    elif pattern in ("BEARISH_ENGULFING", "SHOOTING_STAR", "THREE_RED"):
        score -= 2

    if banknifty["direction"] == "UP":
        score += 1
    elif banknifty["direction"] == "DOWN":
        score -= 1

    if score >= 3:
        signal = "BUY CALL"
    elif score <= -3:
        signal = "BUY PUT"
    else:
        signal = "WAIT"

    return score, signal


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def option_suggestion(signal: str, curr_price: float, atr_pts: float, index: str) -> dict:
    step = 100 if index == "banknifty" else 50
    atm = round(curr_price / step) * step

    if signal == "BUY CALL":
        strike = atm
        entry = round(curr_price - atr_pts * 0.3, 0)
        target = round(curr_price + atr_pts * 1.5, 0)
        sl = round(curr_price - atr_pts * 0.8, 0)
        action = f"BUY {strike} CE"
    elif signal == "BUY PUT":
        strike = atm
        entry = round(curr_price + atr_pts * 0.3, 0)
        target = round(curr_price - atr_pts * 1.5, 0)
        sl = round(curr_price + atr_pts * 0.8, 0)
        action = f"BUY {strike} PE"
    else:
        return {"action": "WAIT — no trade", "atm_strike": atm,
                "entry": None, "target": None, "stop_loss": None}

    return {
        "action": action,
        "atm_strike": atm,
        "entry": entry,
        "target": target,
        "stop_loss": sl,
    }


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "engine": "simple_signal_engine", "port": 8002}


@app.get("/price/{index}")
def get_price(index: str):
    index = index.lower()
    if index not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown index: {index}")
    ticker = TICKERS[index]
    try:
        df = fetch_ohlcv(ticker, period="5d", interval="15m")
        curr = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        now = datetime.now()
        return {
            "index": index,
            "price": round(curr, 2),
            "prev_close": round(prev, 2),
            "change": round(curr - prev, 2),
            "change_pct": round((curr - prev) / prev * 100, 3),
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(float(df["Open"].iloc[-1]), 2),
            "high": round(float(df["High"].iloc[-1]), 2),
            "low": round(float(df["Low"].iloc[-1]), 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal/{index}")
def get_signal(index: str):
    index = index.lower()
    if index not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Unknown index: {index}")
    ticker = TICKERS[index]

    try:
        df = fetch_ohlcv(ticker, period="30d", interval="15m")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    curr_price = float(df["Close"].iloc[-1])
    prev_price = float(df["Close"].iloc[-4]) if len(df) >= 4 else float(df["Close"].iloc[0])
    price_momentum = round((curr_price - prev_price) / prev_price * 100, 4)

    vix = fetch_vix()
    expiry_proximity = days_to_next_thursday()
    news_score = fetch_news_sentiment()
    headlines = fetch_news_headlines()
    pattern = detect_pattern(df)
    banknifty = fetch_banknifty_correlation()

    score, signal = compute_signal(vix, price_momentum, news_score,
                                   expiry_proximity, pattern, banknifty)

    atr_pts = compute_atr(df)
    suggestion = option_suggestion(signal, curr_price, atr_pts, index)
    confidence = round(min(100.0, abs(score) / 7 * 100), 1)

    return {
        "index": index,
        "signal": signal,
        "score": score,
        "confidence_pct": confidence,
        "pattern": pattern,
        "features": {
            "volatility_vix": round(vix, 2),
            "price_momentum_pct": price_momentum,
            "expiry_proximity_days": expiry_proximity,
            "news_score": news_score,
        },
        "banknifty_correlation": banknifty,
        "option_suggestion": suggestion,
        "current_price": round(curr_price, 2),
        "atr_pts": round(atr_pts, 2),
        "news_headlines": headlines,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
