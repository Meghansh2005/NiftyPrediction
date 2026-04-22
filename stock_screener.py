"""
Stock Screener — NSE Top 50 + Stock Chart Prediction
Port: 8003
"""
import os, warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import ta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

app = FastAPI(title="Stock Screener", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","BHARTIARTL.NS","ICICIBANK.NS",
    "INFOSYS.NS","SBIN.NS","HINDUNILVR.NS","ITC.NS","LT.NS",
    "KOTAKBANK.NS","AXISBANK.NS","BAJFINANCE.NS","MARUTI.NS","ASIANPAINT.NS",
    "TITAN.NS","SUNPHARMA.NS","WIPRO.NS","ULTRACEMCO.NS","NESTLEIND.NS",
    "POWERGRID.NS","NTPC.NS","TECHM.NS","HCLTECH.NS","BAJAJFINSV.NS",
    "ONGC.NS","TATAMOTORS.NS","ADANIENT.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","CIPLA.NS","EICHERMOT.NS",
    "HEROMOTOCO.NS","APOLLOHOSP.NS","BRITANNIA.NS","GRASIM.NS","INDUSINDBK.NS",
    "BAJAJ-AUTO.NS","TATACONSUM.NS","HINDALCO.NS","VEDL.NS",
    "BPCL.NS","IOC.NS","SHREECEM.NS","PIDILITIND.NS","SIEMENS.NS","M&M.NS",
]

NAMES = {
    "RELIANCE":"Reliance Industries","TCS":"Tata Consultancy Services",
    "HDFCBANK":"HDFC Bank","BHARTIARTL":"Bharti Airtel","ICICIBANK":"ICICI Bank",
    "INFOSYS":"Infosys","SBIN":"State Bank of India","HINDUNILVR":"Hindustan Unilever",
    "ITC":"ITC Limited","LT":"Larsen & Toubro","KOTAKBANK":"Kotak Mahindra Bank",
    "AXISBANK":"Axis Bank","BAJFINANCE":"Bajaj Finance","MARUTI":"Maruti Suzuki",
    "ASIANPAINT":"Asian Paints","TITAN":"Titan Company","SUNPHARMA":"Sun Pharma",
    "WIPRO":"Wipro","ULTRACEMCO":"UltraTech Cement","NESTLEIND":"Nestle India",
    "POWERGRID":"Power Grid","NTPC":"NTPC","TECHM":"Tech Mahindra",
    "HCLTECH":"HCL Technologies","BAJAJFINSV":"Bajaj Finserv","ONGC":"ONGC",
    "TATAMOTORS":"Tata Motors","ADANIENT":"Adani Enterprises","JSWSTEEL":"JSW Steel",
    "TATASTEEL":"Tata Steel","COALINDIA":"Coal India","DIVISLAB":"Divi's Labs",
    "DRREDDY":"Dr. Reddy's","CIPLA":"Cipla","EICHERMOT":"Eicher Motors",
    "HEROMOTOCO":"Hero MotoCorp","APOLLOHOSP":"Apollo Hospitals",
    "BRITANNIA":"Britannia","GRASIM":"Grasim","INDUSINDBK":"IndusInd Bank",
    "BAJAJ-AUTO":"Bajaj Auto","TATACONSUM":"Tata Consumer","HINDALCO":"Hindalco",
    "VEDL":"Vedanta","BPCL":"BPCL","IOC":"Indian Oil","SHREECEM":"Shree Cement",
    "PIDILITIND":"Pidilite","SIEMENS":"Siemens India","M&M":"Mahindra & Mahindra",
}

BULL_KW = ["rally","surge","jump","gain","bullish","rate cut","fii buying",
           "growth","profit","beat estimates","upgrade","infrastructure","buy"]
BEAR_KW = ["fall","drop","crash","decline","bearish","rate hike","war",
           "tension","fii selling","recession","inflation","miss estimates","downgrade"]


def _name(ticker: str) -> str:
    base = ticker.replace(".NS","")
    return NAMES.get(base, base)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def _rsi(series: pd.Series, p: int = 14) -> float:
    d = series.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return float((100 - 100/(1+g/(l+1e-9))).iloc[-1])


def _news(query: str) -> list:
    if not NEWS_API_KEY:
        return []
    try:
        since = (datetime.now()-timedelta(days=3)).strftime("%Y-%m-%d")
        url = (f"https://newsapi.org/v2/everything?q={query}&from={since}"
               f"&sortBy=publishedAt&language=en&pageSize=8&apiKey={NEWS_API_KEY}")
        arts = requests.get(url, timeout=5).json().get("articles",[])
        out = []
        for a in arts[:5]:
            t = ((a.get("title") or "")+" "+(a.get("description") or "")).lower()
            b = sum(1 for k in BULL_KW if k in t)
            r = sum(1 for k in BEAR_KW if k in t)
            out.append({"title":a.get("title",""),"source":a.get("source",{}).get("name",""),
                        "published":(a.get("publishedAt","")[:16]).replace("T"," "),
                        "sentiment":"bullish" if b>r else "bearish" if r>b else "neutral"})
        return out
    except Exception:
        return []


# ── STOCK PREDICTION MODEL ────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for stock price regression."""
    df = df.copy()
    c = df["Close"]; h = df["High"]; l = df["Low"]
    v = df["Volume"].replace(0,1); o = df["Open"]

    df["rsi"]      = ta.momentum.RSIIndicator(c,14).rsi()
    df["macd"]     = ta.trend.MACD(c).macd_diff()
    bb             = ta.volatility.BollingerBands(c)
    df["bb_pos"]   = (c-bb.bollinger_lband())/(bb.bollinger_hband()-bb.bollinger_lband()+1e-9)
    df["atr_pct"]  = ta.volatility.AverageTrueRange(h,l,c).average_true_range()/c*100
    df["vol_r"]    = v/(v.rolling(20).mean()+1e-9)
    ha_c           = (o+h+l+c)/4
    ha_o           = ha_c.copy()
    for i in range(1,len(df)): ha_o.iloc[i]=(ha_o.iloc[i-1]+ha_c.iloc[i-1])/2
    df["ha_trend"] = ((ha_c>ha_o).astype(float).rolling(3).sum()-1.5)/1.5
    df["ret1"]     = c.pct_change(1)
    df["ret5"]     = c.pct_change(5)
    df.dropna(inplace=True)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    return df


FEAT_COLS = ["rsi","macd","bb_pos","atr_pct","vol_r","ha_trend","ret1","ret5"]


def predict_stock(ticker: str) -> dict:
    """Train regression on 1y daily data, predict next 5 days."""
    df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    if len(df) < 60:
        return {"error": "Not enough data"}

    df = _build_features(df)
    close = df["Close"]

    # Target: next day close - current close (in %)
    df["target"] = close.shift(-1)/close - 1
    df.dropna(inplace=True)

    X = df[FEAT_COLS]; y = df["target"]
    # Use positional indexing to avoid index alignment issues
    valid = X.notna().all(axis=1) & np.isfinite(X.values).all(axis=1)
    X = X.loc[valid]; y = y.loc[valid]

    if len(X) < 50:
        return {"error": "Not enough bars after feature computation"}

    # Train on all but last 30 days, test on last 30
    cutoff = max(30, len(X)-30)
    X_tr, X_te = X.iloc[:cutoff], X.iloc[cutoff:]
    y_tr, y_te = y.iloc[:cutoff], y.iloc[cutoff:]

    model = Pipeline([("sc",StandardScaler()),
                      ("reg",GradientBoostingRegressor(
                          n_estimators=200,learning_rate=0.05,max_depth=3,
                          min_samples_leaf=10,subsample=0.8,random_state=42))])
    model.fit(X_tr, y_tr)

    # MAE on test set
    pred_te = model.predict(X_te)
    mae_pct = float(np.mean(np.abs(y_te.values - pred_te)))
    dir_acc = float(((pred_te>0)==(y_te.values>0)).mean())

    # Predict all historical bars (for chart overlay)
    pred_all      = model.predict(X)
    actual_prices = df.loc[X.index, "Close"].values
    pred_prices   = actual_prices * (1 + pred_all)
    timestamps    = [str(t) for t in X.index]

    # Predict next 5 days
    curr_price = float(close.iloc[-1])
    future_prices = [curr_price]
    last_row = X.iloc[-1].values.copy()
    for _ in range(5):
        delta = float(model.predict(last_row.reshape(1,-1))[0])
        next_p = future_prices[-1] * (1 + delta)
        future_prices.append(round(next_p, 2))
    future_prices = future_prices[1:]

    # Rolling MAE band (in price points)
    errors      = np.abs(actual_prices - pred_prices)
    rolling_mae = pd.Series(errors).rolling(10).mean().fillna(float(errors.mean())).values
    mae_pts     = float(curr_price * mae_pct)

    return {
        "mae_pct":      round(mae_pct*100, 2),
        "mae_pts":      round(mae_pts, 1),
        "dir_accuracy": round(dir_acc*100, 1),
        "chart_data": {
            "timestamps": timestamps,
            "actual":     [round(float(p),2) for p in actual_prices],
            "predicted":  [round(float(p),2) for p in pred_prices],
            "mae_upper":  [round(float(actual_prices[i]+rolling_mae[i]),2) for i in range(len(actual_prices))],
            "mae_lower":  [round(float(actual_prices[i]-rolling_mae[i]),2) for i in range(len(actual_prices))],
        },
        "next_5_days": future_prices,
        "signal": "UP" if future_prices[-1] > curr_price else "DOWN",
        "expected_move_pct": round((future_prices[-1]-curr_price)/curr_price*100, 2),
    }


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status":"ok","port":8003}


@app.get("/top-movers")
def top_movers():
    """Fetch each ticker individually — avoids MultiIndex issues."""
    results = []
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="5d", interval="1d",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if len(df) < 2:
                continue
            curr = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2])
            chg  = round((curr-prev)/prev*100, 2)
            # 52w
            df1y = yf.download(ticker, period="1y", interval="1d",
                               progress=False, auto_adjust=True)
            if isinstance(df1y.columns, pd.MultiIndex):
                df1y.columns = df1y.columns.get_level_values(0)
            df1y = df1y.dropna()
            w52h = round(float(df1y["High"].max()),2) if not df1y.empty else curr
            w52l = round(float(df1y["Low"].min()),2)  if not df1y.empty else curr
            results.append({
                "symbol":      ticker.replace(".NS",""),
                "name":        _name(ticker),
                "price":       round(curr,2),
                "change_pct":  chg,
                "week52_high": w52h,
                "week52_low":  w52l,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    return {
        "gainers":       results[:10],
        "losers":        list(reversed(results[-10:])),
        "total_fetched": len(results),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/stock/{symbol}")
def get_stock(symbol: str):
    ticker = f"{symbol.upper()}.NS"
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info or {}
        df   = yf.download(ticker, period="3mo", interval="1d",
                           progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty:
            raise HTTPException(404, f"No data for {symbol}")

        curr = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df)>1 else curr
        chg  = round((curr-prev)/prev*100, 2)

        df1y = yf.download(ticker, period="1y", interval="1d",
                           progress=False, auto_adjust=True)
        if isinstance(df1y.columns, pd.MultiIndex): df1y.columns = df1y.columns.get_level_values(0)
        df1y = df1y.dropna()
        w52h = round(float(df1y["High"].max()),2) if not df1y.empty else curr
        w52l = round(float(df1y["Low"].min()),2)  if not df1y.empty else curr

        # Signal
        ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
        rsi  = _rsi(df["Close"])
        if curr > ma20 and rsi < 65:
            sig, sig_reason = "BULLISH", f"Price above MA20 ({ma20:,.0f}), RSI {rsi:.0f}"
        elif curr < ma20 and rsi > 35:
            sig, sig_reason = "BEARISH", f"Price below MA20 ({ma20:,.0f}), RSI {rsi:.0f}"
        else:
            sig, sig_reason = "NEUTRAL", f"MA20: {ma20:,.0f}, RSI: {rsi:.0f}"

        # Prediction
        try:
            pred = predict_stock(ticker)
        except Exception as e:
            pred = {"error": str(e)}

        name = info.get("longName") or _name(ticker)
        news = _news(name)

        return {
            "symbol":      symbol.upper(),
            "name":        name,
            "price":       round(curr,2),
            "change":      round(curr-prev,2),
            "change_pct":  chg,
            "week52_high": w52h,
            "week52_low":  w52l,
            "pe_ratio":    round(info.get("trailingPE") or 0, 1) or None,
            "market_cap":  info.get("marketCap"),
            "sector":      info.get("sector","N/A"),
            "industry":    info.get("industry","N/A"),
            "signal":      sig,
            "signal_reason":sig_reason,
            "prediction":  pred,
            "news":        news,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/search")
def search(q: str = Query(..., min_length=1)):
    q_l = q.lower()
    out = []
    for t in TICKERS:
        base = t.replace(".NS","")
        name = _name(t)
        if q_l in base.lower() or q_l in name.lower():
            try:
                df = yf.download(t, period="2d", interval="1d",
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.dropna()
                curr = float(df["Close"].iloc[-1]) if not df.empty else 0
                prev = float(df["Close"].iloc[-2]) if len(df)>1 else curr
                chg  = round((curr-prev)/prev*100,2) if prev else 0
            except Exception:
                curr = chg = 0
            out.append({"symbol":base,"name":name,"price":round(curr,2),"change_pct":chg})
    return {"results":out,"query":q,"count":len(out)}
