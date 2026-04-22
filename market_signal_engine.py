"""
SignalX Engine v5 — Multi-Timeframe Binary Predictor
=====================================================
Key changes from v4:
  - Multi-timeframe analysis: 5m, 15m, 30m, 1h
  - Alert fires when 3+ timeframes agree on direction
  - One model per index × timeframe (12 models total)
  - New /alert/{index} endpoint
  - /candles/{index}?tf=15m&bars=100 supports timeframe selection
  - All v4 endpoints preserved

RUN:
    uvicorn market_signal_engine:app --port 8001
"""

import os, warnings, logging
from datetime import datetime, timedelta, date
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import ta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from dotenv import load_dotenv

try:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
except Exception:
    tv = None

try:
    import nsepython
    _HAS_NSE = True
except Exception:
    _HAS_NSE = False

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SignalX Engine", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS       = {"nifty": "^NSEI", "banknifty": "^NSEBANK", "sensex": "^BSESN"}
VIX_TICKER    = "^INDIAVIX"
GOLD_TICKER   = "GC=F"
NQ_TICKER     = "NQ=F"
USDINR_TICKER = "USDINR=X"
NEWS_API_KEY  = os.getenv("NEWS_API_KEY", "")
NSE_EXPIRY    = os.getenv("NSE_EXPIRY_DATE", "2026-04-24")

TIMEFRAMES = {
    "5m":  {"interval": "5m",  "period": "5d",  "forward": 3,  "label": "5 min"},
    "15m": {"interval": "15m", "period": "30d", "forward": 2,  "label": "15 min"},
    "30m": {"interval": "30m", "period": "60d", "forward": 2,  "label": "30 min"},
    "1h":  {"interval": "1h",  "period": "2y",  "forward": 2,  "label": "1 hour"},
}
ALERT_MIN_TIMEFRAMES = 3   # need 3+ TFs agreeing to fire alert

FEATURE_COLS = ["rsi_14", "macd_hist", "price_vs_vwap",
                "bb_position", "volume_ratio", "atr_pct",
                "ha_trend", "macd_accel", "price_decay", "dte"]

CONFIDENCE_THRESHOLD = 0.55
MIN_BARS_REQUIRED    = 300

_models:       dict = {}   # keyed by "index_tf" e.g. "nifty_15m"
_last_trained: dict = {}
_train_meta:   dict = {}
_pred_log:     dict = {k: [] for k in TICKERS}
_daily_models: dict = {}
_reg_models:   dict = {}   # regression models for price forecasting


# ── SECTION 1: DATA ───────────────────────────────────────────────────────────

def fetch_ohlcv(ticker: str, period: str = "60d", interval: str = "15m") -> pd.DataFrame:
    tv_map = {"^NSEI":("NIFTY","NSE"), "^NSEBANK":("BANKNIFTY","NSE"), "^BSESN":("SENSEX","BSE")}
    if tv is not None and ticker in tv_map:
        sym, exc = tv_map[ticker]
        tv_int   = {"15m": Interval.in_15_minute, "1d": Interval.in_daily}
        n_bars   = 5000 if "y" in period else 2000
        try:
            df = tv.get_hist(symbol=sym, exchange=exc,
                             interval=tv_int.get(interval, Interval.in_15_minute),
                             n_bars=n_bars)
            if df is not None and not df.empty:
                df.rename(columns={"open":"Open","high":"High","low":"Low",
                                   "close":"Close","volume":"Volume"}, inplace=True)
                df.drop(columns=[c for c in ["symbol"] if c in df.columns], inplace=True)
                return df
        except Exception as e:
            logger.warning(f"tvDatafeed failed: {e}")

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def fetch_global_signals() -> dict:
    out = {"nq_change_pct": 0.0, "usdinr_change": 0.0, "gold_change_pct": 0.0}
    for key, tkr, col, mode in [
        ("nq_change_pct",   NQ_TICKER,     "Close", "pct"),
        ("usdinr_change",   USDINR_TICKER, "Close", "diff"),
        ("gold_change_pct", GOLD_TICKER,   "Close", "pct"),
    ]:
        try:
            d = yf.download(tkr, period="5d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            d = d.dropna()
            if len(d) >= 2:
                a, b = float(d[col].iloc[-2]), float(d[col].iloc[-1])
                out[key] = round((b-a)/a*100, 4) if mode == "pct" else round(b-a, 4)
        except Exception:
            pass
    return out


def fetch_vix() -> float:
    try:
        v = yf.download(VIX_TICKER, period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(v.columns, pd.MultiIndex): v.columns = v.columns.get_level_values(0)
        return float(v["Close"].dropna().iloc[-1])
    except Exception:
        return 15.0


def fetch_news_sentiment(query: str = "Nifty NSE India stock market") -> dict:
    empty = {"sentiment_score": 0.0, "top_headline": "N/A", "headlines": []}
    if not NEWS_API_KEY:
        return empty
    BULL = ["rally","surge","jump","gain","bullish","rate cut","stimulus","fii buying",
            "growth","profit","beat estimates","upgrade","infrastructure"]
    BEAR = ["fall","drop","crash","decline","bearish","rate hike","war","tension",
            "fii selling","recession","inflation","miss estimates","downgrade","geopolitical"]
    try:
        since = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        url   = (f"https://newsapi.org/v2/everything?q={query}&from={since}"
                 f"&sortBy=publishedAt&language=en&pageSize=20&apiKey={NEWS_API_KEY}")
        arts  = requests.get(url, timeout=5).json().get("articles", [])
        if not arts:
            return empty
        bt = br = 0
        heads = []
        for a in arts:
            t   = (a.get("title") or "") + " " + (a.get("description") or "")
            txt = t.lower()
            b   = sum(1 for k in BULL if k in txt)
            r   = sum(1 for k in BEAR if k in txt)
            bt += b; br += r
            heads.append({"title": a.get("title",""), "source": a.get("source",{}).get("name",""),
                          "published": (a.get("publishedAt","")[:16]).replace("T"," "),
                          "sentiment": "bullish" if b>r else "bearish" if r>b else "neutral"})
        tot   = bt + br
        score = round(max(-1.0, min(1.0, (bt-br)/tot if tot else 0.0)), 3)
        return {"sentiment_score": score, "top_headline": heads[0]["title"] if heads else "N/A",
                "headlines": heads[:5]}
    except Exception as e:
        logger.warning(f"News failed: {e}")
        return empty


def fetch_options_context(index: str) -> dict:
    if not _HAS_NSE or index == "sensex":
        return {"pcr": 1.0, "status": "N/A"}
    try:
        sym = "NIFTY" if index == "nifty" else "BANKNIFTY"
        oc  = nsepython.option_chain(sym)
        pe  = oc["filtered"]["PE"]["totOI"]
        ce  = oc["filtered"]["CE"]["totOI"]
        return {"pcr": round(pe/ce, 2) if ce else 1.0, "status": "Live"}
    except Exception:
        return {"pcr": 1.0, "status": "Fallback"}


# ── SECTION 2: FEATURES ───────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    10 features combining original 6 + Heikin-Ashi trend + MACD acceleration
    + price decay + days to expiry.

    ha_trend    — Heikin-Ashi consecutive bar count (-3 to +3): filters noise
    macd_accel  — MACD histogram change: is momentum growing or fading?
    price_decay — last bar return vs 3-bar avg: catches momentum exhaustion
    dte         — days to weekly Thursday expiry: theta decay awareness
    """
    df     = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"].replace(0, 1)
    op     = df["Open"]

    # Original 6
    df["rsi_14"]        = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd_obj            = ta.trend.MACD(close)
    df["macd_hist"]     = macd_obj.macd_diff()
    vwap                = (close * volume).cumsum() / volume.cumsum()
    df["price_vs_vwap"] = (close - vwap) / vwap * 100
    bb                  = ta.volatility.BollingerBands(close)
    df["bb_position"]   = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
    df["volume_ratio"]  = volume / (volume.rolling(20).mean() + 1e-9)
    df["atr_pct"]       = ta.volatility.AverageTrueRange(high, low, close).average_true_range() / close * 100

    # Heikin-Ashi trend score
    ha_close = (op + high + low + close) / 4
    ha_open  = ha_close.copy()
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_green = (ha_close > ha_open).astype(float)
    # Rolling sum of last 3 bars: +3=all green, 0=all red, normalized to -1..+1
    df["ha_trend"] = (ha_green.rolling(3).sum() - 1.5) / 1.5

    # MACD acceleration (is histogram growing or shrinking?)
    df["macd_accel"] = df["macd_hist"].diff(1)

    # Price decay
    df["price_decay"] = close.pct_change(1) * 100 - close.pct_change(3) * 100 / 3

    # Days to expiry
    def _dte(ts):
        d = ts.date() if hasattr(ts, "date") else ts
        ahead = 3 - d.weekday()
        return ahead if ahead > 0 else ahead + 7
    try:
        idx = df.index.tz_convert("Asia/Kolkata") if (hasattr(df.index, "tz") and df.index.tz) else df.index
        df["dte"] = [_dte(t) for t in idx]
    except Exception:
        df["dte"] = 3

    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def compute_ha_signal(df: pd.DataFrame) -> dict:
    """
    Pure rule-based Heikin-Ashi + MACD signal.
    No ML — just the strategy logic.
    Returns signal, strength, and reason.
    """
    if len(df) < 20:
        return {"signal": "WAIT", "strength": 0, "reason": "Not enough data"}

    op    = df["Open"]
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    # Compute HA
    ha_close = (op + high + low + close) / 4
    ha_open  = ha_close.copy()
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
    ha_green = ha_close > ha_open

    # Last 2 bars
    g0 = bool(ha_green.iloc[-1])
    g1 = bool(ha_green.iloc[-2])
    g2 = bool(ha_green.iloc[-3]) if len(df) > 2 else g1

    # MACD
    hist = ta.trend.MACD(close).macd_diff()
    macd_now  = float(hist.iloc[-1])
    macd_prev = float(hist.iloc[-2])
    macd_bull = macd_now > 0
    macd_cross_up   = macd_now > 0 and macd_prev <= 0
    macd_cross_down = macd_now < 0 and macd_prev >= 0

    # RSI
    rsi = float(ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1])

    # Volume
    vol = df["Volume"].replace(0, 1)
    vol_ratio = float(vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-9))

    reasons = []
    strength = 0

    if g0 and g1:
        strength += 2; reasons.append("2 consecutive HA green bars")
        if g2:
            strength += 1; reasons.append("3rd HA green bar — strong uptrend")
    elif not g0 and not g1:
        strength -= 2; reasons.append("2 consecutive HA red bars")
        if not g2:
            strength -= 1; reasons.append("3rd HA red bar — strong downtrend")

    if macd_cross_up:
        strength += 2; reasons.append("MACD just crossed UP")
    elif macd_bull:
        strength += 1; reasons.append("MACD positive")
    elif macd_cross_down:
        strength -= 2; reasons.append("MACD just crossed DOWN")
    else:
        strength -= 1; reasons.append("MACD negative")

    if 40 < rsi < 65:
        strength += 1; reasons.append(f"RSI {rsi:.0f} bullish zone")
    elif rsi > 70:
        strength -= 1; reasons.append(f"RSI {rsi:.0f} overbought")
    elif rsi < 35:
        strength += 1; reasons.append(f"RSI {rsi:.0f} oversold")

    if vol_ratio > 1.3:
        strength = strength + 1 if strength > 0 else strength - 1
        reasons.append(f"Volume {vol_ratio:.1f}× avg — conviction")

    if strength >= 3:
        return {"signal": "UP",   "strength": strength, "reasons": reasons}
    elif strength <= -3:
        return {"signal": "DOWN", "strength": abs(strength), "reasons": reasons}
    else:
        return {"signal": "WAIT", "strength": abs(strength), "reasons": reasons}


def compute_support_resistance(df: pd.DataFrame) -> dict:
    try:
        df_r = df.copy()
        if hasattr(df_r.index, "tz") and df_r.index.tz is not None:
            df_r.index = df_r.index.tz_convert("Asia/Kolkata").tz_localize(None)
        daily = df_r["Close"].resample("1D").ohlc().dropna()
        if len(daily) >= 2:
            prev = daily.iloc[-2]
            H, L, C = float(prev["high"]), float(prev["low"]), float(prev["close"])
            pivot = round((H+L+C)/3, 2)
            r1 = round(2*pivot-L, 2); r2 = round(pivot+(H-L), 2); r3 = round(H+2*(pivot-L), 2)
            s1 = round(2*pivot-H, 2); s2 = round(pivot-(H-L), 2); s3 = round(L-2*(H-pivot), 2)
        else:
            raise ValueError()
    except Exception:
        cp = float(df["Close"].iloc[-1])
        pivot = cp
        r1,r2,r3 = round(cp*1.003,2), round(cp*1.006,2), round(cp*1.009,2)
        s1,s2,s3 = round(cp*0.997,2), round(cp*0.994,2), round(cp*0.991,2)

    recent = df.tail(20)
    cp     = float(df["Close"].iloc[-1])
    mag    = 10 ** (len(str(int(cp))) - 2)
    return {
        "pivot": pivot,
        "resistance": {"r1": r1, "r2": r2, "r3": r3},
        "support":    {"s1": s1, "s2": s2, "s3": s3},
        "swing_high": round(float(recent["High"].max()), 2),
        "swing_low":  round(float(recent["Low"].min()), 2),
        "round_above": round(round(cp/mag+0.5)*mag, 2),
        "round_below": round(round(cp/mag-0.5)*mag, 2),
    }


# ── SECTION 3: BINARY MODEL ───────────────────────────────────────────────────

def create_binary_labels(df: pd.DataFrame, forward_bars: int = 2) -> pd.Series:
    """True binary: 1 if next close > current, 0 otherwise."""
    future_ret = df["Close"].shift(-forward_bars) / df["Close"] - 1
    labels = pd.Series(0, index=df.index, name="label")
    labels[future_ret > 0] = 1
    return labels


def build_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        )),
    ])


def train_model(index_name: str, tf_key: str = "15m") -> dict:
    """Train one model for one timeframe, store in _models['{index}_{tf}']."""
    ticker = TICKERS[index_name]
    tf_cfg = TIMEFRAMES[tf_key]
    model_key = f"{index_name}_{tf_key}"
    logger.info(f"Training {model_key}...")

    for period in [tf_cfg["period"], "60d"]:
        try:
            df = fetch_ohlcv(ticker, period=period, interval=tf_cfg["interval"])
            if len(df) >= MIN_BARS_REQUIRED:
                break
        except Exception:
            continue
    else:
        # Last resort: shorter period
        try:
            df = fetch_ohlcv(ticker, period="30d", interval=tf_cfg["interval"])
        except Exception as e:
            raise ValueError(f"Could not fetch data for {model_key}: {e}")

    df     = compute_features(df)
    labels = create_binary_labels(df, forward_bars=tf_cfg["forward"])

    valid = labels.index.intersection(df.index)
    X     = df.loc[valid, FEATURE_COLS].iloc[:-tf_cfg["forward"]].copy()
    y     = labels.loc[valid].iloc[:-tf_cfg["forward"]].copy()
    mask  = X.notna().all(axis=1)
    X, y  = X[mask], y[mask]

    if len(X) < 50:
        raise ValueError(f"Not enough bars for {model_key}: {len(X)}")

    n_splits = min(5, max(2, len(X) // 100))
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    base     = build_model()
    cv_scores = cross_val_score(base, X, y, cv=tscv, scoring="accuracy")

    base.fit(X, y)
    cv_val = min(3, n_splits)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=cv_val)
    cal.fit(X, y)

    _models[model_key]       = cal
    _last_trained[model_key] = datetime.now()

    result = {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std":  round(float(cv_scores.std()), 4),
        "trained_on_bars":  len(X),
        "trained_at":       _last_trained[model_key].isoformat(),
    }
    _train_meta[model_key] = result
    logger.info(f"{model_key} CV: {cv_scores.mean():.2%}±{cv_scores.std():.2%}")
    return result


def _walk_forward_backtest(X: pd.DataFrame, y: pd.Series) -> dict:
    tscv = TimeSeriesSplit(n_splits=5)
    rets, correct, total = [], 0, 0
    for tr, te in tscv.split(X):
        m = build_model(); m.fit(X.iloc[tr], y.iloc[tr])
        probas  = m.predict_proba(X.iloc[te])
        classes = list(m.classes_)
        up_idx  = classes.index(1) if 1 in classes else 1
        for i, p in enumerate(probas):
            conf = p[up_idx]
            pred = 1 if conf >= CONFIDENCE_THRESHOLD else 0
            if conf < CONFIDENCE_THRESHOLD and conf > (1 - CONFIDENCE_THRESHOLD):
                continue
            actual = int(y.iloc[te[i]])
            total += 1
            if pred == actual:
                correct += 1; rets.append(1.0)
            else:
                rets.append(-1.0)
    if not rets:
        return {"win_rate": 0.0, "sharpe": 0.0, "total_signals": 0}
    arr      = np.array(rets)
    win_rate = correct / total if total else 0.0
    sharpe   = (arr.mean() / (arr.std() + 1e-9)) * np.sqrt(6 * 250)
    return {"win_rate": round(win_rate, 4), "sharpe": round(float(sharpe), 3),
            "total_signals": int(total)}


# ── SECTION 4: MULTI-TIMEFRAME ANALYSIS ──────────────────────────────────────

def analyze_timeframe(index: str, tf_key: str, df: pd.DataFrame = None) -> dict:
    """
    Analyze one timeframe for one index.
    Returns signal, confidence, conditions, price info.
    """
    tf_cfg    = TIMEFRAMES[tf_key]
    model_key = f"{index}_{tf_key}"
    ticker    = TICKERS[index]

    if df is None:
        df = fetch_ohlcv(ticker, period=tf_cfg["period"], interval=tf_cfg["interval"])

    df = compute_features(df)
    sr = compute_support_resistance(df)

    curr_price = float(df["Close"].iloc[-1])
    atr_pts    = float(df["atr_pct"].iloc[-1]) / 100 * curr_price
    fv         = df[FEATURE_COLS].iloc[-1]

    # ML prediction
    up_prob = dn_prob = 0.5
    if model_key in _models:
        try:
            model     = _models[model_key]
            proba_arr = model.predict_proba(fv.values.reshape(1, -1))[0]
            classes   = model.classes_.tolist()
            up_idx    = classes.index(1) if 1 in classes else 1
            dn_idx    = classes.index(0) if 0 in classes else 0
            up_prob   = float(proba_arr[up_idx])
            dn_prob   = float(proba_arr[dn_idx])
        except Exception as e:
            logger.warning(f"ML predict failed {model_key}: {e}")

    conditions_bullish = []
    conditions_bearish = []

    # 1. ML vote
    if up_prob > 0.52:
        conditions_bullish.append(f"ML leans UP ({up_prob*100:.0f}%)")
    elif dn_prob > 0.52:
        conditions_bearish.append(f"ML leans DOWN ({dn_prob*100:.0f}%)")

    # 2. MACD histogram
    if float(fv["macd_hist"]) > 0:
        conditions_bullish.append("MACD histogram positive")
    else:
        conditions_bearish.append("MACD histogram negative")

    # 3. RSI zone
    rsi = float(fv["rsi_14"])
    if 45 < rsi < 65:
        conditions_bullish.append(f"RSI {rsi:.0f} in bullish zone")
    elif rsi < 40:
        conditions_bullish.append(f"RSI {rsi:.0f} oversold — bounce likely")
    elif rsi > 70:
        conditions_bearish.append(f"RSI {rsi:.0f} overbought")
    elif 35 < rsi < 55:
        conditions_bearish.append(f"RSI {rsi:.0f} in bearish zone")

    # 4. Price vs VWAP
    pvwap = float(fv["price_vs_vwap"])
    if pvwap > 0.05:
        conditions_bullish.append(f"Price {pvwap:+.2f}% above VWAP")
    elif pvwap < -0.05:
        conditions_bearish.append(f"Price {pvwap:+.2f}% below VWAP")

    # 5. Bollinger position
    bbp = float(fv["bb_position"])
    if bbp < 0.35:
        conditions_bullish.append(f"BB position {bbp:.2f} — near lower band (oversold)")
    elif bbp > 0.65:
        conditions_bearish.append(f"BB position {bbp:.2f} — near upper band (overbought)")

    # 6. Price decay (momentum fading = reversal signal)
    decay = float(fv["price_decay"])
    if decay < -0.05:   # last bar weaker than 3-bar avg → momentum fading → reversal up
        conditions_bullish.append(f"Price decay {decay:.3f} — downward momentum fading")
    elif decay > 0.05:  # last bar stronger than avg → momentum fading → reversal down
        conditions_bearish.append(f"Price decay {decay:.3f} — upward momentum fading")

    # 7. Expiry proximity (theta decay accelerates near expiry)
    dte = float(fv["days_to_expiry"])
    adx = float(fv.get("adx_14", 20)) if "adx_14" in fv.index else 20
    trend_exists = True  # always trade, use volume_ratio as filter instead
    vol_r = float(fv["volume_ratio"])
    if vol_r > 1.3:
        conditions_bullish.append(f"Volume {vol_r:.1f}× avg — conviction move") if float(fv["macd_hist"]) > 0 else conditions_bearish.append(f"Volume {vol_r:.1f}× avg — conviction move")
    if dte <= 1:
        conditions_bearish.append(f"Expiry tomorrow — theta risk high, avoid buying options")

    n_up   = len(conditions_bullish)
    n_down = len(conditions_bearish)

    if n_up >= 4 and trend_exists and n_up > n_down:
        signal     = "UP"
        confidence = round(min(95, 50 + n_up * 8 + (adx - 20) * 0.5), 1)
    elif n_down >= 4 and trend_exists and n_down > n_up:
        signal     = "DOWN"
        confidence = round(min(95, 50 + n_down * 8 + (adx - 20) * 0.5), 1)
    else:
        signal     = "WAIT"
        confidence = round(max(up_prob, dn_prob) * 100, 1)

    return {
        "tf":                 tf_key,
        "label":              tf_cfg["label"],
        "signal":             signal,
        "confidence":         confidence,
        "conditions_met":     max(n_up, n_down),
        "conditions_bullish": conditions_bullish,
        "conditions_bearish": conditions_bearish,
        "adx":                round(adx, 1),
        "trend_exists":       trend_exists,
        "curr_price":         round(curr_price, 2),
        "atr_pts":            round(atr_pts, 2),
        "up_prob":            round(up_prob * 100, 1),
        "dn_prob":            round(dn_prob * 100, 1),
        "sr":                 sr,
    }


def generate_alert(index: str, analyses: list) -> dict:
    """
    Takes list of analyze_timeframe results.
    Fires alert when 3+ TFs agree on UP or DOWN.
    """
    vix = fetch_vix()
    step = 100 if index == "banknifty" else 50

    up_tfs   = [a for a in analyses if a["signal"] == "UP"]
    down_tfs = [a for a in analyses if a["signal"] == "DOWN"]
    total    = len(analyses)

    # Use the most recent (highest-detail) TF for price/ATR
    ref = analyses[-1] if analyses else {}
    curr_price = ref.get("curr_price", 0)
    atr_pts    = ref.get("atr_pts", curr_price * 0.005)
    sr         = ref.get("sr", {})
    atm        = round(curr_price / step) * step

    if vix > 25:
        lot_advice = "1 lot only (VIX >25)"
    elif vix > 20:
        lot_advice = "Max 2 lots (VIX elevated)"
    else:
        lot_advice = "Normal size ok"

    if len(up_tfs) >= ALERT_MIN_TIMEFRAMES:
        agreeing   = [a["tf"] for a in up_tfs]
        direction  = "UP"
        avg_conf   = round(sum(a["confidence"] for a in up_tfs) / len(up_tfs), 1)
        price_target  = round(curr_price + atr_pts * 1.5, 0)
        invalidation  = round(curr_price - atr_pts * 0.8, 0)
        option_action = f"BUY {atm} CE"
        entry_zone    = f"Buy between {curr_price - atr_pts*0.3:,.0f} and {curr_price:,.0f}"
        message       = f"{len(up_tfs)}/{total} timeframes agree: BULLISH setup forming"
    elif len(down_tfs) >= ALERT_MIN_TIMEFRAMES:
        agreeing   = [a["tf"] for a in down_tfs]
        direction  = "DOWN"
        avg_conf   = round(sum(a["confidence"] for a in down_tfs) / len(down_tfs), 1)
        price_target  = round(curr_price - atr_pts * 1.5, 0)
        invalidation  = round(curr_price + atr_pts * 0.8, 0)
        option_action = f"BUY {atm + step} PE"
        entry_zone    = f"Buy between {curr_price:,.0f} and {curr_price + atr_pts*0.3:,.0f}"
        message       = f"{len(down_tfs)}/{total} timeframes agree: BEARISH setup forming"
    else:
        return {
            "alert":               False,
            "direction":           None,
            "timeframes_agreeing": [],
            "timeframes_total":    total,
            "confidence":          0,
            "message":             f"No consensus — monitoring ({len(up_tfs)} UP, {len(down_tfs)} DOWN)",
            "price_target":        None,
            "invalidation":        None,
            "option_action":       None,
            "entry_zone":          None,
            "hold_time":           None,
            "lot_advice":          lot_advice,
        }

    return {
        "alert":               True,
        "direction":           direction,
        "timeframes_agreeing": agreeing,
        "timeframes_total":    total,
        "confidence":          avg_conf,
        "message":             message,
        "price_target":        price_target,
        "invalidation":        invalidation,
        "option_action":       option_action,
        "entry_zone":          entry_zone,
        "hold_time":           "1-3 hours. Exit by 2:30 PM.",
        "lot_advice":          lot_advice,
    }


# ── SECTION 5: OPTION INSTRUCTIONS + LOSS LEARNING ───────────────────────────

def generate_option_instructions(signal: str, confidence: float, curr_price: float,
                                  atr_pts: float, sr: dict, vix: float,
                                  index: str) -> dict:
    step = 100 if index == "banknifty" else 50
    atm  = round(curr_price / step) * step

    if vix > 25:
        lot_advice = "Trade 1 lot only (VIX >25 = high risk)"
    elif vix > 20:
        lot_advice = "Max 2 lots (VIX elevated)"
    else:
        lot_advice = "Normal position size ok"

    if signal == "UP":
        strike     = atm
        option     = "CE (Call)"
        entry_note = f"Buy {strike} CE when price pulls back to {curr_price - atr_pts*0.3:,.0f}–{curr_price:,.0f}"
        target     = round(curr_price + atr_pts * 1.5, 0)
        sl_price   = round(curr_price - atr_pts * 0.8, 0)
        sl_note    = f"Exit if price falls below {sl_price:,.0f}"
        target_note= f"Book profit near {target:,.0f} or R1 ({sr['resistance']['r1']:,.0f})"
    else:
        strike     = atm
        option     = "PE (Put)"
        entry_note = f"Buy {strike} PE when price bounces to {curr_price:,.0f}–{curr_price + atr_pts*0.3:,.0f}"
        target     = round(curr_price - atr_pts * 1.5, 0)
        sl_price   = round(curr_price + atr_pts * 0.8, 0)
        sl_note    = f"Exit if price rises above {sl_price:,.0f}"
        target_note= f"Book profit near {target:,.0f} or S1 ({sr['support']['s1']:,.0f})"

    return {
        "action":    f"BUY {strike} {option}",
        "entry":     entry_note,
        "target":    target_note,
        "stop_loss": sl_note,
        "lot_sizing":lot_advice,
        "hold_time": "15–30 min max. Exit before 3:15 PM.",
        "confidence":f"{confidence:.0f}%",
    }


def log_prediction(index: str, timestamp: str, signal: str,
                   confidence: float, price: float) -> None:
    log = _pred_log.setdefault(index, [])
    log.append({"ts": timestamp, "signal": signal,
                "confidence": confidence, "price_at_signal": price,
                "actual_next_price": None})
    if len(log) > 50:
        _pred_log[index] = log[-50:]


def update_actuals(index: str, current_price: float) -> None:
    log = _pred_log.get(index, [])
    for entry in reversed(log):
        if entry["actual_next_price"] is None:
            entry["actual_next_price"] = current_price
            was_correct = (
                (entry["signal"] == "UP"   and current_price > entry["price_at_signal"]) or
                (entry["signal"] == "DOWN" and current_price < entry["price_at_signal"])
            )
            entry["correct"] = was_correct
            break


def get_recent_accuracy(index: str) -> dict:
    log = [e for e in _pred_log.get(index, []) if e.get("actual_next_price") is not None]
    if not log:
        return {"recent_accuracy": None, "sample_size": 0}
    correct = sum(1 for e in log if e.get("correct"))
    return {"recent_accuracy": round(correct / len(log), 3), "sample_size": len(log)}


# ── SECTION 6: DAILY / SWING MODEL ───────────────────────────────────────────

DAILY_FEATURES = ["rsi_14", "macd_hist", "price_vs_vwap", "bb_position",
                  "volume_ratio", "atr_pct", "nq_ret", "vix_level", "day_of_week"]


def _fetch_daily_df(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No daily data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df


def _compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    df    = df.copy()
    close = df["Close"]; high = df["High"]; low = df["Low"]
    vol   = df["Volume"].replace(0, 1)

    df["rsi_14"]       = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd_hist"]    = ta.trend.MACD(close).macd_diff()
    vwap               = (close * vol).cumsum() / vol.cumsum()
    df["price_vs_vwap"]= (close - vwap) / vwap * 100
    bb                 = ta.volatility.BollingerBands(close)
    df["bb_position"]  = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
    df["volume_ratio"] = vol / (vol.rolling(20).mean() + 1e-9)
    df["atr_pct"]      = ta.volatility.AverageTrueRange(high, low, close).average_true_range() / close * 100

    try:
        nq = yf.download(NQ_TICKER, period="5y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(nq.columns, pd.MultiIndex): nq.columns = nq.columns.get_level_values(0)
        nq_ret = nq["Close"].pct_change()
        df["nq_ret"] = nq_ret.reindex(df.index, method="ffill").values[:len(df)]
    except Exception:
        df["nq_ret"] = 0.0

    try:
        vix = yf.download(VIX_TICKER, period="5y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)
        df["vix_level"] = vix["Close"].reindex(df.index, method="ffill").values[:len(df)]
    except Exception:
        df["vix_level"] = 15.0

    df["day_of_week"] = df.index.dayofweek.astype(float) if hasattr(df.index, "dayofweek") else 2.0
    df.dropna(inplace=True)
    return df


def _train_daily_model(index_name: str) -> None:
    ticker = TICKERS[index_name]
    logger.info(f"Training daily model: {index_name}...")
    df  = _fetch_daily_df(ticker)
    df  = _compute_daily_features(df)
    fut = df["Close"].shift(-1) / df["Close"] - 1
    y   = pd.Series((fut > 0).astype(int), index=df.index)
    cols = [c for c in DAILY_FEATURES if c in df.columns]
    X   = df[cols].iloc[:-1].copy()
    y   = y.iloc[:-1].copy()
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    base = Pipeline([("sc", StandardScaler()),
                     ("clf", GradientBoostingClassifier(
                         n_estimators=200, learning_rate=0.05, max_depth=3,
                         min_samples_leaf=20, subsample=0.8, random_state=42))])
    base.fit(X, y)
    cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
    cal.fit(X, y)
    _daily_models[index_name] = {"model": cal, "cols": cols}
    logger.info(f"Daily model ready: {index_name} ({len(X)} bars)")


def compute_overnight_risk(vix: float, nq_change: float,
                            usdinr_change: float, news_score: float) -> dict:
    score, reasons = 0.0, []
    if vix > 25:
        score += 40; reasons.append(f"VIX {vix:.1f} is HIGH — elevated gap risk")
    elif vix > 20:
        score += 25; reasons.append(f"VIX {vix:.1f} elevated")
    abs_nq = abs(nq_change)
    if abs_nq > 1.5:
        score += 30; reasons.append(f"NQ futures {nq_change:+.2f}% — large gap likely")
    elif abs_nq > 0.8:
        score += 18; reasons.append(f"NQ futures {nq_change:+.2f}% — moderate gap risk")
    elif abs_nq > 0.3:
        score += 8
    if usdinr_change > 0.5:
        score += 15; reasons.append("Rupee weakening — FII outflow risk")
    elif usdinr_change > 0.2:
        score += 8
    if news_score < -0.3:
        score += 15; reasons.append("Negative news sentiment")
    elif news_score < -0.1:
        score += 7
    score = min(100, score)
    if score >= 65:
        verdict, color = "DO NOT HOLD OVERNIGHT", "red"
    elif score >= 40:
        verdict, color = "RISKY — Reduce size if holding", "amber"
    else:
        verdict, color = "Moderate risk — ok with tight SL", "green"
    return {"score": round(score), "verdict": verdict, "color": color, "reasons": reasons}


# ── SECTION 7: API ENDPOINTS ──────────────────────────────────────────────────

@app.get("/price/{index}")
def price_endpoint(index: str):
    """Lightweight live price — just latest close, no ML. Safe to call every 20s."""
    if index not in TICKERS:
        raise HTTPException(404)
    try:
        df = yf.download(TICKERS[index], period="1d", interval="1m", progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("no data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        latest = df.dropna().iloc[-1]
        prev   = df.dropna().iloc[-2] if len(df) > 1 else latest
        return {
            "index":      index.upper(),
            "price":      round(float(latest["Close"]), 2),
            "open":       round(float(latest["Open"]), 2),
            "high":       round(float(df["High"].max()), 2),
            "low":        round(float(df["Low"].min()), 2),
            "prev_close": round(float(prev["Close"]), 2),
            "timestamp":  str(df.index[-1]),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": list(_models.keys()),
            "version": "5.0 — multi-timeframe binary",
            "time": datetime.now().isoformat()}


@app.get("/train/{index}")
def train_endpoint(index: str):
    if index not in TICKERS:
        raise HTTPException(404, f"Must be one of: {list(TICKERS.keys())}")
    try:
        results = {}
        for tf_key in TIMEFRAMES:
            try:
                results[tf_key] = train_model(index, tf_key)
            except Exception as e:
                results[tf_key] = {"error": str(e)}
        return {"status": "trained", "index": index, "timeframes": results}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/backtest/{index}")
def backtest_endpoint(index: str):
    if index not in TICKERS:
        raise HTTPException(404)
    acc = get_recent_accuracy(index)
    meta_out = {}
    for tf_key in TIMEFRAMES:
        key = f"{index}_{tf_key}"
        if key in _train_meta:
            meta_out[tf_key] = _train_meta[key]
    if not meta_out:
        raise HTTPException(400, f"Call /train/{index} first")
    return {
        "index":      index.upper(),
        "timeframes": meta_out,
        "live_accuracy": acc,
        "note": "Binary model: only UP/DOWN. Signals only when 4+ conditions align.",
    }


@app.get("/candles/{index}")
def candles_endpoint(index: str, tf: str = "15m", bars: int = 100):
    """Returns OHLCV candles for the requested timeframe."""
    if index not in TICKERS:
        raise HTTPException(404)
    if tf not in TIMEFRAMES:
        raise HTTPException(400, f"tf must be one of: {list(TIMEFRAMES.keys())}")
    try:
        tf_cfg = TIMEFRAMES[tf]
        df = fetch_ohlcv(TICKERS[index], period=tf_cfg["period"], interval=tf_cfg["interval"])
        df = df.tail(bars)
        return {
            "index": index.upper(),
            "tf":    tf,
            "label": tf_cfg["label"],
            "candles": [
                {"t": str(ts), "o": round(float(r.Open),2), "h": round(float(r.High),2),
                 "l": round(float(r.Low),2), "c": round(float(r.Close),2),
                 "v": int(r.Volume) if hasattr(r, "Volume") else 0}
                for ts, r in df.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/alert/{index}")
def alert_endpoint(index: str):
    """Run all 4 timeframe analyses and generate multi-TF alert."""
    if index not in TICKERS:
        raise HTTPException(404)
    try:
        analyses = []
        for tf_key in TIMEFRAMES:
            try:
                result = analyze_timeframe(index, tf_key)
                analyses.append(result)
            except Exception as e:
                logger.warning(f"analyze_timeframe failed {index}/{tf_key}: {e}")
                analyses.append({
                    "tf": tf_key, "label": TIMEFRAMES[tf_key]["label"],
                    "signal": "WAIT", "confidence": 0, "conditions_met": 0,
                    "conditions_bullish": [], "conditions_bearish": [],
                    "adx": 0, "trend_exists": False,
                    "curr_price": 0, "atr_pts": 0,
                    "up_prob": 50, "dn_prob": 50, "sr": {},
                    "error": str(e),
                })

        alert = generate_alert(index, analyses)

        # Fetch supporting context
        vix  = fetch_vix()
        news = fetch_news_sentiment(f"Nifty NSE India {index}")
        acc  = get_recent_accuracy(index)

        # ── VIX / Premium decay warning ──────────────────────────────────
        # This is why market goes up but your option premium decays:
        # When VIX falls, IV crush kills option premium even on correct direction
        try:
            vix_df = yf.download(VIX_TICKER, period="5d", interval="1d",
                                  progress=False, auto_adjust=True)
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            vix_df = vix_df.dropna()
            vix_change = float(vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[-2]) if len(vix_df) >= 2 else 0
        except Exception:
            vix_change = 0

        if vix_change < -1.0:
            premium_warning = (f"⚠ VIX FALLING ({vix_change:+.2f}) — IV CRUSH RISK. "
                               f"Even if direction is correct, option premium may decay. "
                               f"Consider selling options instead of buying, or use futures.")
        elif vix > 25:
            premium_warning = (f"VIX HIGH ({vix:.1f}) — options are expensive. "
                               f"Good time to BUY options (high premium = big moves expected). "
                               f"Use 1 lot only.")
        elif vix < 13:
            premium_warning = (f"VIX LOW ({vix:.1f}) — options are cheap. "
                               f"Safe to buy options. Premium decay risk is low.")
        else:
            premium_warning = f"VIX normal ({vix:.1f}). Standard option buying conditions."

        # ── Heikin-Ashi rule-based signal ────────────────────────────────
        ha_signal = {"signal": "WAIT", "strength": 0, "reasons": []}
        try:
            df_15m = fetch_ohlcv(TICKERS[index], period="5d", interval="15m")
            ha_signal = compute_ha_signal(df_15m)
        except Exception as e:
            logger.warning(f"HA signal failed: {e}")

        return {
            "index":      index.upper(),
            "timestamp":  datetime.now().isoformat(),
            "alert":      alert,
            "timeframes": analyses,
            "ha_signal":  ha_signal,
            "vix":        round(vix, 2),
            "vix_change": round(vix_change, 2),
            "premium_warning": premium_warning,
            "news":       news,
            "live_accuracy": acc,
        }
    except Exception as e:
        logger.error(f"Alert error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/signal/{index}")
def get_signal(index: str, retrain: Optional[bool] = False):
    if index not in TICKERS:
        raise HTTPException(404)
    try:
        ticker = TICKERS[index]
        df     = fetch_ohlcv(ticker, period="30d", interval="15m")
        df     = compute_features(df)
        latest = df[FEATURE_COLS].iloc[-1]
        if latest.isna().any():
            raise ValueError("NaN in features")

        global_sigs  = fetch_global_signals()
        vix          = fetch_vix()
        news         = fetch_news_sentiment(f"Nifty NSE India {index}")
        options_data = fetch_options_context(index)
        sr_levels    = compute_support_resistance(df)

        model_key = f"{index}_15m"
        if model_key not in _models or retrain:
            train_model(index, "15m")

        model     = _models[model_key]
        proba_arr = model.predict_proba(latest.values.reshape(1, -1))[0]
        classes   = model.classes_.tolist()
        up_idx    = classes.index(1) if 1 in classes else 1
        dn_idx    = classes.index(0) if 0 in classes else 0
        up_prob   = float(proba_arr[up_idx])
        dn_prob   = float(proba_arr[dn_idx])

        curr_price = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        atr_pts    = float(df["atr_pct"].iloc[-1]) / 100 * curr_price

        update_actuals(index, curr_price)

        fv = latest
        conditions_up   = []
        conditions_down = []

        if up_prob > 0.52:
            conditions_up.append(f"ML model leans UP ({up_prob*100:.0f}%)")
        elif dn_prob > 0.52:
            conditions_down.append(f"ML model leans DOWN ({dn_prob*100:.0f}%)")

        if float(fv["macd_hist"]) > 0:
            conditions_up.append("MACD histogram positive (bullish momentum)")
        else:
            conditions_down.append("MACD histogram negative (bearish momentum)")

        rsi = float(fv["rsi_14"])
        if 45 < rsi < 65:
            conditions_up.append(f"RSI {rsi:.0f} in bullish zone (45-65)")
        elif rsi < 40:
            conditions_up.append(f"RSI {rsi:.0f} oversold — bounce likely")
        elif rsi > 70:
            conditions_down.append(f"RSI {rsi:.0f} overbought — pullback likely")
        elif 35 < rsi < 55:
            conditions_down.append(f"RSI {rsi:.0f} in bearish zone (35-55)")

        pve = float(fv["price_vs_ema20"])
        if pve > 0.1:
            conditions_up.append(f"Price {pve:+.2f}% above EMA20 (uptrend)")
        elif pve < -0.1:
            conditions_down.append(f"Price {pve:+.2f}% below EMA20 (downtrend)")

        slope = float(fv["ema_slope"])
        if slope > 0:
            conditions_up.append(f"EMA20 sloping UP ({slope:+.3f}%)")
        else:
            conditions_down.append(f"EMA20 sloping DOWN ({slope:+.3f}%)")

        adx          = float(fv["adx_14"])
        trend_exists = adx > 20

        ret_s = float(fv["ret_short"])
        ret_m = float(fv["ret_medium"])
        if ret_s > 0 and ret_m > 0:
            conditions_up.append(f"Positive momentum: +{ret_s:.2f}% (short), +{ret_m:.2f}% (medium)")
        elif ret_s < 0 and ret_m < 0:
            conditions_down.append(f"Negative momentum: {ret_s:.2f}% (short), {ret_m:.2f}% (medium)")

        n_up   = len(conditions_up)
        n_down = len(conditions_down)

        if n_up >= 4 and trend_exists and n_up > n_down:
            signal     = "UP"
            confidence = round(min(95, 50 + n_up * 8 + (adx - 20) * 0.5), 1)
            setup_desc = f"{n_up}/7 bullish conditions met. ADX={adx:.0f}."
        elif n_down >= 4 and trend_exists and n_down > n_up:
            signal     = "DOWN"
            confidence = round(min(95, 50 + n_down * 8 + (adx - 20) * 0.5), 1)
            setup_desc = f"{n_down}/7 bearish conditions met. ADX={adx:.0f}."
        else:
            signal     = "WAIT"
            confidence = round(max(up_prob, dn_prob) * 100, 1)
            setup_desc = (f"Only {max(n_up,n_down)}/7 conditions met or ADX={adx:.0f}<20. "
                          f"No high-probability setup.")

        log_prediction(index, str(df.index[-1]), signal, confidence, curr_price)

        if signal in ("UP", "DOWN"):
            option_instr = generate_option_instructions(
                signal, confidence, curr_price, atr_pts, sr_levels, vix, index)
        else:
            option_instr = {
                "action":    "NO TRADE",
                "entry":     setup_desc,
                "target":    "–", "stop_loss": "–",
                "lot_sizing":"Wait for next candle",
                "hold_time": "–", "confidence": f"{confidence:.0f}%",
            }

        if signal == "UP":
            candle_desc = f"Setup: {n_up}/7 bullish. Expected: +{atr_pts:.0f} pts over 1-3 hours."
        elif signal == "DOWN":
            candle_desc = f"Setup: {n_down}/7 bearish. Expected: -{atr_pts:.0f} pts over 1-3 hours."
        else:
            candle_desc = setup_desc

        recent_acc = get_recent_accuracy(index)

        return {
            "index":          index.upper(),
            "timestamp":      str(df.index[-1]),
            "current_price":  round(curr_price, 2),
            "prev_close":     round(prev_close, 2),
            "signal":         signal,
            "confidence_pct": confidence,
            "candle_direction": signal if signal != "WAIT" else "SIDEWAYS",
            "candle_description": candle_desc,
            "up_probability":   round(up_prob * 100, 1),
            "down_probability": round(dn_prob * 100, 1),
            "conditions": {
                "bullish": conditions_up,
                "bearish": conditions_down,
                "adx":     round(adx, 1),
                "trend_exists": trend_exists,
            },
            "option_instructions": option_instr,
            "levels": sr_levels,
            "feature_values": {
                "rsi_14":         round(float(fv["rsi_14"]), 1),
                "macd_hist":      round(float(fv["macd_hist"]), 4),
                "price_vs_ema20": round(float(fv["price_vs_ema20"]), 3),
                "ema_slope":      round(float(fv["ema_slope"]), 4),
                "adx_14":         round(float(fv["adx_14"]), 1),
                "bb_position":    round(float(fv["bb_position"]), 3),
                "ret_short":      round(float(fv["ret_short"]), 3),
                "ret_medium":     round(float(fv["ret_medium"]), 3),
                "vol_ratio":      round(float(fv["vol_ratio"]), 2),
                "atr_pct":        round(float(fv["atr_pct"]), 3),
            },
            "context": {
                "india_vix":         round(vix, 2),
                "news_sentiment":    news.get("sentiment_score", 0),
                "top_headline":      news.get("top_headline", "N/A"),
                "headlines":         news.get("headlines", []),
                "pcr_ratio":         options_data["pcr"],
                "options_status":    options_data["status"],
                "nq_futures_change": round(global_sigs.get("nq_change_pct", 0), 2),
                "usdinr_change":     round(global_sigs.get("usdinr_change", 0), 4),
            },
            "live_accuracy": recent_acc,
            "expiry":        NSE_EXPIRY,
        }

    except Exception as e:
        logger.error(f"Signal error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/swing/{index}")
def swing_endpoint(index: str):
    if index not in TICKERS:
        raise HTTPException(404)
    try:
        ticker = TICKERS[index]
        if index not in _daily_models:
            _train_daily_model(index)
        dm    = _daily_models[index]
        df    = _fetch_daily_df(ticker)
        df    = _compute_daily_features(df)
        cols  = dm["cols"]
        lat   = df[cols].iloc[-1]
        proba = dm["model"].predict_proba(lat.values.reshape(1, -1))[0]
        cls   = dm["model"].classes_.tolist()
        up_p  = float(proba[cls.index(1)]) if 1 in cls else 0.5
        dn_p  = 1 - up_p

        global_sigs = fetch_global_signals()
        vix         = fetch_vix()
        news        = fetch_news_sentiment(f"Nifty NSE India {index}")
        nq_chg      = global_sigs.get("nq_change_pct", 0)
        usdinr_chg  = global_sigs.get("usdinr_change", 0)
        gold_chg    = global_sigs.get("gold_change_pct", 0)

        today_bias = "BULLISH" if up_p > 0.55 else "BEARISH" if dn_p > 0.55 else "NEUTRAL"

        gap_score = (
            np.tanh(nq_chg / 0.8) * 0.65
            - np.tanh(usdinr_chg * 8) * 0.20
            - np.tanh(gold_chg / 1.5) * 0.10
            + (up_p - dn_p) * 0.05
        )
        if gap_score > 0.20:
            next_open = "GAP UP"
            gap_pts   = round(abs(nq_chg) * 0.55 * float(df["Close"].iloc[-1]) / 100, 0)
        elif gap_score < -0.20:
            next_open = "GAP DOWN"
            gap_pts   = round(abs(nq_chg) * 0.55 * float(df["Close"].iloc[-1]) / 100, 0)
        else:
            next_open = "FLAT OPEN"
            gap_pts   = 0

        overnight = compute_overnight_risk(vix, nq_chg, usdinr_chg, 0)

        curr   = float(df["Close"].iloc[-1])
        ema20  = float(ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator().iloc[-1])
        ema50  = float(ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator().iloc[-1])
        ema200 = float(ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator().iloc[-1])
        high52 = round(float(df["High"].tail(252).max()), 2)
        low52  = round(float(df["Low"].tail(252).min()), 2)
        wk     = df.tail(5)
        wpivot = round((float(wk["High"].max())+float(wk["Low"].min())+float(wk["Close"].iloc[-1]))/3, 2)
        wr1    = round(2*wpivot - float(wk["Low"].min()), 2)
        ws1    = round(2*wpivot - float(wk["High"].max()), 2)
        atr_d  = float(ta.volatility.AverageTrueRange(df["High"],df["Low"],df["Close"]).average_true_range().iloc[-1])
        step   = 100 if index == "banknifty" else 50
        atm    = round(curr / step) * step
        itm_ce = atm - step
        itm_pe = atm + step

        above_ema20  = curr > ema20
        above_ema50  = curr > ema50
        above_ema200 = curr > ema200
        if above_ema20 and above_ema50:
            daily_trend = "UPTREND — price above EMA20 & EMA50"
        elif not above_ema20 and not above_ema50:
            daily_trend = "DOWNTREND — price below EMA20 & EMA50"
        else:
            daily_trend = "MIXED — between key EMAs, choppy"

        can_trade = overnight["score"] < 55

        if today_bias == "BULLISH" and above_ema20 and can_trade:
            decision      = "BUY CALL"
            decision_color= "green"
            option_action = f"Buy {itm_ce} CE (1 ITM)"
            entry_zone    = f"Entry: {curr - atr_d*0.3:,.0f} – {curr:,.0f}"
            target_zone   = f"Target: {wr1:,.0f} (Wk R1) | Book 50% at {curr + atr_d:,.0f}"
            sl_zone       = f"SL: Below {curr - atr_d*0.8:,.0f} | Exit by 2:30 PM"
            reason        = (f"Daily model {up_p:.0%} bullish. Price above EMA20. "
                             f"NQ {nq_chg:+.2f}%. Overnight risk {overnight['score']}/100.")
        elif today_bias == "BEARISH" and not above_ema20 and can_trade:
            decision      = "BUY PUT"
            decision_color= "red"
            option_action = f"Buy {itm_pe} PE (1 ITM)"
            entry_zone    = f"Entry: {curr:,.0f} – {curr + atr_d*0.3:,.0f}"
            target_zone   = f"Target: {ws1:,.0f} (Wk S1) | Book 50% at {curr - atr_d:,.0f}"
            sl_zone       = f"SL: Above {curr + atr_d*0.8:,.0f} | Exit by 2:30 PM"
            reason        = (f"Daily model {dn_p:.0%} bearish. Price below EMA20. "
                             f"NQ {nq_chg:+.2f}%. Overnight risk {overnight['score']}/100.")
        elif overnight["score"] >= 55:
            decision      = "NO TRADE TODAY"
            decision_color= "amber"
            option_action = "Skip — overnight risk too high"
            entry_zone = target_zone = sl_zone = "–"
            reason        = (f"Overnight risk {overnight['score']}/100 is too high. "
                             f"VIX at {vix:.1f}. Wait for risk to settle.")
        elif today_bias == "BULLISH" and not above_ema20:
            decision      = "NO TRADE — TREND CONFLICT"
            decision_color= "amber"
            option_action = "Model says UP but daily trend is DOWN — skip"
            entry_zone = target_zone = sl_zone = "–"
            reason        = (f"Never fight the daily trend. Price is below EMA20 ({ema20:,.0f}). "
                             f"Wait for price to reclaim EMA20 before buying calls.")
        elif today_bias == "BEARISH" and above_ema20:
            decision      = "NO TRADE — TREND CONFLICT"
            decision_color= "amber"
            option_action = "Model says DOWN but daily trend is UP — skip"
            entry_zone = target_zone = sl_zone = "–"
            reason        = (f"Never fight the daily trend. Price is above EMA20 ({ema20:,.0f}). "
                             f"Wait for price to break below EMA20 before buying puts.")
        else:
            decision      = "WAIT"
            decision_color= "amber"
            option_action = "No clear signal — sit on hands"
            entry_zone = target_zone = sl_zone = "–"
            reason        = "Daily model is neutral. No high-confidence setup today."

        return {
            "index":            index.upper(),
            "date":             str(df.index[-1])[:10],
            "current_price":    round(curr, 2),
            "today_bias":       today_bias,
            "next_open_bias":   next_open,
            "gap_pts_estimate": gap_pts,
            "ml_probabilities": {"bullish": round(up_p, 3), "bearish": round(dn_p, 3)},
            "overnight_risk":   overnight,
            "daily_trend":      daily_trend,
            "above_ema20":      above_ema20,
            "above_ema50":      above_ema50,
            "decision": {
                "action":    decision,
                "color":     decision_color,
                "option":    option_action,
                "entry":     entry_zone,
                "target":    target_zone,
                "stop_loss": sl_zone,
                "reason":    reason,
                "window":    "Trade only 10:30 AM – 2:00 PM. Exit by 2:30 PM.",
            },
            "daily_levels": {
                "ema20":        round(ema20, 2),
                "ema50":        round(ema50, 2),
                "ema200":       round(ema200, 2),
                "high_52w":     high52,
                "low_52w":      low52,
                "weekly_pivot": wpivot,
                "weekly_r1":    wr1,
                "weekly_s1":    ws1,
            },
            "global_context": {
                "nq_change_pct":   round(nq_chg, 2),
                "usdinr_change":   round(usdinr_chg, 4),
                "gold_change_pct": round(gold_chg, 2),
                "india_vix":       round(vix, 2),
                "top_headline":    news.get("top_headline", "N/A"),
                "headlines":       news.get("headlines", []),
                "note":            "News shown for context only — not used in prediction",
            },
        }
    except Exception as e:
        logger.error(f"Swing error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── SECTION 8: PRICE REGRESSION + FORECAST ───────────────────────────────────

REG_FEATURE_COLS = ["rsi_14", "macd_hist", "bb_position", "volume_ratio",
                    "atr_pct", "ha_trend", "macd_accel", "ret1", "ret3"]


def _compute_reg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for regression model (price point prediction)."""
    df    = df.copy()
    close = df["Close"]; high = df["High"]; low = df["Low"]
    vol   = df["Volume"].replace(0, 1); op = df["Open"]
    atr   = ta.volatility.AverageTrueRange(high, low, close).average_true_range()

    df["rsi_14"]      = ta.momentum.RSIIndicator(close, 14).rsi()
    df["macd_hist"]   = ta.trend.MACD(close).macd_diff()
    bb                = ta.volatility.BollingerBands(close)
    df["bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
    df["volume_ratio"]= vol / (vol.rolling(20).mean() + 1e-9)
    df["atr_pct"]     = atr / close * 100
    ha_c = (op + high + low + close) / 4
    ha_o = ha_c.copy()
    for i in range(1, len(df)):
        ha_o.iloc[i] = (ha_o.iloc[i-1] + ha_c.iloc[i-1]) / 2
    df["ha_trend"]  = ((ha_c > ha_o).astype(float).rolling(3).sum() - 1.5) / 1.5
    df["macd_accel"]= df["macd_hist"].diff(1)
    df["ret1"]      = close.pct_change(1)
    df["ret3"]      = close.pct_change(3)
    df.dropna(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def train_regression_model(index_name: str) -> dict:
    """
    Train two regression models:
    - model_close: predicts next bar close - current close (in points)
    - model_range: predicts next bar high-low range (in points)
    Also computes MAE on held-out last 2 months.
    """
    ticker = TICKERS[index_name]
    logger.info(f"Training regression model: {index_name}...")

    try:
        df = fetch_ohlcv(ticker, period="60d", interval="15m")
    except Exception:
        df = fetch_ohlcv(ticker, period="30d", interval="15m")

    df = _compute_reg_features(df)
    close = df["Close"]; high = df["High"]; low = df["Low"]

    # Targets: next bar point change and range
    df["target_close"] = close.shift(-1) - close          # pts up/down
    df["target_high"]  = high.shift(-1) - close           # pts to next high
    df["target_low"]   = low.shift(-1) - close            # pts to next low
    df.dropna(inplace=True)

    X = df[REG_FEATURE_COLS]
    mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1)
    X = X[mask]
    y_close = df["target_close"][mask]
    y_high  = df["target_high"][mask]
    y_low   = df["target_low"][mask]

    # Train/test: last 500 bars as test
    cutoff = max(100, len(X) - 500)
    X_tr, X_te = X.iloc[:cutoff], X.iloc[cutoff:]
    yc_tr, yc_te = y_close.iloc[:cutoff], y_close.iloc[cutoff:]
    yh_tr, yh_te = y_high.iloc[:cutoff],  y_high.iloc[cutoff:]
    yl_tr, yl_te = y_low.iloc[:cutoff],   y_low.iloc[cutoff:]

    def _reg():
        return Pipeline([("sc", StandardScaler()),
                         ("reg", GradientBoostingRegressor(
                             n_estimators=200, learning_rate=0.05, max_depth=3,
                             min_samples_leaf=20, subsample=0.8, random_state=42))])

    m_close = _reg(); m_close.fit(X_tr, yc_tr)
    m_high  = _reg(); m_high.fit(X_tr, yh_tr)
    m_low   = _reg(); m_low.fit(X_tr, yl_tr)

    mae_close = mean_absolute_error(yc_te, m_close.predict(X_te))
    mae_high  = mean_absolute_error(yh_te, m_high.predict(X_te))
    mae_low   = mean_absolute_error(yl_te, m_low.predict(X_te))

    # Direction accuracy
    pred_c = m_close.predict(X_te)
    dir_acc = float(((pred_c > 0) == (yc_te.values > 0)).mean())

    # Rolling prediction error for the last 50 bars (for error band on chart)
    pred_all = m_close.predict(X)
    actual_all = y_close.values
    errors = np.abs(actual_all - pred_all)
    rolling_mae = pd.Series(errors).rolling(20).mean().fillna(mae_close).values

    _reg_models[index_name] = {
        "close": m_close, "high": m_high, "low": m_low,
        "mae_close": mae_close, "mae_high": mae_high, "mae_low": mae_low,
        "dir_acc": dir_acc,
        "df": df, "X": X, "y_close": y_close,
        "pred_all": pred_all, "rolling_mae": rolling_mae,
    }
    logger.info(f"{index_name} regression: MAE={mae_close:.1f}pts DirAcc={dir_acc:.1%}")
    return {"mae_close": round(mae_close, 1), "mae_high": round(mae_high, 1),
            "mae_low": round(mae_low, 1), "dir_acc": round(dir_acc, 4)}


@app.get("/forecast/{index}")
def forecast_endpoint(index: str):
    """
    Returns:
    - Last 50 bars of actual close prices
    - Last 50 bars of model's predicted close (what it would have predicted)
    - Rolling MAE band (prediction error over time)
    - Next 4 bars forecast (ghost candles)
    - Support floor: price level model is confident won't be breached
    - Next day forecast (daily model)
    """
    if index not in TICKERS:
        raise HTTPException(404)
    try:
        if index not in _reg_models:
            train_regression_model(index)

        rm   = _reg_models[index]
        df   = rm["df"]
        X    = rm["X"]
        pred_all    = rm["pred_all"]
        rolling_mae = rm["rolling_mae"]
        mae_close   = rm["mae_close"]

        # Last 80 bars for chart
        n = min(80, len(df))
        df_tail  = df.tail(n)
        pred_tail = pred_all[-n:]
        mae_tail  = rolling_mae[-n:]

        actual_prices = df_tail["Close"].values.tolist()
        pred_prices   = (df_tail["Close"].values + pred_tail).tolist()  # predicted next close
        mae_upper     = (df_tail["Close"].values + mae_tail).tolist()
        mae_lower     = (df_tail["Close"].values - mae_tail).tolist()
        timestamps    = [str(t) for t in df_tail.index]

        # Next 4 bars forecast
        curr_price = float(df["Close"].iloc[-1])
        curr_high  = float(df["High"].iloc[-1])
        curr_low   = float(df["Low"].iloc[-1])
        fv = X.iloc[-1].values.reshape(1, -1)

        next_close_delta = float(rm["close"].predict(fv)[0])
        next_high_delta  = float(rm["high"].predict(fv)[0])
        next_low_delta   = float(rm["low"].predict(fv)[0])

        next_close = round(curr_price + next_close_delta, 2)
        next_high  = round(curr_price + max(next_high_delta, next_close_delta, 0), 2)
        next_low   = round(curr_price + min(next_low_delta, next_close_delta, 0), 2)

        # Support floor: current price minus 1.5× MAE (model is 85% confident price stays above)
        support_floor = round(curr_price - mae_close * 1.5, 0)
        resist_ceil   = round(curr_price + mae_close * 1.5, 0)

        # Signal strength
        if abs(next_close_delta) >= 20:
            if next_close_delta > 0:
                signal = "UP"; signal_pts = round(next_close_delta, 0)
            else:
                signal = "DOWN"; signal_pts = round(abs(next_close_delta), 0)
        else:
            signal = "FLAT"; signal_pts = round(abs(next_close_delta), 0)

        # Next day forecast using daily model
        next_day = None
        try:
            if index in _daily_models:
                dm = _daily_models[index]
                df_d = _fetch_daily_df(TICKERS[index])
                df_d = _compute_daily_features(df_d)
                cols = dm["cols"]
                lat  = df_d[cols].iloc[-1]
                proba = dm["model"].predict_proba(lat.values.reshape(1, -1))[0]
                cls   = dm["model"].classes_.tolist()
                up_p  = float(proba[cls.index(1)]) if 1 in cls else 0.5
                # Daily ATR for range estimate
                d_atr = float(ta.volatility.AverageTrueRange(
                    df_d["High"], df_d["Low"], df_d["Close"]).average_true_range().iloc[-1])
                next_day = {
                    "up_probability": round(up_p * 100, 1),
                    "down_probability": round((1-up_p) * 100, 1),
                    "expected_range_pts": round(d_atr, 0),
                    "bias": "BULLISH" if up_p > 0.55 else "BEARISH" if up_p < 0.45 else "NEUTRAL",
                    "target_up":   round(curr_price + d_atr * 0.8, 0),
                    "target_down": round(curr_price - d_atr * 0.8, 0),
                }
        except Exception as e:
            logger.warning(f"Next day forecast failed: {e}")

        return {
            "index":        index.upper(),
            "current_price":round(curr_price, 2),
            "signal":       signal,
            "signal_pts":   signal_pts,
            "mae_pts":      round(mae_close, 1),
            "dir_accuracy": round(rm["dir_acc"] * 100, 1),
            "next_bar": {
                "open":  curr_price,
                "high":  next_high,
                "low":   next_low,
                "close": next_close,
                "delta": round(next_close_delta, 1),
            },
            "support_floor":  support_floor,
            "resist_ceiling": resist_ceil,
            "chart_data": {
                "timestamps":   timestamps,
                "actual":       [round(p, 2) for p in actual_prices],
                "predicted":    [round(p, 2) for p in pred_prices],
                "mae_upper":    [round(p, 2) for p in mae_upper],
                "mae_lower":    [round(p, 2) for p in mae_lower],
            },
            "next_day": next_day,
        }
    except Exception as e:
        logger.error(f"Forecast error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


# ── SECTION 9: STARTUP ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Train models on startup.
    Only trains 15m intraday (best for 20-40pt moves), daily, and regression.
    Skips 5m/30m/1h — they added noise without improving accuracy.
    """
    for idx in TICKERS:
        try:
            logger.info(f"Training 15m model: {idx}...")
            train_model(idx, "15m")
        except Exception as e:
            logger.warning(f"15m train failed {idx}: {e}")
        try:
            logger.info(f"Training daily model: {idx}...")
            _train_daily_model(idx)
        except Exception as e:
            logger.warning(f"Daily train failed {idx}: {e}")
        try:
            logger.info(f"Training regression model: {idx}...")
            train_regression_model(idx)
        except Exception as e:
            logger.warning(f"Regression train failed {idx}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
