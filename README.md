# SignalX — Nifty Intraday Prediction Engine

A real-time algorithmic trading signal system for NSE indices (Nifty 50, Bank Nifty, Sensex) built with Python, FastAPI, and machine learning. Predicts short-term price movement using multi-timeframe technical analysis and a regression-based price forecasting model.

---

## What it does

- Fetches live 5m / 15m / 30m / 1H OHLCV data from TradingView (via tvDatafeed) with yfinance fallback
- Trains a **GradientBoosting regression model** on historical 15m bars to predict next-bar price movement in points
- Runs a **multi-timeframe alert system** — fires a signal only when 3 out of 4 timeframes agree on direction
- Displays a **live candlestick chart** with predicted next bar (ghost candle), prediction error band (±MAE), and 3-line forecast overlay
- Provides **support floor and resistance ceiling** levels with 85% statistical confidence
- Computes **next-day bias** using a separate daily model trained on 5 years of data
- Shows **VIX / IV crush warning** — tells you when option premium will decay even if direction is correct
- Includes **Heikin-Ashi + MACD rule-based signal** as a secondary confirmation layer

---

## Model Performance

| Metric | Value |
|---|---|
| Training data | ~5,000 bars (15m, ~10 months via tvDatafeed) |
| Price prediction MAE | **±36.6 pts** (next 15-min bar) |
| Predictions within ±40 pts | **72.4%** of test bars |
| Direction accuracy (binary) | **50–53%** (honest — market is efficient) |
| Walk-forward win rate at 55% confidence | **55%** with balanced UP/DOWN signals |
| Daily model (next-day direction) | **52.7% CV accuracy** on 10 years of data |
| Multi-TF alert Sharpe ratio | **2.56** (positive expected value per signal) |

> **Honest note:** Nifty 50 is one of the most liquid indices in the world. No technical model reliably predicts direction above 55% — this is consistent with academic literature on market efficiency. The edge in this system comes from **risk management** (ATR-based SL/TP), **signal filtering** (only trade when 3+ timeframes agree), and **IV awareness** (VIX monitoring to avoid premium decay).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Models | scikit-learn (GradientBoosting, RandomForest, CalibratedClassifierCV) |
| Data | tvDatafeed (TradingView), yfinance |
| Technical Indicators | ta (Technical Analysis library) |
| Frontend | Vanilla JS + Lightweight Charts v4 |
| Environment | Python 3.10+, dotenv |

---

## Project Structure

```
NiftyPrediction/
├── market_signal_engine.py   # FastAPI backend — all models, endpoints, data fetching
├── chart_dashboard.html      # Live chart frontend — candlestick + forecast overlay
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template (copy to .env)
└── README.md
```

---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/Meghansh2005/NiftyPrediction.git
cd NiftyPrediction
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env and add your NewsAPI key (free at newsapi.org)
```

**3. Run the backend**
```bash
python -m uvicorn market_signal_engine:app --port 8001
```
Wait ~5–8 minutes for all models to train on startup (12 intraday + 3 daily + 3 regression models).

**4. Open the dashboard**

Open `chart_dashboard.html` in your browser. The dashboard connects to `localhost:8001` automatically.

---

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Server status and loaded models |
| `GET /forecast/{index}` | Price regression forecast — next bar OHLC, support/resistance, 3-line chart data |
| `GET /alert/{index}` | Multi-timeframe signal — fires when 3/4 TFs agree |
| `GET /signal/{index}` | Single-timeframe signal with option chain instructions |
| `GET /swing/{index}` | Daily swing analysis — tomorrow open bias, overnight risk score |
| `GET /candles/{index}?tf=15m&bars=120` | OHLCV candles with predicted ghost bar |
| `GET /price/{index}` | Lightweight live price tick (safe to call every 20s) |
| `GET /train/{index}` | Retrain all models for an index |
| `GET /backtest/{index}` | Walk-forward backtest results and feature importance |

Supported index values: `nifty`, `banknifty`, `sensex`

---

## Environment Variables

```env
NEWS_API_KEY=your_newsapi_org_key    # https://newsapi.org (free tier, 100 req/day)
NSE_EXPIRY_DATE=2026-04-24           # Update to next monthly expiry
```

---

## Disclaimer

This is a **decision-support tool only**. It is not financial advice. All signals are probabilistic — past accuracy does not guarantee future results. Options trading carries significant risk of total capital loss. Always use stop-losses. Never risk more than 1–2% of capital per trade.

---

## License

MIT License — see [LICENSE](LICENSE)
