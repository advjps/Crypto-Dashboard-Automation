# run_automation.py — 9th Amendment (Safe indicators + rich logging)
# Fully self-contained & indentation-safe

import pandas as pd
import requests
import json
import math
import os
import time
from datetime import datetime
import pytz

# ============== PROXY CONFIG (set if needed) ==============
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

# ============== GENERAL CONFIG ==============
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"
TOP_LIMIT = 70
BINANCE_FAPI = "https://fapi.binance.com"

# Profit evaluation basis (ROI on margin)
LEVERAGE_FOR_PROFIT_EVAL = 7.0
MIN_PROFIT_MARGIN = 2.0           # min % on margin to pass (floor)

# TP/SL (price-based) — per latest spec
TP_PCT_PRICE = 0.0072             # ~0.72% of price  (~5% on 7x margin)
SL_PCT_PRICE = 0.03               # ~3.0% of price (~21% on 7x margin)

# Regime (market/BTC) thresholds
REGIME_BULLISH_BONUS = 10         # add to favored side score
REGIME_STRICTER_STRONG = 5        # raise strong threshold for unfavored side

# ============== SAFE NUMERIC HELPERS ==============
def _isnum(x):
    try:
        return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return False

def _nz(x, default=None):
    return x if _isnum(x) else default

def get_last_valid_value(values):
    if not isinstance(values, list):
        return None
    for v in reversed(values):
        if _isnum(v):
            return v
    return None

# ============== INDICATOR FUNCTIONS (NaN-safe) ==============
def calc_ema(values, period):
    if not isinstance(values, list) or len(values) < max(2, period):
        return [None] * len(values)
    s = pd.Series(values, dtype=float)
    out = s.ewm(span=period, adjust=False).mean().tolist()
    return [v if _isnum(v) else None for v in out]

def calc_rsi(values, period=14):
    if not isinstance(values, list) or len(values) < period + 2:
        return [None] * len(values)
    s = pd.Series(values, dtype=float)
    d = s.diff()
    gain = d.where(d > 0, 0.0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.where(d < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
    # avoid division explosions
    rs = gain / loss.replace(0, pd.NA)
    denom = (1 + rs).replace([math.inf, -math.inf], pd.NA)
    rsi = 100 - (100 / denom)
    return [v if _isnum(v) else None for v in rsi.tolist()]

def calc_macd(values, fast=12, slow=26, signal=9):
    if not isinstance(values, list) or len(values) < slow + signal + 2:
        return {"macd": None, "signal": None, "histogram": None}
    s = pd.Series(values, dtype=float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return {
        "macd": _nz(macd.iloc[-1]),
        "signal": _nz(sig.iloc[-1]),
        "histogram": _nz(hist.iloc[-1]),
    }

def calc_bollinger(values, period=20, mult=2):
    if not isinstance(values, list) or len(values) < period:
        return {"upper": None, "middle": None, "lower": None}
    s = pd.Series(values, dtype=float)
    ma = s.rolling(window=period).mean().iloc[-1]
    sd = s.rolling(window=period).std(ddof=0).iloc[-1]
    if not _isnum(ma) or not _isnum(sd):
        return {"upper": None, "middle": None, "lower": None}
    return {"upper": ma + mult * sd, "middle": ma, "lower": ma - mult * sd}

def calc_cci(highs, lows, closes, period=20):
    if min(len(highs), len(lows), len(closes)) < period:
        return None
    tp = pd.Series([(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)], dtype=float)
    sma = tp.rolling(window=period).mean().iloc[-1]
    md = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False).iloc[-1]
    if not _isnum(sma) or not _isnum(md) or md == 0:
        return None
    return _nz((tp.iloc[-1] - sma) / (0.015 * md))

def calc_williams_r(highs, lows, closes, period=14):
    if min(len(highs), len(lows), len(closes)) < period:
        return None
    hi = max(highs[-period:])
    lo = min(lows[-period:])
    if not _isnum(hi) or not _isnum(lo) or hi == lo:
        return None
    close = closes[-1]
    if not _isnum(close):
        return None
    return _nz(-100.0 * ((hi - close) / (hi - lo)))

def calc_cmf(highs, lows, closes, volumes, period=20):
    if min(len(highs), len(lows), len(closes), len(volumes)) < period:
        return None
    H = pd.Series(highs, dtype=float)
    L = pd.Series(lows, dtype=float)
    C = pd.Series(closes, dtype=float)
    V = pd.Series(volumes, dtype=float)
    rng = (H - L).replace(0, 1e-12)
    mfm = ((C - L) - (H - C)) / rng
    mfv = mfm * V
    num = mfv.rolling(window=period).sum().iloc[-1]
    den = V.rolling(window=period).sum().iloc[-1]
    if not _isnum(num) or not _isnum(den) or den == 0:
        return None
    return _nz(num / den)

# Light/safe placeholders for advanced indicators (you can later replace with true implementations)
def calc_stc(values):
    # Schaff Trend Cycle: return None for now (harmless)
    return None

def calc_tsi(values):
    # True Strength Index: return None for now
    return None

def calc_cvd(closes, volumes):
    # Simple direction label: rising/falling based on most recent vs avg
    if not isinstance(volumes, list) or len(volumes) < 2:
        return None
    look = min(20, len(volumes))
    avg = sum(volumes[-look:]) / max(1, look)
    return "rising" if volumes[-1] > avg else "falling"

def calc_hma(values, period=9):
    if not isinstance(values, list) or len(values) < 2:
        return None
    return "up" if values[-1] > values[-2] else "down"

def calc_alma(values):
    if not isinstance(values, list) or len(values) < 2:
        return None
    return "up" if values[-1] > values[-2] else "down"

def calc_keltner(values, period=20):
    if not isinstance(values, list) or len(values) < period:
        return {"upper": None, "middle": None, "lower": None}
    s = pd.Series(values[-period:], dtype=float)
    mid = s.mean()
    # simple envelope as ATR-like placeholder (safe & bounded)
    half_span = (s.max() - s.min()) / 2.0
    return {"upper": _nz(mid + half_span), "middle": _nz(mid), "lower": _nz(mid - half_span)}

def calc_market_trend(closes):
    if not isinstance(closes, list) or len(closes) < 50:
        return 0
    ema20 = get_last_valid_value(calc_ema(closes, 20))
    ema50 = get_last_valid_value(calc_ema(closes, 50))
    if not _isnum(ema20) or not _isnum(ema50):
        return 0
    if ema20 > ema50:
        return 10
    if ema20 < ema50:
        return -10
    return 0

# ============== DATA FETCHING ==============
def fetch_top_volume_coins(limit=TOP_LIMIT):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
        r = requests.get(url, proxies=proxies, timeout=30)
        r.raise_for_status()
        data = r.json()
        usdt = [d for d in data if 'symbol' in d and d['symbol'].endswith('USDT')]
        top = sorted(usdt, key=lambda x: float(x.get("quoteVolume", 0.0)), reverse=True)[:limit]
        return [t['symbol'] for t in top]
    except Exception as e:
        print(f"[ERR] fetch_top_volume_coins: {e}")
        return []

def fetch_binance_data(symbol, interval="5m", limit=100):
    url = f"{BINANCE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, proxies=proxies, timeout=30)
        r.raise_for_status()
        data = r.json()
        return [{
            "open": float(d[1]),
            "high": float(d[2]),
            "low": float(d[3]),
            "close": float(d[4]),
            "volume": float(d[5])
        } for d in data]
    except Exception as e:
        print(f"[ERR] fetch_binance_data {symbol}:{interval}: {e}")
        return []

# ============== ANALYSIS / SCORING ==============
def analyze_data(symbol, data5m, market_trend):
    """
    9th Amendment scoring (soft-scoring, minimal vetoes), with full JSON logging.
    - Separate BUY / SELL scoring trees
    - Regime-aware bias
    - Per-indicator score breakdown
    - Skips Neutral (no output)
    """
    if not data5m or len(data5m) < 50:
        return None

    # Current price & series
    price = data5m[-1]["close"]
    highs = [d["high"] for d in data5m]
    lows  = [d["low"] for d in data5m]
    closes = [d["close"] for d in data5m]
    vols   = [d["volume"] for d in data5m]

    # -------- Indicators (safe-returning) --------
    rsi = get_last_valid_value(calc_rsi(closes, 14))
    willr = calc_williams_r(highs, lows, closes, 14)
    cci = calc_cci(highs, lows, closes, 20)
    boll = calc_bollinger(closes, 20, 2)
    macd = calc_macd(closes, 12, 26, 9)
    cmf = calc_cmf(highs, lows, closes, vols, 20)
    stc = calc_stc(closes)
    tsi = calc_tsi(closes)
    cvd = calc_cvd(closes, vols)
    hma = calc_hma(closes, 9)
    alma = calc_alma(closes)
    keltner = calc_keltner(closes, 20)

    # -------- Scoring tables --------
    buy_score = 0
    sell_score = 0
    indicator_scores = {}

    # BUY contributions
    if _isnum(rsi):
        sc = 25 if rsi <= 30 else (10 if rsi <= 40 else 0)
        buy_score += sc; indicator_scores["RSI_buy"] = {"value": rsi, "score": sc}
    if _isnum(willr):
        sc = 15 if willr <= -80 else 0
        buy_score += sc; indicator_scores["WilliamsR_buy"] = {"value": willr, "score": sc}
    if _isnum(cci):
        sc = 10 if cci >= 100 else 0
        buy_score += sc; indicator_scores["CCI_buy"] = {"value": cci, "score": sc}
    if boll.get("lower") is not None and price <= boll["lower"]:
        sc = 15; buy_score += sc; indicator_scores["Bollinger_buy"] = {"position": "below_lower", "score": sc}
    if stc is not None:
        sc = 15 if stc < 25 else 0
        buy_score += sc; indicator_scores["STC_buy"] = {"value": stc, "score": sc}
    if tsi is not None:
        sc = 10  # placeholder until true TSI wired
        buy_score += sc; indicator_scores["TSI_buy"] = {"value": tsi, "score": sc}
    if _isnum(cmf):
        sc = 10 if cmf > 0 else 0
        buy_score += sc; indicator_scores["CMF_buy"] = {"value": cmf, "score": sc}
    if cvd is not None:
        sc = 15 if cvd == "rising" else 0
        buy_score += sc; indicator_scores["CVD_buy"] = {"value": cvd, "score": sc}
    if hma is not None:
        sc = 10 if hma == "up" else 0
        buy_score += sc; indicator_scores["HMA_buy"] = {"slope": hma, "score": sc}
    if alma is not None:
        sc = 10 if alma == "up" else 0
        buy_score += sc; indicator_scores["ALMA_buy"] = {"cross": alma, "score": sc}
    if keltner.get("lower") is not None and price <= keltner["lower"]:
        sc = 10; buy_score += sc; indicator_scores["Keltner_buy"] = {"position": "lower_band_touch", "score": sc}
    if _isnum(macd.get("histogram")):
        sc = 10 if macd["histogram"] > 0 else 0
        buy_score += sc; indicator_scores["MACD_buy"] = {
            "histogram": macd["histogram"],
            "trend": "up" if macd["histogram"] > 0 else "down",
            "score": sc
        }

    # SELL contributions
    if _isnum(rsi):
        sc = 25 if rsi >= 70 else (10 if rsi >= 60 else 0)
        sell_score += sc; indicator_scores["RSI_sell"] = {"value": rsi, "score": sc}
    if _isnum(willr):
        sc = 15 if willr >= -20 else 0
        sell_score += sc; indicator_scores["WilliamsR_sell"] = {"value": willr, "score": sc}
    if _isnum(cci):
        sc = 10 if cci <= -100 else 0
        sell_score += sc; indicator_scores["CCI_sell"] = {"value": cci, "score": sc}
    if boll.get("upper") is not None and price >= boll["upper"]:
        sc = 15; sell_score += sc; indicator_scores["Bollinger_sell"] = {"position": "above_upper", "score": sc}
    if stc is not None:
        sc = 15 if stc > 75 else 0
        sell_score += sc; indicator_scores["STC_sell"] = {"value": stc, "score": sc}
    if tsi is not None:
        sc = 10  # placeholder
        sell_score += sc; indicator_scores["TSI_sell"] = {"value": tsi, "score": sc}
    if _isnum(cmf):
        sc = 10 if cmf < 0 else 0
        sell_score += sc; indicator_scores["CMF_sell"] = {"value": cmf, "score": sc}
    if cvd is not None:
        sc = 15 if cvd == "falling" else 0
        sell_score += sc; indicator_scores["CVD_sell"] = {"value": cvd, "score": sc}
    if hma is not None:
        sc = 10 if hma == "down" else 0
        sell_score += sc; indicator_scores["HMA_sell"] = {"slope": hma, "score": sc}
    if alma is not None:
        sc = 10 if alma == "down" else 0
        sell_score += sc; indicator_scores["ALMA_sell"] = {"cross": alma, "score": sc}
    if keltner.get("upper") is not None and price >= keltner["upper"]:
        sc = 10; sell_score += sc; indicator_scores["Keltner_sell"] = {"position": "upper_band_touch", "score": sc}
    if _isnum(macd.get("histogram")):
        sc = 10 if macd["histogram"] < 0 else 0
        sell_score += sc; indicator_scores["MACD_sell"] = {
            "histogram": macd["histogram"],
            "trend": "down" if macd["histogram"] < 0 else "up",
            "score": sc
        }

    # -------- Regime & thresholds --------
    regime = "Bullish" if market_trend > 0 else "Bearish" if market_trend < 0 else "Neutral"
    buy_thresh, strong_buy_thresh = 35, 55
    sell_thresh, strong_sell_thresh = 35, 55
    regime_note = "none"

    if regime == "Bullish":
        buy_score += REGIME_BULLISH_BONUS
        strong_sell_thresh += REGIME_STRICTER_STRONG
        regime_note = f"+{REGIME_BULLISH_BONUS} buy, +{REGIME_STRICTER_STRONG} strong-sell threshold"
    elif regime == "Bearish":
        sell_score += REGIME_BULLISH_BONUS
        strong_buy_thresh += REGIME_STRICTER_STRONG
        regime_note = f"+{REGIME_BULLISH_BONUS} sell, +{REGIME_STRICTER_STRONG} strong-buy threshold"

    # -------- Decide signal (skip Neutral) --------
    signal = "Neutral"; confidence = 0
    if buy_score >= sell_score and buy_score >= buy_thresh:
        signal = "Strong Buy" if buy_score >= strong_buy_thresh else "Buy"
        confidence = int(round(buy_score))
    elif sell_score > buy_score and sell_score >= sell_thresh:
        signal = "Strong Sell" if sell_score >= strong_sell_thresh else "Sell"
        confidence = int(round(sell_score))
    else:
        return None  # do not log neutral

    # -------- TP/SL (fixed percentages per latest spec) --------
    tp = price * (1 + TP_PCT_PRICE) if "Buy" in signal else price * (1 - TP_PCT_PRICE)
    sl = price * (1 - SL_PCT_PRICE) if "Buy" in signal else price * (1 + SL_PCT_PRICE)
    est_profit_margin = TP_PCT_PRICE * 100.0 * LEVERAGE_FOR_PROFIT_EVAL  # on margin

    # Minimal veto: profit floor
    profit_floor_ok = est_profit_margin >= MIN_PROFIT_MARGIN
    if not profit_floor_ok:
        return None

    # -------- Timestamps --------
    utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
    ist_now = utc_now.astimezone(pytz.timezone("Asia/Kolkata"))

    # -------- Return object with rich log --------
    return {
        "coin": symbol,
        "price": round(float(price), 6),
        "tp": round(float(tp), 6),
        "sl": round(float(sl), 6),
        "leverage": f"{int(LEVERAGE_FOR_PROFIT_EVAL)}x",
        "confidence": int(confidence),
        "signal": signal,
        "estimated_profit": f"{est_profit_margin:.2f}%",
        "regime": regime,
        "signal_time_utc": utc_now.isoformat(),
        "signal_time_ist": ist_now.isoformat(),
        "analysis_log": {
            "total_buy_score": int(round(buy_score)),
            "total_sell_score": int(round(sell_score)),
            "thresholds": {
                "buy_threshold": buy_thresh,
                "strong_buy_threshold": strong_buy_thresh,
                "sell_threshold": sell_thresh,
                "strong_sell_threshold": strong_sell_thresh
            },
            "regime_bias": {
                "btc_trend": regime,
                "bias_applied": regime_note
            },
            "veto_checks": {
                "profit_floor_ok": bool(profit_floor_ok)
            },
            "indicator_scores": indicator_scores
        },
        "indicators": {
            "rsi5m": _nz(rsi),
            "williamsR": _nz(willr),
            "cci5m": _nz(cci),
            "stc5m": _nz(stc),
            "tsi5m": _nz(tsi),
            "cmf5m": _nz(cmf),
            "cvd5m": cvd if isinstance(cvd, str) else None,
            "hma5m": hma if isinstance(hma, str) else None,
            "alma5m": alma if isinstance(alma, str) else None,
            "keltner5m": {
                "upper": _nz(keltner.get("upper")),
                "middle": _nz(keltner.get("middle")),
                "lower": _nz(keltner.get("lower")),
            },
            "boll5m": {
                "upper": _nz(boll.get("upper")),
                "middle": _nz(boll.get("middle")),
                "lower": _nz(boll.get("lower")),
            },
            "macd5m": {
                "macd": _nz(macd.get("macd")),
                "signal": _nz(macd.get("signal")),
                "histogram": _nz(macd.get("histogram")),
            },
            "marketTrend": float(market_trend)
        }
    }

# ============== MAIN ==============
if __name__ == "__main__":
    print("[INFO] Starting automated data fetch...")

    # 1) Universe
    coins = fetch_top_volume_coins(TOP_LIMIT)
    if not coins:
        print("[WARN] No coins fetched. Exiting.")
        raise SystemExit(0)
    print(f"[INFO] Found {len(coins)} coins to analyze.")

    # 2) BTC regime
    btc_data = fetch_binance_data("BTCUSDT", "5m", 120)
    btc_closes = [d["close"] for d in btc_data] if btc_data else []
    market_trend = calc_market_trend(btc_closes) if btc_closes else 0
    print(f"[INFO] Market Trend: {market_trend} ({'Bullish' if market_trend>0 else 'Bearish' if market_trend<0 else 'Neutral'})")

    # 3) Analyze coins
    results = []
    for sym in coins:
        print(f" - Analyzing {sym} ...")
        time.sleep(0.18)  # be gentle on the API
        data_5m = fetch_binance_data(sym, "5m", 120)
        if not data_5m:
            continue
        res = analyze_data(sym, data_5m, market_trend)
        if res:
            results.append(res)

    if results:
        strong = [r for r in results if "Strong" in r.get("signal", "")]
        print(f"\n[INFO] Analysis complete. Signals: {len(results)} | Strong: {len(strong)}")
        print("[INFO] Saving files...")

        # Timestamp for filenames in IST
        utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
        ist_now = utc_now.astimezone(pytz.timezone("Asia/Kolkata"))
        stamp = ist_now.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = "_STRONG" if strong else ""
        fname = f"signals_{stamp}{suffix}.json"

        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        archive_path = os.path.join(ARCHIVE_FOLDER, fname)
        with open(archive_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with open(LIVE_FILENAME, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[OK] Saved archive to {archive_path}")
        print(f"[OK] Updated {LIVE_FILENAME}")
    else:
        print("\n[INFO] No signals this run (neutrals are skipped).")
