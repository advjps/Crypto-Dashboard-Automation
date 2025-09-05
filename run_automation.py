# run_automation.py -- 10th Amendment (Complete automation script)
# Requirements: pandas, requests, pytz
# Place this in your repo and run. Adjust configuration constants below to tune.

import os
import time
import math
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import pytz

# ============== CONFIG ==============
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"
TOP_LIMIT = 70
BINANCE_FAPI = "https://fapi.binance.com"

# PROXY (set your proxy credentials or leave PROXY_IP as "YOUR_IP" to disable)
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

# Profit evaluation basis (ROI on margin)
LEVERAGE_FOR_PROFIT_EVAL = 7.0
MIN_PROFIT_MARGIN = 2.0           # min % on margin to pass
PROFIT_CEILING_MARGIN = 999.0     # removed ceiling by default (set high)

# ATR-based TP/SL clamps (as % of price) - used for fallback
TP_PCT_MIN, TP_PCT_MAX = 0.008, 0.016
SL_PCT_MIN, SL_PCT_MAX = 0.008, 0.020

# Thresholds & scoring tuning
STRONG_THRESHOLD = 70     # Confidence >= this => Strong
BUY_SELL_THRESHOLD = 40   # Confidence >= this => Buy/Sell
RECORD_NEUTRALS = False   # If False, don't save neutral signals into archive/live lists

# Lookback and other params
KL_INTERVAL = "5m"
KL_LIMIT = 200
REGIME_HORIZON = 50

# Confluence thresholds (for logging)
CONFLUENCE_MIN_SUCCESS_PCT = 68.0  # used for analysis (not hard gating)

# ============== Helpers ==============
def ist_now_iso():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(pytz.utc).astimezone(ist).isoformat()

def utc_now_iso():
    return datetime.now(pytz.utc).isoformat()

def safe_float(x, default=None):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def clamp01(x):
    return max(0.0, min(1.0, x))

def clamp0_100(x):
    return max(0, min(100, int(round(x))))

def json_safe(obj):
    # Convert numpy types if any to python native
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

# ============== Indicator implementations (robust & JSON-safe) ==============
def calc_ema(values, period):
    if not isinstance(values, list) or len(values) < period:
        return [None] * len(values)
    return pd.Series(values).ewm(span=period, adjust=False).mean().tolist()

def calc_rsi(values, period=14):
    if not isinstance(values, list) or len(values) < period + 1:
        return [None] * len(values)
    series = pd.Series(values)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs)))
    return rsi.tolist()

def get_last_valid_value(values):
    if values is None:
        return None
    for v in reversed(values):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return v
    return None

def calc_macd(values, fast=12, slow=26, signal=9):
    # return last macd, signal, histogram and raw series for normalization if needed
    if len(values) < slow + signal:
        # best-effort compute but may have Nones
        series = pd.Series(values)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
    else:
        series = pd.Series(values)
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    try:
        return {
            "macd": safe_float(macd_line.iloc[-1]),
            "signal": safe_float(signal_line.iloc[-1]),
            "histogram": safe_float(hist.iloc[-1]),
            "hist_series": hist.fillna(0).tolist()
        }
    except Exception:
        return {"macd": None, "signal": None, "histogram": None, "hist_series": []}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period:
        return {"upper": None, "middle": None, "lower": None, "percent_b": None}
    series = pd.Series(values)
    mean = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    if pd.isna(mean) or pd.isna(std):
        return {"upper": None, "middle": None, "lower": None, "percent_b": None}
    upper = mean + (mult * std)
    lower = mean - (mult * std)
    last = values[-1]
    percent_b = None
    try:
        percent_b = (last - lower) / (upper - lower) if (upper - lower) != 0 else None
    except Exception:
        percent_b = None
    return {"upper": safe_float(upper), "middle": safe_float(mean), "lower": safe_float(lower), "percent_b": safe_float(percent_b)}

def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return safe_float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1])

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period:
        return None
    tp = pd.Series([(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)])
    mean = tp.rolling(window=period).mean().iloc[-1]
    mean_dev = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False).iloc[-1]
    if mean_dev == 0 or pd.isna(mean) or pd.isna(mean_dev):
        return 0.0
    return safe_float((tp.iloc[-1] - mean) / (0.015 * mean_dev))

def calc_keltner(values_high, values_low, values_close, period=20, atr_mult=1.5):
    # Keltner Channel: center = EMA(period) of close, upper = center + atr_mult * ATR, lower = center - atr_mult * ATR
    if len(values_close) < period:
        return {"upper": None, "middle": None, "lower": None}
    center = pd.Series(values_close).ewm(span=period, adjust=False).mean().iloc[-1]
    atr = calc_atr(values_high, values_low, values_close, period)
    if atr is None:
        return {"upper": None, "middle": None, "lower": None}
    upper = center + atr_mult * atr
    lower = center - atr_mult * atr
    return {"upper": safe_float(upper), "middle": safe_float(center), "lower": safe_float(lower)}

def calc_hma(values, period=21):
    # Hull Moving Average approximation
    if len(values) < period:
        return None
    series = pd.Series(values)
    half_len = int(period/2)
    wma_half = series.ewm(span=half_len, adjust=False).mean()
    wma_full = series.ewm(span=period, adjust=False).mean()
    diff = 2 * wma_half - wma_full
    hma = diff.ewm(span=int(math.sqrt(period)), adjust=False).mean()
    return safe_float(hma.iloc[-1])

def alma(values, window=9, offset=0.85, sigma=6.0):
    # Arnaud Legoux Moving Average (lightweight)
    if len(values) < window:
        return None
    series = pd.Series(values)
    m = offset * (window - 1)
    s = window / sigma
    w = [math.exp(-((i - m) ** 2) / (2 * (s ** 2))) for i in range(window)]
    w = [wi / sum(w) for wi in w]
    vals = series.rolling(window).apply(lambda x: sum([a*b for a,b in zip(w, x)]), raw=True)
    return safe_float(vals.iloc[-1])

def calc_tsi(values, r1=25, r2=13):
    # True Strength Index simplified implementation (price momentum)
    if len(values) < max(r1, r2) + 5:
        return None
    p = pd.Series(values)
    delta = p.diff()
    abs_delta = delta.abs()
    ema1 = delta.ewm(span=r1, adjust=False).mean()
    ema2 = ema1.ewm(span=r2, adjust=False).mean()
    abs_ema1 = abs_delta.ewm(span=r1, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=r2, adjust=False).mean()
    tsi = 100 * (ema2 / abs_ema2) if abs_ema2.iloc[-1] != 0 else 0.0
    return safe_float(tsi.iloc[-1])

def calc_stc(values, fast=23, slow=50, cycle=10):
    # Approximate Schaff Trend Cycle using MACD-like approach + %K smoothing
    # This is a lightweight approximation to provide a numeric value.
    try:
        macd = pd.Series(values).ewm(span=fast, adjust=False).mean() - pd.Series(values).ewm(span=slow, adjust=False).mean()
        macd_signal = macd.ewm(span=cycle, adjust=False).mean()
        macd_hist = macd - macd_signal
        # Normalize histogram between 0..100 using rolling min/max
        window = max(10, int(len(values)/5))
        mn = macd_hist.rolling(window).min().iloc[-1] if len(macd_hist) >= window else macd_hist.min()
        mx = macd_hist.rolling(window).max().iloc[-1] if len(macd_hist) >= window else macd_hist.max()
        denom = (mx - mn) if (mx - mn) != 0 else 1e-8
        stc = 100 * (macd_hist.iloc[-1] - mn) / denom
        return safe_float(stc)
    except Exception:
        return None

def calc_vol_profile(closes, highs, lows, volumes, bins=10):
    try:
        df = pd.DataFrame({'price': closes, 'volume': volumes})
        price_range = max(highs) - min(lows)
        if price_range == 0:
            return {'bullish_score': 0, 'bearish_score': 0}
        # Group volume into bins across price and find POC
        cuts = pd.cut(df['price'], bins=bins)
        vol_by_bin = df.groupby(cuts, observed=False)['volume'].sum()
        if vol_by_bin.empty:
            return {'bullish_score': 0, 'bearish_score': 0}
        poc_interval = vol_by_bin.idxmax()
        try:
            poc = (poc_interval.left + poc_interval.right) / 2.0
        except Exception:
            poc = df['price'].median()
        current_price = closes[-1]
        # scoring is lightweight: 5 if inside POC window, 3 if above/below, 0 otherwise
        if current_price > poc:
            return {'bullish_score': 3, 'bearish_score': 0}
        if current_price < poc:
            return {'bullish_score': 0, 'bearish_score': 3}
        return {'bullish_score': 5, 'bearish_score': 5}
    except Exception:
        return {'bullish_score': 1, 'bearish_score': 1}

def calc_cvd(closes, volumes):
    # Approximate Cumulative Volume Delta using direction-aware volume:
    # if price_up (close>prev_close) => +volume, else -volume
    if len(closes) < 2 or len(volumes) < 2:
        return {"trend": "unknown", "value": 0.0}
    deltas = []
    for i in range(1, len(closes)):
        direction = 1 if closes[i] > closes[i - 1] else (-1 if closes[i] < closes[i - 1] else 0)
        deltas.append(direction * volumes[i])
    cum = sum(deltas[-20:])  # sum last 20 bars
    # simple classification
    trend = "rising" if cum > 0 else ("falling" if cum < 0 else "flat")
    return {"trend": trend, "value": safe_float(cum)}

def calc_market_trend(closes):
    # Simple market trend based on EMA20/EMA50 & last price
    if len(closes) < 50:
        return 0.0
    ema20 = get_last_valid_value(calc_ema(closes, 20))
    ema50 = get_last_valid_value(calc_ema(closes, 50))
    if ema20 is None or ema50 is None:
        return 0.0
    last = closes[-1]
    if ema20 > ema50 and last > ema20:
        return 10.0
    if ema20 > ema50:
        return 5.0
    if ema20 < ema50 and last < ema20:
        return -10.0
    if ema20 < ema50:
        return -5.0
    return 0.0

# ============== Data fetchers ==============
def fetch_top_volume_coins(limit=TOP_LIMIT):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
        resp = requests.get(url, proxies=proxies, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        usdt = [t for t in data if 'symbol' in t and t['symbol'].endswith('USDT')]
        usdt_sorted = sorted(usdt, key=lambda x: float(x.get('quoteVolume', 0) or 0), reverse=True)
        return [c['symbol'] for c in usdt_sorted[:limit]]
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def fetch_binance_data(symbol, timeframe=KL_INTERVAL, limit=KL_LIMIT):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        resp = requests.get(url, proxies=proxies, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        cleaned = []
        for d in data:
            cleaned.append({
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
                "open_time": int(d[0]),
                "close_time": int(d[6])
            })
        return cleaned
    except Exception as e:
        print(f"  - Could not fetch data for {symbol}: {e}")
        return []

# ============== 10th Amendment: analyze_data() ==============
def analyze_data(symbol, data5m, market_trend):
    """
    Analyze data and return a JSON-serializable dict with:
    - coin, price, tp, sl, leverage, confidence, signal, estimated_profit, regime, timestamps
    - analysis_log with components, indicator_scores, confluence_flags, would_be_strong_if
    - indicators raw values
    """
    # Minimal data check
    if not data5m or len(data5m) < 60:
        return None

    closes = [d["close"] for d in data5m]
    highs = [d["high"] for d in data5m]
    lows  = [d["low"] for d in data5m]
    volumes = [d["volume"] for d in data5m]
    current_price = safe_float(closes[-1])
    if current_price is None:
        return None

    # --- Indicators ---
    latest_rsi = get_last_valid_value(calc_rsi(closes, 14))
    macd = calc_macd(closes, 12, 26, 9)
    macd_hist = safe_float(macd.get("histogram"))
    macd_hist_series = macd.get("hist_series", [])
    boll = calc_bollinger(closes, 20, 2)
    atr = calc_atr(highs, lows, closes, 14) or (current_price * 0.002)
    latest_cci = calc_cci(highs, lows, closes, 20)
    vol_profile = calc_vol_profile(closes, highs, lows, volumes)
    latest_ema50 = get_last_valid_value(calc_ema(closes, 50))
    kelt = calc_keltner(highs, lows, closes, 20, atr_mult=1.5)
    hma = calc_hma(closes, 21)
    alma_val = alma(closes, window=9)
    tsi_val = calc_tsi(closes)
    stc_val = calc_stc(closes)
    cvd_obj = calc_cvd(closes, volumes)
    # Derived flags
    percent_b = boll.get("percent_b")
    # Avoid None issues
    latest_rsi = safe_float(latest_rsi)
    latest_cci = safe_float(latest_cci)
    latest_ema50 = safe_float(latest_ema50)
    tsi_val = safe_float(tsi_val)
    stc_val = safe_float(stc_val)
    hma = safe_float(hma)
    alma_val = safe_float(alma_val)
    kelt_upper = safe_float(kelt.get("upper"))
    kelt_lower = safe_float(kelt.get("lower"))

    # --- Component / indicator scoring (separate engines) ---
    indicator_scores = {}
    base_buy = base_sell = 0.0
    momentum_buy = momentum_sell = 0.0
    trend_buy = trend_sell = 0.0
    volume_buy = volume_sell = 0.0
    confluence_bonus = 0.0
    profit_check = 0.0

    # --- BaseScore mapping (Buy) ---
    # Price vs Bollinger lower
    if percent_b is not None and percent_b <= 0.0:
        indicator_scores['boll_buy_touch'] = 20.0
        base_buy += 20.0
    else:
        indicator_scores['boll_buy_touch'] = 0.0

    # RSI oversold
    if latest_rsi is not None and latest_rsi <= 30:
        indicator_scores['rsi_buy'] = 12.0
        base_buy += 12.0
    else:
        indicator_scores['rsi_buy'] = 0.0

    # CCI extreme
    if latest_cci is not None and latest_cci >= 100:
        indicator_scores['cci_buy'] = 8.0
        base_buy += 8.0
    else:
        indicator_scores['cci_buy'] = 0.0

    # Keltner lower touch or %B very low
    kelt_buy_touch = 0
    if kelt_lower is not None and current_price <= kelt_lower:
        indicator_scores['keltner_buy_touch'] = 6.0
        base_buy += 6.0
        kelt_buy_touch = 1
    else:
        indicator_scores['keltner_buy_touch'] = 0.0
    if percent_b is not None and percent_b <= 0.05:
        indicator_scores['percentb_buy'] = 6.0
        base_buy += 6.0
    else:
        # ensure field exists
        indicator_scores['percentb_buy'] = 0.0

    # --- BaseScore mapping (Sell) ---
    if percent_b is not None and percent_b >= 1.0:
        indicator_scores['boll_sell_touch'] = 20.0
        base_sell += 20.0
    else:
        indicator_scores['boll_sell_touch'] = 0.0

    if latest_rsi is not None and latest_rsi >= 70:
        indicator_scores['rsi_sell'] = 12.0
        base_sell += 12.0
    else:
        indicator_scores['rsi_sell'] = 0.0

    if latest_cci is not None and latest_cci <= -100:
        indicator_scores['cci_sell'] = 8.0
        base_sell += 8.0
    else:
        indicator_scores['cci_sell'] = 0.0

    if percent_b is not None and percent_b >= 0.95:
        indicator_scores['percentb_sell'] = 6.0
        base_sell += 6.0
    else:
        indicator_scores['percentb_sell'] = 0.0

    # --- Momentum (MACD normalized) ---
    # Normalize macd_hist relative to ATR to prevent domination by large raw hist
    macd_norm = 0.0
    if macd_hist is not None and atr and atr > 0:
        macd_norm = clamp01(abs(macd_hist) / max(atr * 1.0, 1e-8))
    hist_sign = 0
    if macd_hist is not None:
        hist_sign = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)

    # Buy momentum: positive MACD hist helps buy
    indicator_scores['macd_buy'] = safe_float(10.0 * macd_norm if hist_sign > 0 else 0.0)
    momentum_buy += indicator_scores['macd_buy']

    # Sell momentum: negative MACD hist helps sell
    indicator_scores['macd_sell'] = safe_float(10.0 * macd_norm if hist_sign < 0 else 0.0)
    momentum_sell += indicator_scores['macd_sell']

    # TSI & STC contributions
    indicator_scores['tsi_val'] = safe_float(tsi_val or 0.0)
    indicator_scores['stc_val'] = safe_float(stc_val or 0.0)
    # small scoring from TSI / STC
    if tsi_val is not None and tsi_val > 0:
        momentum_buy += 5.0
        indicator_scores['tsi_buy'] = 5.0
    else:
        indicator_scores['tsi_buy'] = 0.0
    if tsi_val is not None and tsi_val < 0:
        momentum_sell += 5.0
        indicator_scores['tsi_sell'] = 5.0
    else:
        indicator_scores['tsi_sell'] = 0.0

    if stc_val is not None and stc_val > 50:
        momentum_buy += 5.0
        indicator_scores['stc_buy'] = 5.0
    else:
        indicator_scores['stc_buy'] = 0.0
    if stc_val is not None and stc_val < 50:
        momentum_sell += 5.0
        indicator_scores['stc_sell'] = 5.0
    else:
        indicator_scores['stc_sell'] = 0.0

    # Recent candle thrust simple heuristic: compare last two closes
    recent_thrust = 0.0
    if len(closes) >= 3:
        if closes[-1] > closes[-2] and closes[-2] > closes[-3]:
            recent_thrust = 5.0  # bullish thrust
            momentum_buy += 5.0
            indicator_scores['thrust_buy'] = 5.0
            indicator_scores['thrust_sell'] = 0.0
        elif closes[-1] < closes[-2] and closes[-2] < closes[-3]:
            recent_thrust = -5.0
            momentum_sell += 5.0
            indicator_scores['thrust_sell'] = 5.0
            indicator_scores['thrust_buy'] = 0.0
        else:
            indicator_scores['thrust_buy'] = 0.0
            indicator_scores['thrust_sell'] = 0.0
    else:
        indicator_scores['thrust_buy'] = 0.0
        indicator_scores['thrust_sell'] = 0.0

    # --- Trend alignment ---
    # BTC market_trend used as bias
    trend_bias_buy = 0.0
    trend_bias_sell = 0.0
    if market_trend >= 5:
        trend_bias_buy += 8.0
    elif market_trend <= -5:
        trend_bias_sell += 8.0
    # EMA50 slope
    if latest_ema50 is not None:
        ema50_series = calc_ema(closes, 50)
        ema50_prev = ema50_series[-2] if len(ema50_series) >= 2 else None
        if ema50_prev is not None:
            if ema50_series[-1] > ema50_prev and current_price > ema50_series[-1]:
                trend_buy += 7.0
                indicator_scores['ema50_buy'] = 7.0
            else:
                indicator_scores['ema50_buy'] = 0.0
            if ema50_series[-1] < ema50_prev and current_price < ema50_series[-1]:
                trend_sell += 7.0
                indicator_scores['ema50_sell'] = 7.0
            else:
                indicator_scores['ema50_sell'] = 0.0
        else:
            indicator_scores['ema50_buy'] = 0.0
            indicator_scores['ema50_sell'] = 0.0
    else:
        indicator_scores['ema50_buy'] = 0.0
        indicator_scores['ema50_sell'] = 0.0

    # HMA / ALMA trend contributions
    if hma is not None and hma > 0:
        # crude direction: compare HMA last vs previous
        # recompute HMA series quickly
        hma_series = []
        try:
            # compute small rolling HMA values for direction if possible
            hma_series = pd.Series(closes).rolling(window=21).apply(lambda x: calc_hma(x.tolist(), 21)).dropna().tolist()
        except Exception:
            hma_series = []
        if len(hma_series) >= 2 and hma_series[-1] > hma_series[-2]:
            trend_buy += 5.0
            indicator_scores['hma_buy'] = 5.0
        else:
            indicator_scores['hma_buy'] = 0.0
        if len(hma_series) >= 2 and hma_series[-1] < hma_series[-2]:
            trend_sell += 5.0
            indicator_scores['hma_sell'] = 5.0
        else:
            indicator_scores['hma_sell'] = 0.0
    else:
        indicator_scores['hma_buy'] = 0.0
        indicator_scores['hma_sell'] = 0.0

    if alma_val is not None:
        indicator_scores['alma_val'] = alma_val
        # crude rule: if ALMA recent slope up
        try:
            alma_series = pd.Series(closes).rolling(window=9).apply(lambda x: alma(x.tolist(), window=9)).dropna().tolist()
        except Exception:
            alma_series = []
        if len(alma_series) >= 2 and alma_series[-1] > alma_series[-2]:
            trend_buy += 5.0
            indicator_scores['alma_buy'] = 5.0
        else:
            indicator_scores['alma_buy'] = 0.0
        if len(alma_series) >= 2 and alma_series[-1] < alma_series[-2]:
            trend_sell += 5.0
            indicator_scores['alma_sell'] = 5.0
        else:
            indicator_scores['alma_sell'] = 0.0
    else:
        indicator_scores['alma_buy'] = 0.0
        indicator_scores['alma_sell'] = 0.0

    # Add market trend bias
    trend_buy += trend_bias_buy
    trend_sell += trend_bias_sell

    # Cap trend components to configured max (0..15)
    trend_buy = min(15.0, trend_buy)
    trend_sell = min(15.0, trend_sell)

    # --- Volume / Order-flow ---
    if vol_profile and vol_profile.get("bullish_score", 0) > 0:
        volume_buy += 5.0
        indicator_scores['volprofile_buy'] = float(vol_profile.get("bullish_score", 0))
    else:
        indicator_scores['volprofile_buy'] = 0.0

    if vol_profile and vol_profile.get("bearish_score", 0) > 0:
        volume_sell += 5.0
        indicator_scores['volprofile_sell'] = float(vol_profile.get("bearish_score", 0))
    else:
        indicator_scores['volprofile_sell'] = 0.0

    # CVD trend
    cvd_trend = cvd_obj.get("trend", "flat")
    if cvd_trend == "falling":
        volume_buy += 5.0
        indicator_scores['cvd_buy'] = 5.0
        indicator_scores['cvd_trend'] = "falling"
    else:
        indicator_scores['cvd_buy'] = 0.0
    if cvd_trend == "rising":
        volume_sell += 5.0
        indicator_scores['cvd_sell'] = 5.0
        indicator_scores['cvd_trend'] = "rising"
    else:
        indicator_scores['cvd_sell'] = 0.0

    # --- Confluence bonuses (additive only) ---
    confluence_flags = []
    # RSI<=30 & CVD falling
    if latest_rsi is not None and latest_rsi <= 30 and cvd_trend == "falling":
        confluence_bonus += 12.0
        confluence_flags.append("RSI<=30 & CVD_falling")
    # RSI<=30 & HMA up & ALMA up
    if latest_rsi is not None and latest_rsi <= 30 and ("hma_buy" in indicator_scores and indicator_scores['hma_buy'] > 0) and ("alma_buy" in indicator_scores and indicator_scores['alma_buy'] > 0):
        confluence_bonus += 16.0
        confluence_flags.append("RSI<=30 & HMA_up & ALMA_up")
    # CVD falling & HMA up
    if cvd_trend == "falling" and ("hma_buy" in indicator_scores and indicator_scores['hma_buy'] > 0):
        confluence_bonus += 10.0
        confluence_flags.append("CVD_falling & HMA_up")
    # RSI>=70 & CVD rising
    if latest_rsi is not None and latest_rsi >= 70 and cvd_trend == "rising":
        confluence_bonus += 12.0
        confluence_flags.append("RSI>=70 & CVD_rising")
    # RSI>=70 & HMA down & ALMA down
    if latest_rsi is not None and latest_rsi >= 70 and ("hma_sell" in indicator_scores and indicator_scores['hma_sell'] > 0) and ("alma_sell" in indicator_scores and indicator_scores['alma_sell'] > 0):
        confluence_bonus += 16.0
        confluence_flags.append("RSI>=70 & HMA_down & ALMA_down")
    # CVD rising & HMA down
    if cvd_trend == "rising" and ("hma_sell" in indicator_scores and indicator_scores['hma_sell'] > 0):
        confluence_bonus += 10.0
        confluence_flags.append("CVD_rising & HMA_down")

    # clamp confluence bonus to max 20
    confluence_bonus = min(20.0, confluence_bonus)

    # --- Profit / Risk sizing check ---
    # dynamic TP/SL based on ATR (TP = +/- 1.8*ATR unless overridden)
    tp_factor = 1.8
    sl_factor = 1.8
    tp = current_price
    sl = current_price
    tp = current_price + (atr * tp_factor)
    sl = current_price - (atr * sl_factor)
    profit_pct_on_margin = 0.0
    if current_price > 0:
        profit_pct_on_margin = abs(((tp - current_price) / current_price) * 100.0) * LEVERAGE_FOR_PROFIT_EVAL
    indicator_scores['estimated_profit_margin_pct'] = safe_float(profit_pct_on_margin)

    if profit_pct_on_margin >= MIN_PROFIT_MARGIN:
        profit_check += 3.0
    if profit_pct_on_margin >= 4.0:
        profit_check = min(5.0, profit_check + 2.0)

    # --- Compose components (cap them to their maxima) ---
    base_buy = min(40.0, base_buy)
    base_sell = min(40.0, base_sell)
    momentum_buy = min(20.0, momentum_buy)
    momentum_sell = min(20.0, momentum_sell)
    volume_buy = min(10.0, volume_buy)
    volume_sell = min(10.0, volume_sell)
    profit_check = min(5.0, profit_check)
    # We already capped confluence_bonus

    # Total confidence by engine
    confidence_buy_raw = base_buy + momentum_buy + trend_buy + volume_buy + confluence_bonus + profit_check
    confidence_sell_raw = base_sell + momentum_sell + trend_sell + volume_sell + confluence_bonus + profit_check

    confidence_buy = clamp0_100(confidence_buy_raw)
    confidence_sell = clamp0_100(confidence_sell_raw)

    # initial directional preference (higher of the two)
    engine = "Neutral"
    chosen_confidence = 0
    signal_label = "Neutral"
    if confidence_buy > confidence_sell and confidence_buy >= BUY_SELL_THRESHOLD:
        engine = "Buy"
        chosen_confidence = confidence_buy
        if confidence_buy >= STRONG_THRESHOLD:
            signal_label = "Strong Buy"
        else:
            signal_label = "Buy"
    elif confidence_sell > confidence_buy and confidence_sell >= BUY_SELL_THRESHOLD:
        engine = "Sell"
        chosen_confidence = confidence_sell
        if confidence_sell >= STRONG_THRESHOLD:
            signal_label = "Strong Sell"
        else:
            signal_label = "Sell"
    else:
        engine = "Neutral"
        chosen_confidence = max(confidence_buy, confidence_sell)
        signal_label = "Neutral"

    # Decide leverage suggestion
    leverage = 5
    if chosen_confidence >= 80:
        leverage = 9
    elif chosen_confidence >= 65:
        leverage = 7
    elif chosen_confidence >= 50:
        leverage = 6

    # Build analysis_log components
    components = {
        "base_score": float(base_buy if engine == "Buy" else base_sell if engine == "Sell" else max(base_buy, base_sell)),
        "momentum_score": float(momentum_buy if engine == "Buy" else momentum_sell if engine == "Sell" else max(momentum_buy, momentum_sell)),
        "trend_score": float(trend_buy if engine == "Buy" else trend_sell if engine == "Sell" else max(trend_buy, trend_sell)),
        "volume_score": float(volume_buy if engine == "Buy" else volume_sell if engine == "Sell" else max(volume_buy, volume_sell)),
        "confluence_bonus": float(confluence_bonus),
        "profit_check": float(profit_check)
    }

    # Top missing components calculation for would_be_strong_if
    current_conf = chosen_confidence
    missing_points = max(0, STRONG_THRESHOLD - current_conf)
    # compute per-component deficits sorted descending
    component_gaps = []
    for k, v in components.items():
        # potential maximum for each component (as we designed): base(40), momentum(20), trend(15), volume(10), confluence(20), profit(5)
        potential_max = {
            "base_score": 40.0, "momentum_score": 20.0, "trend_score": 15.0,
            "volume_score": 10.0, "confluence_bonus": 20.0, "profit_check": 5.0
        }.get(k, 0.0)
        gap = max(0.0, potential_max - v)
        component_gaps.append({"component": k, "gap": float(round(gap, 2))})

    component_gaps_sorted = sorted(component_gaps, key=lambda x: x["gap"], reverse=True)[:3]

    would_be = {
        "if_confidence_needed": STRONG_THRESHOLD,
        "missing_points": int(math.ceil(missing_points)),
        "top_missing_components": component_gaps_sorted
    }

    # Build the final return object (JSON-safe)
    result = {
        "coin": symbol,
        "price": float(round(current_price, 6)),
        "tp": float(round(tp, 6)),
        "sl": float(round(sl, 6)),
        "leverage": f"{leverage}x",
        "confidence": int(chosen_confidence),
        "signal": signal_label,
        "estimated_profit": f"{profit_pct_on_margin:.2f}%",
        "regime": ("Bullish" if market_trend > 0 else "Bearish" if market_trend < 0 else "Neutral"),
        "signal_time_utc": utc_now_iso(),
        "signal_time_ist": ist_now_iso(),
        "analysis_log": {
            "engine": engine,
            "confidence": int(chosen_confidence),
            "components": components,
            "indicator_scores": {k: (float(v) if isinstance(v, (int,float)) else v) for k, v in indicator_scores.items()},
            "confluence_flags": confluence_flags,
            "would_be_strong_if": would_be,
            "regime_bias": float(market_trend)
        },
        "indicators": {
            "rsi5m": safe_float(latest_rsi),
            "macd5m": {"macd": safe_float(macd.get("macd")), "signal": safe_float(macd.get("signal")), "histogram": safe_float(macd_hist)},
            "boll5m": {"upper": safe_float(boll.get("upper")), "middle": safe_float(boll.get("middle")), "lower": safe_float(boll.get("lower")), "percent_b": safe_float(percent_b)},
            "cci5m": safe_float(latest_cci),
            "marketTrend": float(market_trend),
            "volProfile": {"bullish_score": float(vol_profile.get("bullish_score", 0)), "bearish_score": float(vol_profile.get("bearish_score", 0))},
            "ema50_5m": safe_float(latest_ema50),
            "keltner5m": {"upper": kelt_upper, "middle": safe_float(kelt.get("middle")), "lower": kelt_lower},
            "hma5m": safe_float(hma),
            "alma5m": safe_float(alma_val),
            "tsi5m": safe_float(tsi_val),
            "stc5m": safe_float(stc_val),
            "cvd5m": {"trend": cvd_obj.get("trend"), "value": safe_float(cvd_obj.get("value"))}
        }
    }

    # ensure json safety: convert any non-serializable through json_safe
    # but prefer simple types
    return json.loads(json.dumps(result, default=json_safe))

# ============== Main Execution ==============
def main():
    print("Starting automated data fetch...")
    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting.")
        return

    print(f"Found {len(top_coins)} coins to analyze.")
    # ensure folders exist
    os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

    # Get BTC data for market trend
    btc_data = fetch_binance_data("BTCUSDT", timeframe=KL_INTERVAL, limit=REGIME_HORIZON+10)
    btc_closes = [d["close"] for d in btc_data] if btc_data else []
    market_trend = calc_market_trend(btc_closes)
    print(f"Market Trend: {market_trend}")

    all_results = []
    for coin in top_coins:
        print(f" - Analyzing {coin} ...")
        time.sleep(0.15)
        data_5m = fetch_binance_data(coin)
        if not data_5m:
            continue
        try:
            res = analyze_data(coin, data_5m, market_trend)
            if res is None:
                continue
            # Optionally skip neutral records
            if not RECORD_NEUTRALS and res.get("signal", "") == "Neutral":
                continue
            all_results.append(res)
        except Exception as e:
            print(f"   ERROR analyzing {coin}: {e}")

    strong_signals = [s for s in all_results if "Strong" in s.get("signal", "")]
    print(f"\nAnalysis complete. Found {len(strong_signals)} strong signals.")

    # Timestamp for filenames (IST)
    utc_now = datetime.now(pytz.utc)
    ist_tz = pytz.timezone("Asia/Kolkata")
    ist_now = utc_now.astimezone(ist_tz)
    timestamp_str = ist_now.strftime("%Y-%m-%d_%H-%M-%S")

    file_suffix = "_STRONG" if strong_signals else ""
    archive_filename = f"signals_{timestamp_str}{file_suffix}.json"
    archive_filepath = os.path.join(ARCHIVE_FOLDER, archive_filename)

    # Save archive and live files
    if all_results:
        with open(archive_filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Archive file saved to {archive_filepath}")

        with open(LIVE_FILENAME, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Live data file saved as {LIVE_FILENAME}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
