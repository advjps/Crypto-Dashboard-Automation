#!/usr/bin/env python3
# run_automation.py (10B - 10A + Supervisor Promoter Option 3)
# Requirements: pandas, requests, pytz
# Usage: python run_automation.py

import os
import time
import json
import math
import requests
import pandas as pd
import pytz

from datetime import datetime, timezone, timedelta

# ----------------- PROXY CONFIGURATION -----------------
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

# ----------------- GENERAL CONFIG -----------------
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"
TOP_LIMIT = 70
BINANCE_FAPI = "https://fapi.binance.com"

# Profit evaluation basis (ROI on margin)
LEVERAGE_FOR_PROFIT_EVAL = 7.0

# User requested fixed TP/SL as *price-move percent* so that margin profit matches:
# TP_price_move_pct = 0.72% -> margin profit ~= 0.72 * leverage = 5.04% (with 7x)
# SL_price_move_pct = 3.00% -> margin loss ~= 3.00 * leverage = 21.00%
FORCE_FIXED_TP_SL = True
TP_PRICE_MOVE_PCT = 0.72 / 100.0    # 0.0072 (=0.72% price move)
SL_PRICE_MOVE_PCT = 3.00 / 100.0    # 0.03   (=3% price move)

# Keep ATR fallback & clamps (not used if FORCE_FIXED_TP_SL=True)
TP_PCT_MIN, TP_PCT_MAX = 0.0072, 0.016
SL_PCT_MIN, SL_PCT_MAX = 0.008, 0.020
TP_ATR_FACTOR = 1.8
SL_ATR_FACTOR = 1.8

# Scoring thresholds / caps
STRONG_THRESHOLD = 65   # can tune
MIN_CONF_FOR_SIGNAL = 40

# HMA gatekeeper settings (applies only to signals within HMA_GAP_THRESHOLD below STRONG)
HMA_GATEKEEPER_ENABLED = True
HMA_GAP_THRESHOLD = 15            # points below STRONG threshold where gatekeeper applies
HMA_GATEKEEPER_BOOST = 8         # additive boost if gatekeeper condition satisfied

# Supervisor Promoter (Option 3) settings
# Applies only to signals where hma_gate_applied == True and still below STRONG
SUPERVISOR_ENABLED = True
SUPERVISOR_BOOST = 7
SUP_HIST_NORM_THRESH = 0.6
SUP_HMA_SLOPE_PCT_THRESH = 0.15  # percent (e.g. 0.15 => 0.15%)
# cvd_support: buy-supporting when cvd trend is 'falling' (hidden buy) ; sell-supporting when 'rising'

# Regime hysteresis
REGIME_HOLD_MINUTES = 60
REGIME_CONFIRM_BARS = 2
REGIME_STATE_FILE = os.path.join(ARCHIVE_FOLDER, ".regime_state.json")

# Request config
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 0.8

# Lookback for kline fetch
KLINE_LIMIT_5M = 200

# Ensure folders exist
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

# ---------- Utilities ----------
def request_with_retries(url, params=None, proxies_local=proxies, timeout=REQUEST_TIMEOUT):
    last_exc = None
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, params=params, proxies=proxies_local, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(REQUEST_BACKOFF * (1 + attempt))
    raise last_exc

def utcnow_iso():
    return datetime.now(timezone.utc).isoformat()

def utc_to_ist_iso(utc_iso):
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = pytz.timezone("Asia/Kolkata")
        return dt.astimezone(ist).isoformat()
    except Exception:
        return ""

def safe_cast_float(x):
    try:
        return float(x)
    except Exception:
        return None

def get_last_valid_value(values):
    """Return last non-None, non-NaN value from a sequence"""
    for v in reversed(values):
        try:
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return v
        except Exception:
            return v
    return None

# ----------------- INDICATORS -----------------
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
    rsi = (100 - (100 / (1 + rs))).tolist()
    return rsi

def calc_macd(values, fast=12, slow=26, signal=9):
    series = pd.Series(values)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = (macd_line - signal_line)
    try:
        return {'macd': float(macd_line.iloc[-1]), 'signal': float(signal_line.iloc[-1]), 'histogram': float(histogram.iloc[-1])}
    except Exception:
        return {'macd': None, 'signal': None, 'histogram': 0.0}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period:
        return {'upper': None, 'middle': None, 'lower': None, 'percent_b': None}
    series = pd.Series(values)
    mean = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    if pd.isna(mean) or pd.isna(std):
        return {'upper': None, 'middle': None, 'lower': None, 'percent_b': None}
    upper = float(mean + (mult * std))
    middle = float(mean)
    lower = float(mean - (mult * std))
    percent_b = None
    try:
        if upper != lower:
            percent_b = float((values[-1] - lower) / (upper - lower))
    except Exception:
        percent_b = None
    return {'upper': upper, 'middle': middle, 'lower': lower, 'percent_b': percent_b}

def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    try:
        return float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1])
    except Exception:
        return None

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period:
        return None
    tp = pd.Series([(h + l + c) / 3.0 for h, l, c in zip(highs, lows, closes)])
    mean = tp.rolling(window=period).mean().iloc[-1]
    mean_dev = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False).iloc[-1]
    if mean_dev == 0 or pd.isna(mean) or pd.isna(mean_dev):
        return 0.0
    return float((tp.iloc[-1] - mean) / (0.015 * mean_dev))

def calc_vol_profile(closes, highs, lows, volumes):
    try:
        df = pd.DataFrame({'price': closes, 'volume': volumes})
        price_range = max(highs) - min(lows)
        if price_range == 0:
            return {'bullish_score': 0.0, 'bearish_score': 0.0}
        bins = 10
        df['bin'] = pd.cut(df['price'], bins=bins)
        agg = df.groupby('bin', observed=False)['volume'].sum()
        if agg.empty:
            return {'bullish_score': 0.0, 'bearish_score': 0.0}
        poc_interval = agg.idxmax()
        if hasattr(poc_interval, 'mid'):
            poc = float(poc_interval.mid)
        else:
            poc = float(df['price'].mean())
        current_price = closes[-1]
        if current_price > poc:
            return {'bullish_score': 3.0, 'bearish_score': 0.0}
        if current_price < poc:
            return {'bullish_score': 0.0, 'bearish_score': 3.0}
        return {'bullish_score': 5.0, 'bearish_score': 5.0}
    except Exception:
        return {'bullish_score': 1.0, 'bearish_score': 1.0}

# ----------------- OPTIONAL INDICATOR IMPLEMENTATIONS / STUBS -----------------
def calc_hma(values, period=16):
    try:
        series = pd.Series(values)
        half = max(1, int(period/2))
        wma_half = series.rolling(window=half).mean()
        wma_full = series.rolling(window=period).mean()
        diff = 2 * wma_half - wma_full
        hma = diff.rolling(window=max(1,int(math.sqrt(period)))).mean()
        return hma.tolist()
    except Exception:
        return None

def calc_alma(values, window=9, offset=0.85, sigma=6):
    try:
        s = pd.Series(values)
        return s.rolling(window=window, win_type='gaussian').mean(std=sigma).tolist()
    except Exception:
        return None

def calc_tsi(values, r=25, s=13):
    try:
        series = pd.Series(values)
        delta = series.diff()
        num = delta.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
        den = delta.abs().ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
        tsi = (100 * (num / den)).tolist()
        return get_last_valid_value(tsi)
    except Exception:
        return None

def calc_stc(values):
    # STC is tuning-heavy; provide None if not implementable reliably
    return None

def calc_cvd(values, volumes):
    try:
        if not values or not volumes:
            return None
        deltas = []
        for i in range(1, len(values)):
            sign = 1 if (values[i] - values[i-1]) > 0 else (-1 if (values[i] - values[i-1]) < 0 else 0)
            deltas.append(sign * (volumes[i] or 0.0))
        window = 20
        recent = deltas[-window:] if len(deltas) >= window else deltas
        cvd_val = float(sum(recent)) if recent else 0.0
        trend = "rising" if cvd_val > 0 else ("falling" if cvd_val < 0 else "flat")
        return {"trend": trend, "value": cvd_val}
    except Exception:
        return None

# ----------------- DATA FETCH -----------------
def fetch_top_volume_coins(limit=TOP_LIMIT):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
        resp = request_with_retries(url)
        data = resp.json()
        usdt_pairs = [t for t in data if 'symbol' in t and t['symbol'].endswith('USDT')]
        usdt_pairs_sorted = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0) or 0), reverse=True)
        if not usdt_pairs_sorted:
            return []
        symbols = [c['symbol'] for c in usdt_pairs_sorted[:limit]]
        return symbols
    except Exception as e:
        print(f"[Error] fetch_top_volume_coins: {e}")
        return []

def fetch_binance_data(symbol, timeframe='5m', limit=KLINE_LIMIT_5M):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": timeframe, "limit": limit}
        resp = request_with_retries(url, params=params)
        data = resp.json()
        out = []
        for d in data:
            out.append({
                "open_time": int(d[0]),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
                "close_time": int(d[6])
            })
        return out
    except Exception as e:
        print(f"  - Could not fetch data for {symbol}: {e}")
        return []

def calc_market_trend(closes):
    if len(closes) < 50:
        return 0.0
    ema20 = get_last_valid_value(calc_ema(closes, 20))
    ema50 = get_last_valid_value(calc_ema(closes, 50))
    if ema20 is None or ema50 is None:
        return 0.0
    if ema20 > ema50 and closes[-1] > ema20:
        return 10.0
    if ema20 > ema50:
        return 5.0
    if ema20 < ema50 and closes[-1] < ema20:
        return -10.0
    if ema20 < ema50:
        return -5.0
    return 0.0

# ----------------- Regime persistence -----------------
def load_regime_state():
    try:
        if os.path.exists(REGIME_STATE_FILE):
            with open(REGIME_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"regime": None, "ts": None}

def save_regime_state(state):
    try:
        with open(REGIME_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass

# ----------------- ANALYZE DATA (10B) -----------------
def analyze_data(symbol, data5m, market_trend):
    """
    Returns a JSON-serializable dict for the signal, or None if Neutral (we don't record neutrals).
    """
    # basic guards
    if not data5m or len(data5m) < 50:
        return None
    try:
        current_price = float(data5m[-1].get("close", 0.0))
        if current_price <= 0:
            return None
    except Exception:
        return None

    closes = [float(d["close"]) for d in data5m]
    highs = [float(d["high"]) for d in data5m]
    lows  = [float(d["low"]) for d in data5m]
    volumes = [float(d.get("volume", 0) or 0) for d in data5m]

    # safe calls to indicator funcs
    def safe_call(fn, *args, **kwargs):
        try:
            if callable(fn):
                return fn(*args, **kwargs)
        except Exception:
            return None

    latest_rsi = safe_call(get_last_valid_value, safe_call(calc_rsi, closes, 14))
    latest_rsi = float(latest_rsi) if latest_rsi is not None else None

    macd_obj = safe_call(calc_macd, closes, 12, 26, 9) or {}
    macd_hist = macd_obj.get("histogram", 0.0)
    try:
        macd_hist = float(macd_hist)
    except Exception:
        macd_hist = 0.0

    boll = safe_call(calc_bollinger, closes, 20, 2) or {}
    boll_upper = boll.get("upper"); boll_lower = boll.get("lower"); boll_mid = boll.get("middle"); percent_b = boll.get("percent_b")

    atr = safe_call(calc_atr, highs, lows, closes, 14)
    try:
        atr = float(atr) if atr is not None and not (isinstance(atr, float) and math.isnan(atr)) else None
    except Exception:
        atr = None

    latest_cci = safe_call(calc_cci, highs, lows, closes, 20)
    latest_cci = float(latest_cci) if latest_cci is not None else None

    vol_profile = safe_call(calc_vol_profile, closes, highs, lows, volumes) or {"bullish_score": 0.0, "bearish_score": 0.0}
    vp_bull = float(vol_profile.get("bullish_score", 0) or 0)
    vp_bear = float(vol_profile.get("bearish_score", 0) or 0)

    latest_ema50 = safe_call(get_last_valid_value, safe_call(calc_ema, closes, 50))
    latest_ema50 = float(latest_ema50) if latest_ema50 is not None else None

    # optional indicators
    tsi_val = safe_call(calc_tsi, closes)
    stc_val = safe_call(calc_stc, closes)
    hma_series = safe_call(calc_hma, closes, 16)
    hma_val = float(hma_series[-1]) if isinstance(hma_series, (list, tuple)) and hma_series[-1] is not None else None
    # compute hma slope (very simple last - prev)
    hma_slope = None
    hma_slope_pct = None
    try:
        if isinstance(hma_series, (list, tuple)) and len(hma_series) >= 3:
            a = hma_series[-3]; b = hma_series[-2]; c = hma_series[-1]
            if a is not None and b is not None and c is not None:
                hma_slope = (c - b)
                if hma_val and hma_val != 0:
                    hma_slope_pct = (hma_slope / hma_val) * 100.0
    except Exception:
        hma_slope = None
        hma_slope_pct = None

    alma_series = safe_call(calc_alma, closes, 9, 0.85, 6)
    alma_val = float(alma_series[-1]) if isinstance(alma_series, (list, tuple)) and alma_series[-1] is not None else None

    cvd_obj = safe_call(calc_cvd, closes, volumes) or {}
    cvd_trend = cvd_obj.get("trend") if isinstance(cvd_obj, dict) else None
    cvd_value = float(cvd_obj.get("value")) if isinstance(cvd_obj, dict) and cvd_obj.get("value") is not None else None

    # --- TP/SL selection ---
    if FORCE_FIXED_TP_SL:
        tp_pct = TP_PRICE_MOVE_PCT
        sl_pct = SL_PRICE_MOVE_PCT
    else:
        effective_atr = atr if (atr and atr > 0) else (current_price * 0.002)
        tp_pct_from_atr = (effective_atr * TP_ATR_FACTOR) / current_price if current_price else TP_PCT_MIN
        tp_pct = max(TP_PCT_MIN, min(TP_PCT_MAX, tp_pct_from_atr))
        tp_pct = max(tp_pct, TP_PRICE_MOVE_PCT)
        sl_pct_from_atr = (effective_atr * SL_ATR_FACTOR) / current_price if current_price else SL_PCT_MIN
        sl_pct = max(SL_PCT_MIN, min(SL_PCT_MAX, sl_pct_from_atr))

    tp_buy = current_price + (tp_pct * current_price)
    sl_buy = current_price - (sl_pct * current_price)
    tp_sell = current_price - (tp_pct * current_price)
    sl_sell = current_price + (sl_pct * current_price)

    # --- Scoring engines (Buy / Sell separate) ---
    def buy_base_score():
        s = 0.0
        if percent_b is not None:
            if percent_b <= 0.0:
                s += 20.0
            elif percent_b <= 0.2:
                s += 12.0
            elif percent_b <= 0.4:
                s += 6.0
        if latest_rsi is not None:
            if latest_rsi <= 30:
                s += 12.0
            elif latest_rsi <= 40:
                s += 6.0
        if latest_cci is not None:
            if latest_cci >= 100:
                s += 8.0
            elif latest_cci >= 60:
                s += 4.0
        try:
            if hma_val is not None and alma_val is not None:
                if latest_ema50 is not None and current_price > latest_ema50:
                    s += 6.0
        except Exception:
            pass
        return min(40.0, s)

    def sell_base_score():
        s = 0.0
        if percent_b is not None:
            if percent_b >= 1.0:
                s += 20.0
            elif percent_b >= 0.8:
                s += 12.0
            elif percent_b >= 0.6:
                s += 6.0
        if latest_rsi is not None:
            if latest_rsi >= 70:
                s += 12.0
            elif latest_rsi >= 60:
                s += 6.0
        if latest_cci is not None:
            if latest_cci <= -100:
                s += 8.0
            elif latest_cci <= -60:
                s += 4.0
        try:
            if hma_val is not None and alma_val is not None:
                if latest_ema50 is not None and current_price < latest_ema50:
                    s += 6.0
        except Exception:
            pass
        return min(40.0, s)

    def buy_momentum_score():
        s = 0.0
        effective_atr = atr if atr and atr > 0 else current_price * 0.002
        hist_norm = min(1.5, abs(macd_hist) / (0.5 * effective_atr)) if effective_atr else min(1.5, abs(macd_hist))
        if macd_hist > 0:
            s += 12.0 * min(1.0, hist_norm)
        if isinstance(tsi_val, (int, float)) and tsi_val > 0:
            s += 4.0
        if isinstance(stc_val, (int, float)) and stc_val > 50:
            s += 4.0
        try:
            last = data5m[-1]; prev = data5m[-2]
            if float(last["close"]) > float(last["open"]) and float(last["close"]) > float(prev["high"]):
                s += 5.0
        except Exception:
            pass
        return min(20.0, s)

    def sell_momentum_score():
        s = 0.0
        effective_atr = atr if atr and atr > 0 else current_price * 0.002
        hist_norm = min(1.5, abs(macd_hist) / (0.5 * effective_atr)) if effective_atr else min(1.5, abs(macd_hist))
        if macd_hist < 0:
            s += 12.0 * min(1.0, hist_norm)
        if isinstance(tsi_val, (int, float)) and tsi_val < 0:
            s += 4.0
        if isinstance(stc_val, (int, float)) and stc_val < 50:
            s += 4.0
        try:
            last = data5m[-1]; prev = data5m[-2]
            if float(last["close"]) < float(last["open"]) and float(last["close"]) < float(prev["low"]):
                s += 5.0
        except Exception:
            pass
        return min(20.0, s)

    def buy_trend_score():
        s = 0.0
        try:
            mt = float(market_trend)
            if mt >= 5:
                s += 8.0
        except Exception:
            pass
        try:
            ema_series = safe_call(calc_ema, closes, 50)
            if isinstance(ema_series, (list, tuple)) and len(ema_series) >= 2:
                if ema_series[-1] is not None and ema_series[-2] is not None and ema_series[-1] > ema_series[-2] and current_price > ema_series[-1]:
                    s += 7.0
        except Exception:
            pass
        return min(15.0, s)

    def sell_trend_score():
        s = 0.0
        try:
            mt = float(market_trend)
            if mt <= -5:
                s += 8.0
        except Exception:
            pass
        try:
            ema_series = safe_call(calc_ema, closes, 50)
            if isinstance(ema_series, (list, tuple)) and len(ema_series) >= 2:
                if ema_series[-1] is not None and ema_series[-2] is not None and ema_series[-1] < ema_series[-2] and current_price < ema_series[-1]:
                    s += 7.0
        except Exception:
            pass
        return min(15.0, s)

    def buy_volume_score():
        s = 0.0
        if vp_bull > 0:
            s += 5.0
        try:
            if cvd_trend and "fall" in str(cvd_trend).lower():
                s += 5.0
        except Exception:
            pass
        return min(10.0, s)

    def sell_volume_score():
        s = 0.0
        if vp_bear > 0:
            s += 5.0
        try:
            if cvd_trend and "rise" in str(cvd_trend).lower():
                s += 5.0
        except Exception:
            pass
        return min(10.0, s)

    # confluence bonuses (additive, cap applied later)
    confluence_flags = []
    confluence_points = 0.0

    # buy confluences
    if latest_rsi is not None and latest_rsi <= 30:
        if cvd_trend and "fall" in str(cvd_trend).lower():
            confluence_flags.append("RSI<=30 & CVD_falling"); confluence_points += 12.0
        if (hma_val is not None and alma_val is not None) and (latest_ema50 is not None and current_price > latest_ema50):
            confluence_flags.append("RSI<=30 & HMA_up & ALMA_up"); confluence_points += 16.0
    if cvd_trend and ("fall" in str(cvd_trend).lower()) and (hma_val is not None and alma_val is not None):
        confluence_flags.append("CVD_falling & HMA_up"); confluence_points += 10.0

    # sell confluences
    if latest_rsi is not None and latest_rsi >= 70:
        if cvd_trend and "rise" in str(cvd_trend).lower():
            confluence_flags.append("RSI>=70 & CVD_rising"); confluence_points += 12.0
        if (hma_val is not None and alma_val is not None) and (latest_ema50 is not None and current_price < latest_ema50):
            confluence_flags.append("RSI>=70 & HMA_down & ALMA_down"); confluence_points += 16.0
    if cvd_trend and ("rise" in str(cvd_trend).lower()) and (hma_val is not None and alma_val is not None):
        confluence_flags.append("CVD_rising & HMA_down"); confluence_points += 10.0

    confluence_points = min(30.0, float(confluence_points))

    # compute components
    base_buy = buy_base_score()
    base_sell = sell_base_score()
    mom_buy = buy_momentum_score()
    mom_sell = sell_momentum_score()
    tr_buy = buy_trend_score()
    tr_sell = sell_trend_score()
    vol_buy = buy_volume_score()
    vol_sell = sell_volume_score()
    conf_points = confluence_points

    subtotal_buy = base_buy + mom_buy + tr_buy + vol_buy
    subtotal_sell = base_sell + mom_sell + tr_sell + vol_sell

    initial_signal = "Neutral"
    if subtotal_buy > subtotal_sell and subtotal_buy > 0:
        initial_signal = "Buy"
    elif subtotal_sell > subtotal_buy and subtotal_sell > 0:
        initial_signal = "Sell"
    else:
        initial_signal = "Neutral"

    # profit check
    if initial_signal == "Buy":
        tp = tp_buy; sl = sl_buy
    elif initial_signal == "Sell":
        tp = tp_sell; sl = sl_sell
    else:
        tp = tp_buy; sl = sl_buy

    # estimated margin percent
    price_move_pct = abs((tp - current_price) / current_price) if current_price else 0.0
    estimated_profit_margin_pct = float(price_move_pct * LEVERAGE_FOR_PROFIT_EVAL * 100.0)  # e.g. 5.04%

    profit_comp = 0.0
    if estimated_profit_margin_pct >= 2.0:
        profit_comp = 3.0
    if estimated_profit_margin_pct >= 4.0:
        profit_comp = 5.0

    # compute raw confidence (additive, no veto)
    if initial_signal == "Buy":
        components = {"base_score": base_buy, "momentum_score": mom_buy, "trend_score": tr_buy, "volume_score": vol_buy, "confluence_bonus": conf_points, "profit_check": profit_comp}
        confidence_raw = base_buy + mom_buy + tr_buy + vol_buy + conf_points + profit_comp
    elif initial_signal == "Sell":
        components = {"base_score": base_sell, "momentum_score": mom_sell, "trend_score": tr_sell, "volume_score": vol_sell, "confluence_bonus": conf_points, "profit_check": profit_comp}
        confidence_raw = base_sell + mom_sell + tr_sell + vol_sell + conf_points + profit_comp
    else:
        components = {"base_score": 0.0, "momentum_score": 0.0, "trend_score": 0.0, "volume_score": 0.0, "confluence_bonus": 0.0, "profit_check": 0.0}
        confidence_raw = 0.0

    confidence = int(max(0, min(100, round(float(confidence_raw)))))

    # HMA Gatekeeper: if enabled and signal is close to STRONG (but not strong), allow gatekeeper to push some to strong
    hma_gate_applied = False
    hma_gate_info = {}
    if HMA_GATEKEEPER_ENABLED and initial_signal in ("Buy", "Sell"):
        if confidence < STRONG_THRESHOLD and confidence >= (STRONG_THRESHOLD - HMA_GAP_THRESHOLD):
            aligned = False
            try:
                if initial_signal == "Buy":
                    aligned = (hma_slope is not None and hma_slope > 0)
                elif initial_signal == "Sell":
                    aligned = (hma_slope is not None and hma_slope < 0)
            except Exception:
                aligned = False
            if aligned:
                before_conf = confidence
                confidence = int(min(100, confidence + HMA_GATEKEEPER_BOOST))
                hma_gate_applied = True
                hma_gate_info = {"applied": True, "before_confidence": before_conf, "after_confidence": confidence, "hma_slope": float(hma_slope) if hma_slope is not None else None, "hma_slope_pct": float(hma_slope_pct) if hma_slope_pct is not None else None}
            else:
                hma_gate_info = {"applied": False, "reason": "hma_not_aligned", "hma_slope": float(hma_slope) if hma_slope is not None else None, "hma_slope_pct": float(hma_slope_pct) if hma_slope_pct is not None else None}
        else:
            hma_gate_info = {"applied": False, "reason": "not_in_gap_or_already_strong", "confidence": confidence}

    # Supervisor Promoter (Option 3): apply only if HMA gatekeeper was applied and signal still not strong
    supervisor_applied = False
    supervisor_info = {}
    if SUPERVISOR_ENABLED and hma_gate_applied and confidence < STRONG_THRESHOLD:
        # compute hist_norm
        try:
            effective_atr_for_norm = atr if atr and atr > 0 else (current_price * 0.002)
            hist_norm = min(1.5, abs(macd_hist) / (0.5 * effective_atr_for_norm)) if effective_atr_for_norm else min(1.5, abs(macd_hist))
        except Exception:
            hist_norm = 0.0
        # determine cvd_support: for Buy -> cvd falling supports buy; for Sell -> cvd rising supports sell
        cvd_support = False
        try:
            if initial_signal == "Buy" and cvd_trend and "fall" in str(cvd_trend).lower():
                cvd_support = True
            if initial_signal == "Sell" and cvd_trend and "rise" in str(cvd_trend).lower():
                cvd_support = True
        except Exception:
            cvd_support = False
        # compute hma_slope_pct (already attempted above); ensure numeric
        hma_slope_pct_val = float(hma_slope_pct) if hma_slope_pct is not None else 0.0
        # apply Option 3 condition
        try:
            cond_hist = (hist_norm >= SUP_HIST_NORM_THRESH)
            cond_hma = (hma_slope_pct_val >= SUP_HMA_SLOPE_PCT_THRESH)
            cond = (cond_hist and (cond_hma or cvd_support))
            if cond:
                before_conf = confidence
                confidence = int(min(100, confidence + SUPERVISOR_BOOST))
                supervisor_applied = True
                supervisor_info = {
                    "applied": True,
                    "before_confidence": before_conf,
                    "after_confidence": confidence,
                    "hist_norm": float(hist_norm),
                    "hma_slope_pct": float(hma_slope_pct_val),
                    "cvd_support": bool(cvd_support),
                    "rule": "hist_norm>=0.6 AND (hma_slope_pct>=0.15 OR cvd_support)"
                }
            else:
                supervisor_info = {
                    "applied": False,
                    "hist_norm": float(hist_norm),
                    "hma_slope_pct": float(hma_slope_pct_val),
                    "cvd_support": bool(cvd_support),
                    "reason": "conditions_not_met"
                }
        except Exception as e:
            supervisor_info = {"applied": False, "error": str(e)}

    # final label
    if initial_signal == "Buy":
        if confidence >= STRONG_THRESHOLD:
            final_signal = "Strong Buy"
        elif confidence >= MIN_CONF_FOR_SIGNAL:
            final_signal = "Buy"
        else:
            final_signal = "Neutral"
    elif initial_signal == "Sell":
        if confidence >= STRONG_THRESHOLD:
            final_signal = "Strong Sell"
        elif confidence >= MIN_CONF_FOR_SIGNAL:
            final_signal = "Sell"
        else:
            final_signal = "Neutral"
    else:
        final_signal = "Neutral"

    # prepare would_be_strong_if diagnostics
    if final_signal.startswith("Strong"):
        would_be = {"if_confidence_needed": STRONG_THRESHOLD, "missing_points": 0, "top_missing_components": []}
    else:
        missing_points = max(0, int(max(0, STRONG_THRESHOLD - confidence)))
        caps = {"base_score": 40.0, "momentum_score": 20.0, "trend_score": 15.0, "volume_score": 10.0, "confluence_bonus": 30.0, "profit_check": 5.0}
        comp_gaps = []
        for k, v in components.items():
            gap = max(0.0, caps.get(k, 0.0) - float(v))
            comp_gaps.append({"component": k, "gap": round(gap, 2)})
        comp_gaps = sorted(comp_gaps, key=lambda x: x["gap"], reverse=True)
        top_missing_components = comp_gaps[:3]
        would_be = {"if_confidence_needed": STRONG_THRESHOLD, "missing_points": missing_points, "top_missing_components": top_missing_components}

    # build canonical indicator_scores (simple numeric fields for backtests)
    indicator_scores = {
        "percent_b": float(percent_b) if percent_b is not None else None,
        "rsi": float(latest_rsi) if latest_rsi is not None else None,
        "cci": float(latest_cci) if latest_cci is not None else None,
        "macd_hist": float(macd_hist),
        "atr": float(atr) if atr is not None else None,
        "hma": float(hma_val) if hma_val is not None else None,
        "alma": float(alma_val) if alma_val is not None else None,
        "tsi": float(tsi_val) if isinstance(tsi_val, (int, float)) else None,
        "stc": float(stc_val) if isinstance(stc_val, (int, float)) else None,
        "volProfile_bull": float(vp_bull),
        "volProfile_bear": float(vp_bear),
        "cvd_trend": cvd_trend,
        "cvd_value": float(cvd_value) if cvd_value is not None else None,
        "hma_slope": float(hma_slope) if hma_slope is not None else None,
        "hma_slope_pct": float(hma_slope_pct) if hma_slope_pct is not None else None,
        "estimated_profit_margin_pct": float(estimated_profit_margin_pct)
    }

    # Compose analysis_log (JSON-safe)
    analysis_log = {
        "engine": str(initial_signal),
        "confidence": int(confidence),
        "components": {k: float(v) for k, v in components.items()},
        "indicator_scores": indicator_scores,
        "confluence_flags": list(confluence_flags),
        "would_be_strong_if": would_be,
        "regime_bias": float(market_trend) if market_trend is not None else None,
        "hma_gatekeeper": hma_gate_info,
        "supervisor": supervisor_info
    }

    # Final result object
    result = {
        "coin": symbol,
        "price": round(float(current_price), 8),
        "tp": round(float(tp), 8),
        "sl": round(float(sl), 8),
        "leverage": f"{int(LEVERAGE_FOR_PROFIT_EVAL)}x",
        "confidence": int(confidence),
        "signal": final_signal,
        "estimated_profit": f"{estimated_profit_margin_pct:.2f}%",
        "regime": ("Bullish" if (market_trend is not None and float(market_trend) >= 5) else ("Bearish" if (market_trend is not None and float(market_trend) <= -5) else "Neutral")),
        "signal_time_utc": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "signal_time_ist": datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(pytz.timezone("Asia/Kolkata")).isoformat(),
        "analysis_log": analysis_log,
        "indicators": {
            "rsi5m": (float(latest_rsi) if latest_rsi is not None else None),
            "macd5m": {"macd": float(macd_obj.get("macd")) if macd_obj.get("macd") is not None else None,
                      "signal": float(macd_obj.get("signal")) if macd_obj.get("signal") is not None else None,
                      "histogram": float(macd_obj.get("histogram")) if macd_obj.get("histogram") is not None else None},
            "boll5m": {"upper": (float(boll_upper) if boll_upper is not None else None),
                       "middle": (float(boll_mid) if boll_mid is not None else None),
                       "lower": (float(boll_lower) if boll_lower is not None else None),
                       "percent_b": (float(percent_b) if percent_b is not None else None)},
            "cci5m": (float(latest_cci) if latest_cci is not None else None),
            "marketTrend": float(market_trend) if market_trend is not None else None,
            "volProfile": {"bullish_score": float(vp_bull), "bearish_score": float(vp_bear)},
            "ema50_5m": (float(latest_ema50) if latest_ema50 is not None else None),
            "atr5m": (float(atr) if atr is not None else None),
            "hma5m": (float(hma_val) if hma_val is not None else None),
            "hma_slope": (float(hma_slope) if hma_slope is not None else None),
            "hma_slope_pct": (float(hma_slope_pct) if hma_slope_pct is not None else None),
            "alma5m": (float(alma_val) if alma_val is not None else None),
            "tsi5m": (float(tsi_val) if isinstance(tsi_val, (int, float)) else None),
            "stc5m": (float(stc_val) if isinstance(stc_val, (int, float)) else None),
            "cvd5m": (cvd_obj if isinstance(cvd_obj, dict) else None)
        }
    }

    # Drop neutrals (per preference)
    if final_signal == "Neutral":
        return None

    return result

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    print("[INFO] Starting automated data fetch...")
    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("[ERROR] Could not fetch top coins. Exiting.")
        raise SystemExit(1)

    print(f"[INFO] Found {len(top_coins)} coins to analyze.")

    # determine BTC market trend using BTCUSDT 5m closes
    btc_data = fetch_binance_data("BTCUSDT")
    if not btc_data:
        print("[WARN] Could not fetch BTC data for market trend. Defaulting to 0.")
        market_trend = 0.0
    else:
        market_trend = calc_market_trend([d["close"] for d in btc_data])

    # Region hysteresis: simple persistence
    state = load_regime_state()
    prev_regime = state.get("regime")
    prev_ts = state.get("ts")
    regime_name = ("Bullish" if (market_trend is not None and float(market_trend) >= 5) else ("Bearish" if (market_trend is not None and float(market_trend) <= -5) else "Neutral"))
    save_regime_state({"regime": regime_name, "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()})

    print(f"[INFO] Market Trend: {market_trend} ({regime_name})")

    all_results = []
    for coin in top_coins:
        print(f" - Analyzing {coin} ...")
        time.sleep(0.18)  # polite spacing
        data_5m = fetch_binance_data(coin)
        if not data_5m:
            continue
        try:
            res = analyze_data(coin, data_5m, market_trend)
            if res:
                all_results.append(res)
        except
