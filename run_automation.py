# run_automation.py (8b Amendment – IST Timestamps, Fixed 5% TP/SL on margin + Deserving Strong)
import pandas as pd
import requests
import json
from datetime import datetime
import time
import os
import math
import pytz

# --- PROXY CONFIGURATION ---
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
MIN_PROFIT_MARGIN = 2.0           # min % on margin to pass
# 8b: fixed TP/SL = 5% on margin, 1:1 R/R (≈ 0.714% raw move at 7x)
FIXED_TP_SL_MARGIN = 5.0

# ATR-based clamps from 8th Amendment are superseded by fixed TP/SL in 8b
# (left here for reference)
TP_PCT_MIN, TP_PCT_MAX = 0.008, 0.016
SL_PCT_MIN, SL_PCT_MAX = 0.008, 0.020

# Regime hysteresis (stickiness)
REGIME_HOLD_MINUTES = 60
REGIME_CONFIRM_BARS = 2

# ============== INDICATORS ==============
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
    return (100 - (100 / (1 + rs))).tolist()

def get_last_valid_value(values):
    for value in reversed(values):
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            return value
    return None

def calc_macd(values, fast=12, slow=26, signal=9):
    series = pd.Series(values)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = (macd_line - signal_line)
    return {
        'macd': float(macd_line.iloc[-1]),
        'signal': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1])
    }

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period:
        return {'upper': None, 'middle': None, 'lower': None}
    series = pd.Series(values)
    mean = float(series.rolling(window=period).mean().iloc[-1])
    std = float(series.rolling(window=period).std().iloc[-1])
    return {'upper': mean + (mult * std), 'middle': mean, 'lower': mean - (mult * std)}

def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1])

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period:
        return None
    tp_series = pd.Series([(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)])
    mean = float(tp_series.rolling(window=period).mean().iloc[-1])
    mean_dev = float(tp_series.rolling(window=period).apply(
        lambda x: (x - x.mean()).abs().mean(), raw=False).iloc[-1])
    if mean_dev == 0:
        return 0.0
    return float((tp_series.iloc[-1] - mean) / (0.015 * mean_dev))

def calc_market_trend(closes):
    if len(closes) < 50:
        return 0.0
    ema20_list = calc_ema(closes, 20)
    ema50_list = calc_ema(closes, 50)
    ema20, ema50 = get_last_valid_value(ema20_list), get_last_valid_value(ema50_list)
    if ema20 is None or ema50 is None:
        return 0.0
    price = closes[-1]
    if ema20 > ema50 and price > ema20: return 10.0
    if ema20 > ema50: return 5.0
    if ema20 < ema50 and price < ema20: return -10.0
    if ema20 < ema50: return -5.0
    return 0.0

def calc_vol_profile(closes, highs, lows, volumes):
    try:
        df = pd.DataFrame({'price': closes, 'volume': volumes})
        price_range = max(highs) - min(lows)
        if price_range == 0:
            return {'bullish_score': 0.0, 'bearish_score': 0.0}
        poc = df.groupby(pd.cut(df['price'], bins=10, include_lowest=True, right=True), observed=False)['volume'] \
                .sum().idxmax().mid
        current_price = closes[-1]
        if current_price > poc: return {'bullish_score': 3.0, 'bearish_score': 0.0}
        if current_price < poc: return {'bullish_score': 0.0, 'bearish_score': 3.0}
        return {'bullish_score': 5.0, 'bearish_score': 5.0}
    except Exception:
        return {'bullish_score': 1.0, 'bearish_score': 1.0}

# ============== DATA FETCH ==============
def fetch_top_volume_coins(limit=TOP_LIMIT):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        usdt_pairs = [t for t in data if 'symbol' in t and str(t['symbol']).endswith('USDT')]
        sorted_syms = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0.0)), reverse=True)
        return [c['symbol'] for c in sorted_syms[:limit]]
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def fetch_binance_data(symbol, timeframe='5m', limit=120):
    """Fetches and formats kline data from Binance Futures."""
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [
            {
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5])
            } for d in data
        ]
    except Exception as e:
        print(f"  - Could not fetch data for {symbol}: {e}")
        return []

# ============== ANALYZE (8th Amendment core + 8b TP/SL + Deserving Strong) ==============
def analyze_data(symbol, data5m, market_trend):
    """
    8th Amendment logic with 8b changes:
    - Fixed TP/SL = 5% on margin (7x), 1:1 R/R
    - Add 'deserving_strong' tag for near-strongs (Sell ≥68, Buy ≥75)
    - JSON-safe outputs
    """
    if not data5m or len(data5m) < 60:
        return None

    current_price = data5m[-1].get("close")
    if current_price is None:
        return None

    closes = [d["close"] for d in data5m]
    highs  = [d["high"] for d in data5m]
    lows   = [d["low"] for d in data5m]
    volumes = [d["volume"] for d in data5m]

    # Indicators
    latest_rsi = get_last_valid_value(calc_rsi(closes, 14))
    macd_obj = calc_macd(closes, 12, 26, 9)
    latest_macd_hist = macd_obj.get("histogram") if isinstance(macd_obj, dict) else 0.0
    boll = calc_bollinger(closes, 20, 2)
    atr = calc_atr(highs, lows, closes, 14) or (current_price * 0.002)
    latest_cci = calc_cci(highs, lows, closes, 20)
    latest_ema50 = get_last_valid_value(calc_ema(closes, 50))
    vol_profile = calc_vol_profile(closes, highs, lows, volumes)

    if any(v is None for v in [latest_rsi, latest_cci, boll.get("lower"), boll.get("upper")]):
        return None

    # --- Scoring (kept consistent with 8th Amendment) ---
    buy_score = 0.0
    sell_score = 0.0

    # Bollinger touches
    if current_price <= boll["lower"]: buy_score += 35
    if current_price >= boll["upper"]: sell_score += 35

    # RSI extremes
    if latest_rsi <= 30: buy_score += 30
    elif 30 < latest_rsi <= 40: buy_score += 15
    if latest_rsi >= 70: sell_score += 30
    elif 60 <= latest_rsi < 70: sell_score += 15

    # CCI extremes
    if latest_cci >= 100: buy_score += 15
    if latest_cci <= -100: sell_score += 15

    # MACD soft bias
    if latest_macd_hist < 0: buy_score += 5
    else: sell_score += 5

    # EMA50 side bias
    if current_price > latest_ema50: buy_score += 10
    else: sell_score += 10

    # Market regime bias (from EMA20/50 model)
    if market_trend >= 5: buy_score += 10; sell_score -= 10
    elif market_trend <= -5: sell_score += 10; buy_score -= 10

    # Initial direction (with simple regime gating)
    initial_signal = "Neutral"
    if market_trend >= 5:
        if buy_score > 0: initial_signal = "Buy"
    elif market_trend <= -5:
        if sell_score > 0: initial_signal = "Sell"
    else:
        if buy_score > sell_score and buy_score > 0: initial_signal = "Buy"
        elif sell_score > buy_score and sell_score > 0: initial_signal = "Sell"

    # Confluence & overshoot
    bb_touch_buy = (current_price <= boll["lower"])
    bb_touch_sell = (current_price >= boll["upper"])
    rsi_buy = (latest_rsi <= 30)
    rsi_sell = (latest_rsi >= 70)
    cci_buy = (latest_cci >= 100)
    cci_sell = (latest_cci <= -100)

    num_conf_buy = int(bb_touch_buy) + int(rsi_buy) + int(cci_buy)
    num_conf_sell = int(bb_touch_sell) + int(rsi_sell) + int(cci_sell)

    percentB = (current_price - boll["lower"]) / max(1e-9, (boll["upper"] - boll["lower"]))
    overshoot_buy = (percentB <= 0.05) or rsi_buy or bb_touch_buy
    overshoot_sell = (percentB >= 0.95) or rsi_sell or bb_touch_sell

    # Volume profile OK
    passes_vol_buy = (vol_profile["bullish_score"] > 0)
    passes_vol_sell = (vol_profile["bearish_score"] > 0)

    # Base gates
    base_buy_ok = (buy_score >= 18)
    base_sell_ok = (sell_score >= 18)

    # Confluence gates (neutral stricter)
    if market_trend >= 5:
        conf_buy_ok = (num_conf_buy >= 2); conf_sell_ok = False
    elif market_trend <= -5:
        conf_sell_ok = (num_conf_sell >= 2); conf_buy_ok = False
    else:
        conf_buy_ok = (num_conf_buy >= 3); conf_sell_ok = (num_conf_sell >= 3)

    # --- 8b: Fixed TP/SL 5% on margin, 1:1 ---
    raw_move_frac = (FIXED_TP_SL_MARGIN / LEVERAGE_FOR_PROFIT_EVAL) / 100.0  # ≈ 0.007142857
    if initial_signal == "Buy":
        tp = current_price * (1.0 + raw_move_frac)
        sl = current_price * (1.0 - raw_move_frac)
    elif initial_signal == "Sell":
        tp = current_price * (1.0 - raw_move_frac)
        sl = current_price * (1.0 + raw_move_frac)
    else:
        tp = current_price
        sl = current_price

    estimated_profit_margin_pct = FIXED_TP_SL_MARGIN
    passes_min_profit = (estimated_profit_margin_pct >= MIN_PROFIT_MARGIN)

    # Confidence (8th weights: Base 40 + Conf 40 + Veto 20) – unchanged in 8b
    base_score = buy_score if initial_signal == "Buy" else sell_score if initial_signal == "Sell" else 0.0
    num_conf = num_conf_buy if initial_signal == "Buy" else num_conf_sell if initial_signal == "Sell" else 0
    vol_ok = passes_vol_buy if initial_signal == "Buy" else passes_vol_sell if initial_signal == "Sell" else False
    overshoot_ok = overshoot_buy if initial_signal == "Buy" else overshoot_sell if initial_signal == "Sell" else False

    base_component = max(0.0, min(1.0, base_score / 100.0)) * 0.40
    conf_component = max(0.0, min(1.0, (num_conf / 3.0))) * 0.40
    veto_passes = 0
    if (initial_signal == "Buy" and base_buy_ok) or (initial_signal == "Sell" and base_sell_ok):
        veto_passes += 1
    if vol_ok: veto_passes += 1
    if passes_min_profit: veto_passes += 1
    veto_component = (veto_passes / 3.0) * 0.20
    confidence = int(round((base_component + conf_component + veto_component) * 100.0))
    confidence = max(0, min(100, confidence))

    # RSI-based small boosts (kept modest in 8b)
    rsi_boost = 0
    if initial_signal == "Sell" and rsi_sell and not (latest_macd_hist > 0):
        rsi_boost += 6
        if (percentB >= 0.95 or bb_touch_sell): rsi_boost += 3
    if initial_signal == "Buy" and rsi_buy and not (latest_macd_hist < 0):
        rsi_boost += 6
        if (percentB <= 0.05 or bb_touch_buy): rsi_boost += 3
    confidence = max(0, min(100, confidence + rsi_boost))

    # Final label (8th thresholds)
    # Bearish market prefers Sells, Bullish prefers Buys; Neutral stricter
    if market_trend <= -5:
        strong_sell_thr, strong_buy_thr = 70, 80
    elif market_trend >= 5:
        strong_sell_thr, strong_buy_thr = 75, 78
    else:
        strong_sell_thr, strong_buy_thr = 75, 80

    final_signal = "Neutral"
    if initial_signal == "Buy":
        if base_buy_ok and conf_component > 0 and vol_ok and overshoot_ok and passes_min_profit and confidence >= strong_buy_thr:
            final_signal = "Strong Buy"
        elif confidence >= 40:
            final_signal = "Buy"
        else:
            final_signal = "Neutral"
    elif initial_signal == "Sell":
        if base_sell_ok and conf_component > 0 and vol_ok and overshoot_ok and passes_min_profit and confidence >= strong_sell_thr:
            final_signal = "Strong Sell"
        elif confidence >= 40:
            final_signal = "Sell"
        else:
            final_signal = "Neutral"

    # Leverage suggestion (display only)
    leverage = 7 if final_signal.startswith("Strong") else (6 if confidence >= 50 else 5)
    leverage_str = f"{int(leverage)}x"

    # --- 8b: Deserving Strong tag (analysis-only helper) ---
    deserving_strong = False
    if initial_signal == "Buy" and confidence >= 75:
        deserving_strong = True
    elif initial_signal == "Sell" and confidence >= 68:
        deserving_strong = True

    # Build JSON-safe analysis_log
    analysis_log = {
        "initial_signal": str(initial_signal),
        "buy_score": int(round(buy_score)),
        "sell_score": int(round(sell_score)),
        "num_confluence_met": int(num_conf),
        "vol_profile_ok": bool(vol_ok),
        "overshoot_ok": bool(overshoot_ok),
        "min_profit_ok": bool(passes_min_profit),
        "rsi_conf_boost_points": int(rsi_boost),
        "deserving_strong": bool(deserving_strong),
        "raw_move_pct": round(raw_move_frac * 100.0, 4)  # ~0.7143
    }

    return {
        "coin": symbol,
        "price": round(float(current_price), 6),
        "tp": round(float(tp), 6),
        "sl": round(float(sl), 6),
        "leverage": leverage_str,
        "confidence": int(confidence),
        "signal": final_signal,
        "estimated_profit": f"{estimated_profit_margin_pct:.2f}%",
        "deserving_strong": bool(deserving_strong),
        "analysis_log": analysis_log,
        "indicators": {
            "rsi5m": float(latest_rsi) if latest_rsi is not None else None,
            "macd_hist5m": float(latest_macd_hist) if latest_macd_hist is not None else None,
            "boll5m": {
                "upper": float(boll["upper"]),
                "lower": float(boll["lower"]),
                "middle": float(boll["middle"])
            },
            "cci5m": float(latest_cci) if latest_cci is not None else None,
            "marketTrend": float(market_trend),
            "volProfile": {
                "bullish_score": float(vol_profile["bullish_score"]),
                "bearish_score": float(vol_profile["bearish_score"])
            },
            "ema50_5m": float(latest_ema50) if latest_ema50 is not None else None,
            "percentB": float(percentB)
        }
    }

# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    print("Starting automated data fetch...")

    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting.")
        exit(0)
    print(f"Found {len(top_coins)} coins to analyze.")

    btc_data = fetch_binance_data("BTCUSDT")
    market_trend = calc_market_trend([d["close"] for d in btc_data]) if btc_data else 0.0
    print(f"Market Trend determined: {market_trend}")

    all_results = []
    for coin in top_coins:
        print(f" - Analyzing {coin}...")
        time.sleep(0.2)  # gentle pacing
        data_5m = fetch_binance_data(coin)
        if not data_5m:
            continue
        result = analyze_data(coin, data_5m, market_trend)
        if result:
            all_results.append(result)

    if all_results:
        strong_signals = [s for s in all_results if "Strong" in s.get('signal', '')]
        print(f"\nAnalysis complete. Found {len(strong_signals)} strong signals.")
        print("Saving full analysis file...")

        # **IST Timestamp filenames**
        utc_now = datetime.now(pytz.utc)
        ist_tz = pytz.timezone("Asia/Kolkata")
        ist_now = utc_now.astimezone(ist_tz)
        timestamp_str = ist_now.strftime("%Y-%m-%d_%H-%M-%S")

        file_suffix = "_STRONG" if strong_signals else ""
        archive_filename = f"signals_{timestamp_str}{file_suffix}.json"

        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        archive_filepath = os.path.join(ARCHIVE_FOLDER, archive_filename)

        with open(archive_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Archive file saved to {archive_filepath}")

        with open(LIVE_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Live data file saved as {LIVE_FILENAME}")
    else:
        print("\nNo results generated. No file will be saved.")
