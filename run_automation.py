# run_automation.py — 9th Amendment
# Regime-gated models, Confidence 2.0, fixed TP/SL=3% margin (7x), no Neutral output

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
MIN_PROFIT_MARGIN = 2.0                 # minimum % on margin
FIXED_TP_SL_MARGIN = 3.0                # 9th: TP=SL=3% on margin (~0.4286% raw @7x), 1:1

# Regime hysteresis knobs (placeholder; regime calc uses EMA20/50)
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
            return float(value)
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

def calc_adx(highs, lows, closes, period=14):
    # Lightweight ADX estimator (optional bonus; safe if missing)
    try:
        import numpy as np
        if len(highs) < period + 1:
            return None
        df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
        up = df['high'].diff()
        down = -df['low'].diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([float('inf'), -float('inf')], 0.0) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
        return float(adx)
    except Exception:
        return None

def calc_market_trend(closes):
    # EMA20/EMA50 regime: 10/5/-5/-10
    if len(closes) < 50:
        return 0.0
    ema20_list = calc_ema(closes, 20)
    ema50_list = calc_ema(closes, 50)
    ema20, ema50 = get_last_valid_value(ema20_list), get_last_valid_value(ema50_list)
    if ema20 is None or ema50 is None:
        return 0.0
    price = float(closes[-1])
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

# ============== 9th AMENDMENT CORE HELPERS ==============
def score_buy_model(current_price, boll, rsi, cci, macd_hist, ema50, regime):
    """Return base_score (0-100), confluence_count, overshoot_ok, feature flags."""
    base = 0.0
    bb_touch = current_price <= boll["lower"]
    if bb_touch: base += 35
    rsi_extreme = rsi is not None and rsi <= 30
    if rsi_extreme: base += 30
    elif rsi is not None and rsi <= 40: base += 15
    cci_extreme = cci is not None and cci >= 100
    if cci_extreme: base += 15
    if macd_hist is not None and macd_hist < 0: base += 5
    if ema50 is not None and current_price > ema50: base += 10
    if regime is not None and regime >= 5: base += 10

    num_conf = int(bool(bb_touch)) + int(bool(rsi_extreme)) + int(bool(cci_extreme))
    # %B
    denom = max(1e-9, (boll["upper"] - boll["lower"]))
    percentB = (current_price - boll["lower"]) / denom
    overshoot_ok = (percentB <= 0.05) or bb_touch or rsi_extreme

    return float(base), int(num_conf), bool(overshoot_ok), float(percentB)

def score_sell_model(current_price, boll, rsi, cci, macd_hist, ema50, regime):
    base = 0.0
    bb_touch = current_price >= boll["upper"]
    if bb_touch: base += 35
    rsi_extreme = rsi is not None and rsi >= 70
    if rsi_extreme: base += 30
    elif rsi is not None and rsi >= 60: base += 15
    cci_extreme = cci is not None and cci <= -100
    if cci_extreme: base += 15
    if macd_hist is not None and macd_hist > 0: base += 5
    if ema50 is not None and current_price < ema50: base += 10
    if regime is not None and regime <= -5: base += 10

    num_conf = int(bool(bb_touch)) + int(bool(rsi_extreme)) + int(bool(cci_extreme))
    denom = max(1e-9, (boll["upper"] - boll["lower"]))
    percentB = (current_price - boll["lower"]) / denom
    overshoot_ok = (percentB >= 0.95) or bb_touch or rsi_extreme

    return float(base), int(num_conf), bool(overshoot_ok), float(percentB)

def confidence_v2(base_score, num_conf, regime, overshoot_ok, vol_ok, side, macd_hist, adx):
    """
    Confidence 2.0 (0-100):
      Base 25 + Confluence 35 + Regime 20 + Overshoot 10 + Veto pass 10
      Guarded RSI/overshoot boosts applied outside this function.
    """
    # Base (scale to 0..1)
    base_comp = max(0.0, min(1.0, base_score / 100.0)) * 25.0

    # Confluence depth (0..3 → 0..1)
    conf_comp = max(0.0, min(1.0, num_conf / 3.0)) * 35.0

    # Regime alignment
    reg_comp = 0.0
    if side == "Buy":
        reg_comp = (1.0 if regime is not None and regime >= 5 else 0.5 if regime == 0 else 0.0) * 20.0
    elif side == "Sell":
        reg_comp = (1.0 if regime is not None and regime <= -5 else 0.5 if regime == 0 else 0.0) * 20.0

    # Overshoot quality
    over_comp = (10.0 if overshoot_ok else 0.0)

    # Veto pass
    veto_comp = (10.0 if vol_ok else 0.0)

    conf = base_comp + conf_comp + reg_comp + over_comp + veto_comp

    # Penalties for trend continuation risk
    # (ADX≥25 AND MACD pro-trend) → minus up to 10
    penalty = 0.0
    if adx is not None and adx >= 25:
        if side == "Buy" and macd_hist is not None and macd_hist < 0:
            penalty += 7.0
        if side == "Sell" and macd_hist is not None and macd_hist > 0:
            penalty += 7.0

    conf = max(0.0, min(100.0, conf - penalty))
    return int(round(conf))

# ============== ANALYZE ==============
def analyze_data(symbol, data5m, market_trend):
    """
    9th Amendment:
      - Regime-gated signals (bullish→Buy only, bearish→Sell only, neutral stricter)
      - Separate Buy/Sell models
      - Confidence 2.0
      - Promotion requires Overshoot + (Vol OK or Strong Confluence)
      - Fixed TP/SL = 3% on margin (7x), 1:1 R/R
      - Neutral signals are NOT returned
    """
    if not data5m or len(data5m) < 60:
        return None

    current_price = float(data5m[-1].get("close", 0.0))
    if current_price <= 0:
        return None

    closes = [float(d["close"]) for d in data5m]
    highs  = [float(d["high"])  for d in data5m]
    lows   = [float(d["low"])   for d in data5m]
    volumes = [float(d["volume"]) for d in data5m]

    # Indicators
    rsi = get_last_valid_value(calc_rsi(closes, 14))
    macd = calc_macd(closes, 12, 26, 9)
    macd_hist = macd.get("histogram", 0.0)
    boll = calc_bollinger(closes, 20, 2)
    cci = calc_cci(highs, lows, closes, 20)
    ema50 = get_last_valid_value(calc_ema(closes, 50))
    adx = calc_adx(highs, lows, closes, 14)
    vol_profile = calc_vol_profile(closes, highs, lows, volumes)

    if any(v is None for v in [rsi, cci, boll.get("lower"), boll.get("upper")]):
        return None

    # Determine regime class
    regime = float(market_trend or 0.0)  # -10, -5, 0, +5, +10
    regime_class = "neutral"
    if regime >= 5: regime_class = "bullish"
    elif regime <= -5: regime_class = "bearish"

    # Gate universe by regime
    consider_buy = (regime_class == "bullish") or (regime_class == "neutral")
    consider_sell = (regime_class == "bearish") or (regime_class == "neutral")

    # Volume-profile sides
    vol_ok_buy = bool(vol_profile["bullish_score"] > 0.0)
    vol_ok_sell = bool(vol_profile["bearish_score"] > 0.0)

    # Score both, then pick allowed side with higher base evidence (neutral can choose stronger)
    buy_pack = sell_pack = None

    if consider_buy:
        b_base, b_conf, b_over, b_percentB = score_buy_model(current_price, boll, rsi, cci, macd_hist, ema50, regime)
        buy_pack = {"base": b_base, "conf": b_conf, "over": b_over, "percentB": b_percentB}
    if consider_sell:
        s_base, s_conf, s_over, s_percentB = score_sell_model(current_price, boll, rsi, cci, macd_hist, ema50, regime)
        sell_pack = {"base": s_base, "conf": s_conf, "over": s_over, "percentB": s_percentB}

    initial_signal = "Neutral"
    pack = None
    vol_ok = False
    if regime_class == "bullish":
        if buy_pack and buy_pack["base"] > 0:
            initial_signal = "Buy"
            pack = buy_pack
            vol_ok = vol_ok_buy
    elif regime_class == "bearish":
        if sell_pack and sell_pack["base"] > 0:
            initial_signal = "Sell"
            pack = sell_pack
            vol_ok = vol_ok_sell
    else:
        # neutral: pick stronger side by base score
        if buy_pack and sell_pack:
            if buy_pack["base"] >= sell_pack["base"] and buy_pack["base"] > 0:
                initial_signal, pack, vol_ok = "Buy", buy_pack, vol_ok_buy
            elif sell_pack["base"] > 0:
                initial_signal, pack, vol_ok = "Sell", sell_pack, vol_ok_sell
        elif buy_pack and buy_pack["base"] > 0:
            initial_signal, pack, vol_ok = "Buy", buy_pack, vol_ok_buy
        elif sell_pack and sell_pack["base"] > 0:
            initial_signal, pack, vol_ok = "Sell", sell_pack, vol_ok_sell

    if initial_signal == "Neutral" or not pack:
        return None  # do not output neutral rows

    base_score = float(pack["base"])
    num_conf = int(pack["conf"])
    overshoot_ok = bool(pack["over"])
    percentB = float(pack["percentB"])

    # Strong confluence definition (helps bypass vol_profile when setup is compelling)
    strong_conf_buy  = (num_conf >= 3) or (num_conf >= 2 and percentB <= 0.03)
    strong_conf_sell = (num_conf >= 3) or (num_conf >= 2 and percentB >= 0.97)
    strong_conf = strong_conf_buy if initial_signal == "Buy" else strong_conf_sell

    # Confluence gates per regime
    if regime_class == "bullish":
        confluence_ok = (num_conf >= 2)
    elif regime_class == "bearish":
        confluence_ok = (num_conf >= 2)
    else:
        confluence_ok = (num_conf >= 3)  # neutral stricter

    # Fixed TP/SL (3% margin) → raw move fraction
    raw_move_frac = (FIXED_TP_SL_MARGIN / LEVERAGE_FOR_PROFIT_EVAL) / 100.0  # e.g., 3 / 7 / 100
    if initial_signal == "Buy":
        tp = current_price * (1.0 + raw_move_frac)
        sl = current_price * (1.0 - raw_move_frac)
        vol_ok_side = vol_ok_buy
    else:
        tp = current_price * (1.0 - raw_move_frac)
        sl = current_price * (1.0 + raw_move_frac)
        vol_ok_side = vol_ok_sell

    estimated_profit_margin_pct = FIXED_TP_SL_MARGIN
    min_profit_ok = bool(estimated_profit_margin_pct >= MIN_PROFIT_MARGIN)

    # Confidence 2.0
    conf = confidence_v2(
        base_score=base_score,
        num_conf=num_conf,
        regime=regime,
        overshoot_ok=overshoot_ok,
        vol_ok=vol_ok_side,
        side=initial_signal,
        macd_hist=macd_hist,
        adx=adx
    )

    # RSI overshoot guarded boosts (skip if MACD strongly pro-trend)
    rsi_boost = 0
    if initial_signal == "Sell" and rsi is not None and rsi >= 70 and overshoot_ok and not (macd_hist is not None and macd_hist > 0.1):
        rsi_boost += 6
        if percentB >= 0.98: rsi_boost += 4  # close outside band
    if initial_signal == "Buy" and rsi is not None and rsi <= 30 and overshoot_ok and not (macd_hist is not None and macd_hist < -0.1):
        rsi_boost += 6
        if percentB <= 0.02: rsi_boost += 4
    conf = max(0, min(100, conf + rsi_boost))

    # Promotion thresholds
    if regime_class == "bearish":
        strong_thr = 70  # Strong Sell threshold
    elif regime_class == "bullish":
        strong_thr = 75  # Strong Buy threshold
    else:
        strong_thr = 78  # Neutral stricter

    base_ok = bool(base_score >= 18)
    # Final gates: Overshoot required; and (VolProfile OK OR Strong Confluence)
    final_gate = bool(confluence_ok and overshoot_ok and min_profit_ok and (vol_ok_side or strong_conf) and base_ok)

    # Final label (no Neutral written)
    if final_gate and conf >= strong_thr:
        final_signal = "Strong " + initial_signal
    else:
        # Require a minimal confidence to emit non-strongs
        final_signal = initial_signal if conf >= 40 else None

    if final_signal is None:
        return None  # still do not output ultra-weak signals

    # Deserving Strong tag (threshold - 5)
    deserving_strong = bool(conf >= (strong_thr - 5))

    # Leverage suggestion (display only)
    leverage = 7 if final_signal.startswith("Strong") else (6 if conf >= 50 else 5)
    leverage_str = f"{int(leverage)}x"

    # Build JSON-safe analysis_log/indicators
    analysis_log = {
        "model_side": "buy_model" if initial_signal == "Buy" else "sell_model",
        "regime": float(regime),
        "base_score": int(round(base_score)),
        "num_confluence": int(num_conf),
        "overshoot_ok": bool(overshoot_ok),
        "strong_confluence": bool(strong_conf),
        "vol_profile_ok": bool(vol_ok_side),
        "min_profit_ok": bool(min_profit_ok),
        "confidence": int(conf),
        "rsi_boost": int(rsi_boost),
        "promotion_gate_passed": bool(final_gate),
        "promotion_threshold": int(strong_thr),
        "deserving_strong": bool(deserving_strong),
        "tp_pct_margin": float(FIXED_TP_SL_MARGIN),
        "sl_pct_margin": float(FIXED_TP_SL_MARGIN),
        "raw_move_pct": round(raw_move_frac * 100.0, 4),  # ≈ 0.4286
        "percentB": float(percentB),
    }

    indicators = {
        "rsi5m": float(rsi) if rsi is not None else None,
        "macd_hist5m": float(macd_hist) if macd_hist is not None else None,
        "boll5m": {
            "upper": float(boll["upper"]),
            "lower": float(boll["lower"]),
            "middle": float(boll["middle"])
        },
        "cci5m": float(cci) if cci is not None else None,
        "ema50_5m": float(ema50) if ema50 is not None else None,
        "marketTrend": float(regime),
        "volProfile": {
            "bullish_score": float(vol_profile["bullish_score"]),
            "bearish_score": float(vol_profile["bearish_score"])
        },
        "adx14": float(adx) if adx is not None else None,
        "percentB": float(percentB)
    }

    return {
        "coin": symbol,
        "price": round(float(current_price), 6),
        "tp": round(float(tp), 6),
        "sl": round(float(sl), 6),
        "leverage": leverage_str,
        "confidence": int(conf),
        "signal": final_signal,
        "estimated_profit": f"{estimated_profit_margin_pct:.2f}%",
        "deserving_strong": bool(deserving_strong),
        "analysis_log": analysis_log,
        "indicators": indicators
    }

# ============== MAIN EXECUTION ==============
if __name__ == "__main__":
    print("Starting automated data fetch...")

    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting.")
        raise SystemExit(0)
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

        # IST Timestamp filenames
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
