# run_automation.py (V12 – 6th Amendment: Regime-aware Dual Engine + IST timestamps)
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

# --- General Configuration ---
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"
TOP_COINS_LIMIT = 70

# =========================
# Utilities & Indicators
# =========================
def calc_ema(values, period):
    if not isinstance(values, list) or len(values) < period:
        return [None] * len(values)
    return pd.Series(values).ewm(span=period, adjust=False).mean().tolist()

def calc_rsi(values, period=14):
    if not isinstance(values, list) or len(values) < period + 1:
        return [None] * len(values)
    series = pd.Series(values)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
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
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {"macd": macd_line.iloc[-1], "signal": signal_line.iloc[-1], "histogram": histogram.iloc[-1]}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period:
        return {"upper": None, "middle": None, "lower": None}
    series = pd.Series(values)
    mean = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    return {"upper": mean + mult * std, "middle": mean, "lower": mean - mult * std}

def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period:
        return None
    tp_series = pd.Series([(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)])
    mean = tp_series.rolling(window=period).mean().iloc[-1]
    mean_dev = tp_series.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False).iloc[-1]
    if mean_dev == 0:
        return 0
    return (tp_series.iloc[-1] - mean) / (0.015 * mean_dev)

def calc_vol_profile(closes, highs, lows, volumes):
    try:
        df = pd.DataFrame({"price": closes, "volume": volumes})
        price_range = max(highs) - min(lows)
        if price_range == 0:
            return {"bullish_score": 0, "bearish_score": 0}
        poc = df.groupby(pd.cut(df["price"], bins=10), observed=False)["volume"].sum().idxmax().mid
        current_price = closes[-1]
        if current_price > poc:
            return {"bullish_score": 3, "bearish_score": 0}
        if current_price < poc:
            return {"bullish_score": 0, "bearish_score": 3}
        return {"bullish_score": 5, "bearish_score": 5}
    except Exception:
        return {"bullish_score": 1, "bearish_score": 1}

def calc_adx(highs, lows, closes, period=14):
    """Basic ADX(14). Returns list same length as inputs."""
    if len(highs) < period + 1:
        return [None] * len(highs)
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["+DM"] = (df["high"] - df["prev_high"]).clip(lower=0.0)
    df["-DM"] = (df["prev_low"] - df["low"]).clip(lower=0.0)
    df["+DM"] = df["+DM"].where(df["+DM"] > df["-DM"], 0.0)
    df["-DM"] = df["-DM"].where(df["-DM"] > df["+DM"], 0.0)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["prev_close"]).abs()
    tr3 = (df["low"] - df["prev_close"]).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = df["TR"].ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (df["+DM"].ewm(alpha=1 / period, adjust=False).mean() / atr)
    minus_di = 100 * (df["-DM"].ewm(alpha=1 / period, adjust=False).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([float("inf"), -float("inf")], 0.0) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx.tolist()

# =========================
# Data fetching
# =========================
def fetch_top_volume_coins(limit=TOP_COINS_LIMIT):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        usdt_pairs = [t for t in data if "symbol" in t and t["symbol"].endswith("USDT")]
        return [c["symbol"] for c in sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)[:limit]]
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def fetch_binance_data(symbol, timeframe="5m", limit=300):
    """Fetches and formats kline data from Binance Futures."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        resp = requests.get(url, proxies=proxies, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [
            {"open": float(d[1]), "high": float(d[2]), "low": float(d[3]), "close": float(d[4]), "volume": float(d[5])}
            for d in data
        ]
    except Exception as e:
        print(f"  - Could not fetch data for {symbol} ({timeframe}): {e}")
        return []

# =========================
# 6th Amendment: Regime API
# =========================
def _safe(val, cast=float, default=None):
    try:
        return cast(val)
    except Exception:
        return default

def _ema_rel(a, b):
    if a is None or b is None:
        return 0
    return 1 if a > b else (-1 if a < b else 0)

def detect_regime(btc_5m, btc_15m, btc_1h, btc_4h, prev_regime=None, prev_score=None):
    """
    Decide current market regime from BTC data (with hysteresis).
    Returns dict: {"regime": "Bullish"/"Bearish"/"Neutral", "regime_score": int, "adx15m": float or None}
    """
    for arr in (btc_5m, btc_15m, btc_1h, btc_4h):
        if not arr or len(arr) < 60:
            return {"regime": "Neutral", "regime_score": 0, "adx15m": None}

    c5 = [d["close"] for d in btc_5m]
    c15 = [d["close"] for d in btc_15m]
    c1h = [d["close"] for d in btc_1h]
    c4h = [d["close"] for d in btc_4h]
    h15 = [d["high"] for d in btc_15m]
    l15 = [d["low"] for d in btc_15m]

    ema20_1h = get_last_valid_value(calc_ema(c1h, 20))
    ema50_1h = get_last_valid_value(calc_ema(c1h, 50))
    ema200_1h = get_last_valid_value(calc_ema(c1h, 200))
    rsi1h = get_last_valid_value(calc_rsi(c1h, 14))
    macd1h = calc_macd(c1h, 12, 26, 9); macd1h_hist = _safe(macd1h.get("histogram")) if isinstance(macd1h, dict) else None
    ema20_4h = get_last_valid_value(calc_ema(c4h, 20))
    ema50_4h = get_last_valid_value(calc_ema(c4h, 50))

    try:
        adx15 = get_last_valid_value(calc_adx(h15, l15, c15, 14))
    except Exception:
        adx15 = None

    ema20_5m = get_last_valid_value(calc_ema(c5, 20))
    ema50_5m = get_last_valid_value(calc_ema(c5, 50))

    score = 0
    # 1h EMA stack
    if _ema_rel(ema20_1h, ema50_1h) == 1 and _ema_rel(ema50_1h, ema200_1h) == 1:
        score += 2
    elif _ema_rel(ema20_1h, ema50_1h) == -1 and _ema_rel(ema50_1h, ema200_1h) == -1:
        score -= 2
    # 1h RSI
    if rsi1h is not None:
        if rsi1h > 55: score += 1
        elif rsi1h < 45: score -= 1
    # 1h MACD
    if macd1h_hist is not None:
        score += 1 if macd1h_hist > 0 else -1
    # 4h bias
    if _ema_rel(ema20_4h, ema50_4h) == 1: score += 1
    elif _ema_rel(ema20_4h, ema50_4h) == -1: score -= 1
    # 5m micro timing penalty/bonus
    micro = _ema_rel(ema20_5m, ema50_5m)
    score += (-1 if micro == -1 else (1 if micro == 1 else 0))

    scale = 1.2 if (adx15 is not None and adx15 >= 20) else 0.9
    raw_score = int(score)

    if raw_score >= 3:
        regime = "Bullish"
    elif raw_score <= -3:
        regime = "Bearish"
    else:
        if prev_regime in ("Bullish", "Bearish") and prev_score is not None:
            if prev_regime == "Bullish" and prev_score >= 3 and raw_score >= 1:
                regime = "Bullish"
            elif prev_regime == "Bearish" and prev_score <= -3 and raw_score <= -1:
                regime = "Bearish"
            else:
                regime = "Neutral"
        else:
            regime = "Neutral"

    return {"regime": regime, "regime_score": int(round(raw_score * scale)), "adx15m": _safe(adx15, float)}

# ===========================================
# 6th Amendment: Regime-aware analyze_data()
# ===========================================
def analyze_data(symbol, data5m, market_ctx):
    """
    Regime-aware dual-engine analyzer.
    - market_ctx = {"regime": ..., "regime_score": ..., "adx15m": float or None}
    """
    if not data5m or len(data5m) < 50:
        return None

    regime = (market_ctx or {}).get("regime", "Neutral")
    regime_score = (market_ctx or {}).get("regime_score", 0) or 0
    adx15m = (market_ctx or {}).get("adx15m", None)

    price = data5m[-1].get("close")
    if price is None:
        return None

    closes = [d["close"] for d in data5m]
    highs = [d["high"] for d in data5m]
    lows = [d["low"] for d in data5m]
    volumes = [d["volume"] for d in data5m]

    # Indicators (5m)
    rsi = get_last_valid_value(calc_rsi(closes, 14))
    macd = calc_macd(closes, 12, 26, 9); macd_hist = macd.get("histogram") if isinstance(macd, dict) else None
    boll = calc_bollinger(closes, 20, 2)
    atr = calc_atr(highs, lows, closes, 14)
    cci = calc_cci(highs, lows, closes, 20)
    ema50 = get_last_valid_value(calc_ema(closes, 50))
    volp = calc_vol_profile(closes, highs, lows, volumes)

    if any(v is None for v in [rsi, cci, boll.get("lower"), boll.get("upper")]):
        return None

    lower, upper, middle = boll["lower"], boll["upper"], boll["middle"]
    band_rng = (upper - lower) if (upper - lower) != 0 else price * 1e-9
    pct_b = (price - lower) / band_rng  # <0 below lower, >1 above upper

    # Confluence flags
    bb_touch_buy = price <= lower
    rsi_ext_buy = rsi is not None and rsi <= 30
    cci_ext_buy = cci is not None and cci <= -100
    num_conf_buy = int(bb_touch_buy) + int(rsi_ext_buy) + int(cci_ext_buy)

    bb_touch_sell = price >= upper
    rsi_ext_sell = rsi is not None and rsi >= 70
    cci_ext_sell = cci is not None and cci >= +100
    macd_up = macd_hist is not None and macd_hist > 0
    adx_gate = adx15m is not None and float(adx15m) >= 20.0
    num_conf_sell = int(bb_touch_sell) + int(rsi_ext_sell) + int(macd_up) + int(adx_gate)

    ema_gap_atr = None
    if ema50 is not None and atr and atr > 0:
        ema_gap_atr = (price - ema50) / atr

    # BUY engine
    buy_score = 0
    if bb_touch_buy: buy_score += 35
    if rsi_ext_buy: buy_score += 30
    elif rsi is not None and 30 < rsi <= 40: buy_score += 15
    if cci_ext_buy: buy_score += 15
    if pct_b <= -0.10: buy_score += 15
    elif pct_b <= -0.05: buy_score += 10
    if macd_hist is not None: buy_score += 5 if macd_hist < 0 else -5
    if regime == "Bullish": buy_score += 10
    if ema_gap_atr is not None:
        if -1.5 <= ema_gap_atr <= -0.8: buy_score += 5
        elif ema_gap_atr < -1.5: buy_score -= 5

    # SELL engine
    sell_score = 0
    if bb_touch_sell: sell_score += 25
    if rsi_ext_sell: sell_score += 25
    elif rsi is not None and 60 <= rsi < 70: sell_score += 10
    if cci_ext_sell: sell_score += 10
    if pct_b >= 1.10: sell_score += 15
    elif pct_b >= 1.02: sell_score += 10
    if macd_hist is not None and macd_hist > 0: sell_score += 20
    if regime == "Bearish": sell_score += 10
    if adx_gate: sell_score += 10

    # Regime side gating
    consider_buy = regime in ("Bullish", "Neutral")
    consider_sell = regime in ("Bearish", "Neutral")

    initial_signal = "Neutral"
    if consider_buy and (buy_score > sell_score) and buy_score > 0:
        initial_signal = "Buy"
    if consider_sell and (sell_score > buy_score) and sell_score > 0:
        initial_signal = "Sell"
    if regime == "Bullish" and initial_signal == "Sell":
        initial_signal = "Neutral"
    if regime == "Bearish" and initial_signal == "Buy":
        initial_signal = "Neutral"

    if initial_signal == "Neutral":
        return {
            "coin": symbol,
            "price": round(float(price), 4),
            "signal": "Neutral",
            "confidence": 0,
            "estimated_profit": "0.00%",
            "analysis_log": {
                "buy_score": int(round(buy_score)),
                "sell_score": int(round(sell_score)),
                "initial_signal": "Neutral",
                "num_confluence_met": 0,
                "base_threshold_ok": False,
                "vol_profile_ok": False,
                "min_profit_ok": False,
                "profit_ceiling_ok": False,
                "confidence": 0,
                "bb_touch": bool(bb_touch_buy or bb_touch_sell),
                "rsi_extreme": bool(rsi_ext_buy or rsi_ext_sell),
                "cci_extreme": bool(cci_ext_buy or cci_ext_sell),
                "vetoes_passed": [],
                "overshoot_ok": False,
                "ema_gap_atr": float(ema_gap_atr) if ema_gap_atr is not None else None,
                "is_regime_aligned": False
            },
            "indicators": {
                "rsi5m": float(rsi) if rsi is not None else None,
                "macd_hist5m": float(macd_hist) if macd_hist is not None else None,
                "boll5m": {"upper": float(upper), "lower": float(lower), "middle": float(middle)},
                "cci5m": float(cci) if cci is not None else None,
                "ema50_5m": float(ema50) if ema50 is not None else None,
                "marketRegime": regime,
                "regimeScore": int(regime_score),
                "adx15m": float(adx15m) if adx15m is not None else None,
                "percentB": float(pct_b)
            }
        }

    # Risk model per regime
    aligned = (regime == "Bullish" and initial_signal == "Buy") or (regime == "Bearish" and initial_signal == "Sell")
    tp_factor = 2.2 if aligned else 2.0
    sl_factor = 1.8 if aligned else 2.0
    if initial_signal == "Buy" and regime != "Bullish" and ema_gap_atr is not None and ema_gap_atr < 0:
        tp_factor, sl_factor = 1.7, 2.3

    eff_atr = atr if atr and atr > 0 else price * 0.002
    if initial_signal == "Buy":
        tp = price + eff_atr * tp_factor
        sl = price - eff_atr * sl_factor
    else:
        tp = price - eff_atr * tp_factor
        sl = price + eff_atr * sl_factor

    profit_pct = abs(((tp - price) / price) * 100) if price else 0.0
    min_profit_ok = profit_pct >= 2.0
    profit_ceiling_ok = profit_pct <= 10.0

    vetoes_passed = []
    vol_ok_buy = volp["bullish_score"] > 0
    vol_ok_sell = volp["bearish_score"] > 0

    if initial_signal == "Buy":
        base_ok = buy_score >= 20
        strong_base_ok = buy_score >= 55
        conf_cnt = num_conf_buy
        overshoot_ok = (pct_b <= -0.05) or (rsi is not None and rsi <= 28)
        vol_ok = True if conf_cnt == 3 else vol_ok_buy
        is_aligned = (regime == "Bullish")
        regime_mult = 1.0 if is_aligned else (0.85 if regime == "Neutral" else 0.6)

        S = min(1.0, buy_score / 100.0)
        C = min(1.0, conf_cnt / 3.0)
        checks = 0; V_raw = 0
        for cond, name in [(base_ok, "base_ok"), (vol_ok, "vol_ok"), (min_profit_ok and profit_ceiling_ok, "profit_window")]:
            checks += 1
            if cond: V_raw += 1; vetoes_passed.append(name)
        V = V_raw / max(1, checks)

        conf = 100.0 * (0.40 * S + 0.40 * C + 0.20 * V)
        if pct_b is not None and pct_b < -0.05:
            conf += min(5.0, (abs(pct_b) - 0.05) * 100.0)
        conf *= regime_mult

        strong_threshold = 80 if regime == "Bullish" else (85 if regime == "Neutral" else 999)
        signal = "Buy"
        if strong_base_ok and conf_cnt >= 2 and overshoot_ok and vol_ok and min_profit_ok and profit_ceiling_ok and conf >= strong_threshold:
            signal = "Strong Buy"

    else:
        base_ok = sell_score >= 25
        strong_base_ok = sell_score >= 50
        conf_cnt = num_conf_sell
        overshoot_ok = (pct_b >= 1.02) or (rsi is not None and rsi >= 72)
        vol_ok = True if (conf_cnt >= 3 and adx_gate) else vol_ok_sell
        is_aligned = (regime == "Bearish")
        regime_mult = 1.0 if is_aligned else (0.85 if regime == "Neutral" else 0.6)

        S = min(1.0, sell_score / 100.0)
        C = min(1.0, conf_cnt / 3.0)
        checks = 0; V_raw = 0
        for cond, name in [(base_ok, "base_ok"), (vol_ok, "vol_ok"), (min_profit_ok and profit_ceiling_ok, "profit_window")]:
            checks += 1
            if cond: V_raw += 1; vetoes_passed.append(name)
        V = V_raw / max(1, checks)

        conf = 100.0 * (0.40 * S + 0.40 * C + 0.20 * V)
        if pct_b is not None and pct_b > 1.02:
            conf += min(5.0, (pct_b - 1.02) * 100.0)
        if adx_gate:
            conf += 3.0
        conf *= regime_mult

        strong_threshold = 80 if regime == "Bearish" else (85 if regime == "Neutral" else 999)
        signal = "Sell"
        if strong_base_ok and conf_cnt >= 2 and overshoot_ok and vol_ok and min_profit_ok and profit_ceiling_ok and conf >= strong_threshold:
            signal = "Strong Sell"

    confidence = int(max(0, min(100, round(conf))))
    leverage = 9 if ("Strong" in signal) else (7 if confidence >= 65 else (6 if confidence >= 50 else 5))

    analysis_log = {
        "buy_score": int(round(buy_score)),
        "sell_score": int(round(sell_score)),
        "initial_signal": initial_signal,
        "num_confluence_met": int(conf_cnt),
        "base_threshold_ok": bool(base_ok),
        "vol_profile_ok": bool(vol_ok),
        "min_profit_ok": bool(min_profit_ok),
        "profit_ceiling_ok": bool(profit_ceiling_ok),
        "confidence": int(confidence),
        "bb_touch": bool(bb_touch_buy or bb_touch_sell),
        "rsi_extreme": bool(rsi_ext_buy or rsi_ext_sell),
        "cci_extreme": bool(cci_ext_buy or cci_ext_sell),
        "vetoes_passed": list(vetoes_passed),
        "overshoot_ok": bool(overshoot_ok),
        "ema_gap_atr": float(ema_gap_atr) if ema_gap_atr is not None else None,
        "is_regime_aligned": bool(is_aligned)
    }

    return {
        "coin": symbol,
        "price": round(float(price), 4),
        "tp": round(float(tp), 4),
        "sl": round(float(sl), 4),
        "leverage": f"{leverage}x",
        "confidence": int(confidence),
        "signal": signal,
        "estimated_profit": f"{profit_pct:.2f}%",
        "analysis_log": analysis_log,
        "indicators": {
            "rsi5m": float(rsi) if rsi is not None else None,
            "macd_hist5m": float(macd_hist) if macd_hist is not None else None,
            "boll5m": {"upper": float(upper), "lower": float(lower), "middle": float(middle)},
            "cci5m": float(cci) if cci is not None else None,
            "ema50_5m": float(ema50) if ema50 is not None else None,
            "marketRegime": regime,
            "regimeScore": int(regime_score),
            "adx15m": float(adx15m) if adx15m is not None else None,
            "percentB": float(pct_b)
        }
    }

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    print("Starting automated data fetch (6th Amendment)...")

    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting.")
        exit()

    print(f"Found {len(top_coins)} coins to analyze.")

    # --- BTC context for regime detection ---
    btc_5m = fetch_binance_data("BTCUSDT", "5m", 300)
    btc_15m = fetch_binance_data("BTCUSDT", "15m", 300)
    btc_1h = fetch_binance_data("BTCUSDT", "1h", 300)
    btc_4h = fetch_binance_data("BTCUSDT", "4h", 300)

    market_ctx = detect_regime(btc_5m, btc_15m, btc_1h, btc_4h)
    print(f"Regime: {market_ctx['regime']}  |  Score: {market_ctx['regime_score']}  |  ADX15m: {market_ctx['adx15m']}")

    all_results = []
    for coin in top_coins:
        print(f" - Analyzing {coin}...")
        time.sleep(0.2)

        data_5m = fetch_binance_data(coin, "5m", 300)
        if not data_5m:
            continue

        result = analyze_data(coin, data_5m, market_ctx)
        if result:
            all_results.append(result)

    if all_results:
        strong_signals = [s for s in all_results if "Strong" in s.get("signal", "")]
        print(f"\nAnalysis complete. Found {len(strong_signals)} strong signals.")
        print("Saving full analysis file...")

        utc_now = datetime.now(pytz.utc)
        ist_tz = pytz.timezone("Asia/Kolkata")
        ist_now = utc_now.astimezone(ist_tz)
        timestamp_str = ist_now.strftime("%Y-%m-%d_%H-%M-%S")
        file_suffix = "_STRONG" if strong_signals else ""
        archive_filename = f"signals_{timestamp_str}{file_suffix}.json"

        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        archive_filepath = os.path.join(ARCHIVE_FOLDER, archive_filename)
        with open(archive_filepath, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Archive file saved to {archive_filepath}")

        with open(LIVE_FILENAME, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Live data file saved as {LIVE_FILENAME}")
    else:
        print("\nNo results generated. No file will be saved.")
