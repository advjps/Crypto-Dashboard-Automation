# run_automation.py (V11 - IST Timestamps)
import pandas as pd
import requests
import json
from datetime import datetime
import time
import os
import math
import pytz # <-- New import for time zones

# --- PROXY CONFIGURATION ---
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = { "http": proxy_url, "https": proxy_url } if "YOUR_IP" not in PROXY_IP else None

# --- General Configuration ---
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"

# --- [All indicator and data fetching functions remain the same] ---
def calc_ema(values, period):
    if not isinstance(values, list) or len(values) < period: return [None] * len(values)
    return pd.Series(values).ewm(span=period, adjust=False).mean().tolist()
def calc_rsi(values, period=14):
    if not isinstance(values, list) or len(values) < period + 1: return [None] * len(values)
    series = pd.Series(values)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).tolist()
# In run_automation.py
def get_last_valid_value(values):
    """Iterates backwards through a list to find the last valid number."""
    # The 'for' loop starts a new block
    for value in reversed(values):
        # This 'if' statement MUST be indented to be inside the loop
        if value is not None and not math.isnan(value):
            # This 'return' must be indented to be inside the 'if'
            return value
    # This 'return' is outside the loop and runs if no valid value was found
    return None
def calc_macd(values, fast=12, slow=26, signal=9):
    series = pd.Series(values)
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = (macd_line - signal_line)
    return {'macd': macd_line.iloc[-1], 'signal': signal_line.iloc[-1], 'histogram': histogram.iloc[-1]}
def calc_bollinger(values, period=20, mult=2):
    if len(values) < period: return {'upper': None, 'middle': None, 'lower': None}
    series = pd.Series(values)
    mean = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    return {'upper': mean + (mult * std), 'middle': mean, 'lower': mean - (mult * std)}
def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1: return None
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1]
def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period: return None
    tp_series = pd.Series([(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)])
    mean = tp_series.rolling(window=period).mean().iloc[-1]
    mean_dev = tp_series.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean(), raw=False).iloc[-1]
    if mean_dev == 0: return 0
    return (tp_series.iloc[-1] - mean) / (0.015 * mean_dev)
def calc_market_trend(closes):
    if len(closes) < 50: return 0
    ema20, ema50 = calc_ema(closes, 20)[-1], calc_ema(closes, 50)[-1]
    if ema20 is None or ema50 is None: return 0
    if ema20 > ema50 and closes[-1] > ema20: return 10
    if ema20 > ema50: return 5
    if ema20 < ema50 and closes[-1] < ema20: return -10
    if ema20 < ema50: return -5
    return 0
def calc_vol_profile(closes, highs, lows, volumes):
    try:
        df = pd.DataFrame({'price': closes, 'volume': volumes})
        price_range = max(highs) - min(lows)
        if price_range == 0: return {'bullish_score': 0, 'bearish_score': 0}
        poc = df.groupby(pd.cut(df['price'], bins=10), observed=False)['volume'].sum().idxmax().mid
        current_price = closes[-1]
        if current_price > poc: return {'bullish_score': 3, 'bearish_score': 0}
        if current_price < poc: return {'bullish_score': 0, 'bearish_score': 3}
        return {'bullish_score': 5, 'bearish_score': 5}
    except:
        return {'bullish_score': 1, 'bearish_score': 1}
def fetch_top_volume_coins(limit=70):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        usdt_pairs = [t for t in data if 'symbol' in t and t['symbol'].endswith('USDT')]
        return [c['symbol'] for c in sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]]
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []
# In run_automation.py

def fetch_binance_data(symbol, timeframe='5m', limit=100):
    """Fetches and formats kline data from Binance Futures."""
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        # Make sure proxies are defined if you need them
        response = requests.get(url, proxies=proxies, timeout=30) 
        response.raise_for_status()
        data = response.json()
        
        # --- CORRECTED SECTION ---
        # This entire block must be indented inside the 'try'
        return [
            {
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5])
            }
            for d in data
        ]
    # The 'except' must be at the same indentation level as 'try'
    except Exception as e:
        print(f"  - Could not fetch data for {symbol}: {e}")
        return []

def analyze_data(symbol, data5m, market_trend):
    """
    5th Amendment version of analyze_data with confidence scoring and 2-of-3 confluence logic.
    """
    if not data5m or len(data5m) < 50:
        return None

    current_price = data5m[-1].get("close")
    if not current_price:
        return None

    closes = [d["close"] for d in data5m]
    highs = [d["high"] for d in data5m]
    lows  = [d["low"] for d in data5m]
    volumes = [d["volume"] for d in data5m]

    # Indicator calculations
    latest_rsi = get_last_valid_value(calc_rsi(closes, 14))
    macd_obj = calc_macd(closes, 12, 26, 9)
    boll = calc_bollinger(closes, 20, 2)
    atr = calc_atr(highs, lows, closes, 14)
    latest_cci = calc_cci(highs, lows, closes, 20)
    latest_ema50 = get_last_valid_value(calc_ema(closes, 50))
    vol_profile_scores = calc_vol_profile(closes, highs, lows, volumes)

    latest_macd_hist = macd_obj.get("histogram")

    # Abort if important values missing
    if any(v is None for v in [latest_rsi, latest_cci, boll.get("lower")]):
        return None

    # --------------------------------------------------------------------
    # Scoring system (similar to 4th, with MACD handled differently)
    buy_score = 0
    sell_score = 0

    # BB + RSI + CCI
    if current_price <= boll["lower"]:
        buy_score += 35
    if latest_rsi <= 30:
        buy_score += 30
    elif 30 < latest_rsi <= 40:
        buy_score += 15
    if latest_cci >= 100:
        buy_score += 15  # reduced weight

    if current_price >= boll["upper"]:
        sell_score += 35
    if latest_rsi >= 70:
        sell_score += 30
    elif 60 <= latest_rsi < 70:
        sell_score += 15
    if latest_cci <= -100:
        sell_score += 15  # reduced weight

    # MACD as additive noise
    if latest_macd_hist > 0:
        sell_score -= 5
    else:
        sell_score += 5

    if latest_macd_hist < 0:
        buy_score += 5
    else:
        buy_score -= 5

    # Trend modifier
    if market_trend <= -5:
        sell_score += 10
        buy_score -= 10
    elif market_trend >= 5:
        buy_score += 10
        sell_score -= 10

    # Basic direction
    initial_signal = "Neutral"
    if buy_score > sell_score and buy_score > 0:
        initial_signal = "Buy"
    elif sell_score > buy_score and sell_score > 0:
        initial_signal = "Sell"

    # Confluence — count how many major signals are present
    confluence_signals = {
        "bb_touch": (current_price <= boll["lower"] or current_price >= boll["upper"]),
        "rsi_extreme": (latest_rsi <= 30 or latest_rsi >= 70),
        "cci_extreme": (latest_cci >= 100 or latest_cci <= -100)
    }
    num_confluence_met = sum(1 for k, v in confluence_signals.items() if v)
    passes_confluence = num_confluence_met >= 2

    # Base threshold of 18
    passes_base_buy  = buy_score  >= 18
    passes_base_sell = sell_score >= 18

    # Volume profile
    passes_vol_buy  = vol_profile_scores["bullish_score"] > 0
    passes_vol_sell = vol_profile_scores["bearish_score"] > 0

    # --------------------------------------------------------------------
    # Risk / TP SL
    tp_factor = 1.8
    sl_factor = 1.8
    effective_atr = atr if atr and atr > 0 else current_price * 0.002

    tp, sl = current_price, current_price
    side = initial_signal

    if initial_signal == "Buy":
        tp = current_price + (effective_atr * tp_factor)
        sl = current_price - (effective_atr * sl_factor)
    elif initial_signal == "Sell":
        tp = current_price - (effective_atr * tp_factor)
        sl = current_price + (effective_atr * sl_factor)

    profit_pct = abs(((tp - current_price) / current_price) * 100) if current_price else 0

    passes_min_profit    = profit_pct >= 2.0
    passes_profit_ceiling = profit_pct <= 10.0

    # --------------------------------------------------------------------
    # Confidence calculation
    # Only compute if at least initial buy/sell detected
    base_score_val = buy_score if initial_signal == "Buy" else sell_score if initial_signal == "Sell" else 0
    base_component = (base_score_val / 100) * 0.4
    confluence_component = (num_confluence_met / 3) * 0.4
    veto_pass_count = 0
    total_veto_checks = 3

    if (initial_signal == "Buy" and passes_base_buy) or (initial_signal == "Sell" and passes_base_sell):
        veto_pass_count += 1
    if (initial_signal == "Buy" and passes_vol_buy) or (initial_signal == "Sell" and passes_vol_sell):
        veto_pass_count += 1
    if passes_min_profit and passes_profit_ceiling:
        veto_pass_count += 1

    veto_component = (veto_pass_count / total_veto_checks) * 0.2

    confidence = max(0, min(100, round((base_component + confluence_component + veto_component) * 100)))

    # --------------------------------------------------------------------
    # Decide final signal label
    signal_label = "Neutral"

    if initial_signal == "Buy":
        if confidence >= 80:
            signal_label = "Strong Buy"
        elif confidence >= 65:
            signal_label = "Buy+"
        elif confidence >= 40:
            signal_label = "Buy"
        else:
            signal_label = "Neutral"

    elif initial_signal == "Sell":
        if confidence >= 80:
            signal_label = "Strong Sell"
        elif confidence >= 65:
            signal_label = "Sell+"
        elif confidence >= 40:
            signal_label = "Sell"
        else:
            signal_label = "Neutral"

    # --------------------------------------------------------------------
    # Leverage suggestion based on confidence
    leverage = 5
    if confidence >= 80:
        leverage = 9
    elif confidence >= 65:
        leverage = 7
    elif confidence >= 50:
        leverage = 6

    # Construct analysis_log
    analysis_log = {
        "buy_score": round(buy_score),
        "sell_score": round(sell_score),
        "initial_signal": initial_signal,
        "num_confluence_met": num_confluence_met,
        "base_threshold_ok": passes_base_buy if initial_signal == "Buy" else passes_base_sell if initial_signal == "Sell" else False,
        "vol_profile_ok": passes_vol_buy if initial_signal == "Buy" else passes_vol_sell if initial_signal == "Sell" else False,
        "min_profit_ok": passes_min_profit,
        "profit_ceiling_ok": passes_profit_ceiling,
        "confidence": confidence,
        "confluence_booleans": {k: str(v) for k, v in confluence_signals.items()}
    }

    return {
        "coin": symbol,
        "price": round(current_price, 4),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "leverage": f"{leverage}x",
        "confidence": confidence,
        "signal": signal_label,
        "estimated_profit": f"{profit_pct:.2f}%",
        "analysis_log": analysis_log,
        "indicators": {
            "rsi5m": latest_rsi,
            "macd5m": macd_obj,
            "boll5m": boll,
            "cci5m": latest_cci,
            "marketTrend": market_trend,
            "volProfile": vol_profile_scores,
            "ema50_5m": latest_ema50
        }
    }
    
# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting automated data fetch...")
    
    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting."); exit()
    
    print(f"Found {len(top_coins)} coins to analyze.")
    
    btc_data = fetch_binance_data("BTCUSDT")
    # This is the corrected line using the dictionary key
    market_trend = calc_market_trend([d["close"] for d in btc_data])
    print(f"Market Trend determined: {market_trend}")

    all_results = []
    for coin in top_coins:
        print(f" - Analyzing {coin}...")
        time.sleep(0.2)
        
        data_5m = fetch_binance_data(coin)
        if not data_5m: continue
        
        result = analyze_data(coin, data_5m, market_trend)
        if result:
            all_results.append(result)

    if all_results:
        strong_signals = [s for s in all_results if "Strong" in s.get('signal', '')]
        print(f"\nAnalysis complete. Found {len(strong_signals)} strong signals.")
        print("Saving full analysis file...")

        # **UPDATED LOGIC for IST Timestamps**
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



















