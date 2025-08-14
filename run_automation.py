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
    Analyzes market data to generate a trading signal based on the "3rd Amendment"
    strategy, now with enhanced logging for detailed backtest analysis.
    """
    # --- Data Extraction & Initial Checks ---
    if not data5m or len(data5m) < 50:
        return None

    current_price = data5m[-1]["close"]
    if not current_price:
        return None

    closes = [d["close"] for d in data5m]
    highs = [d["high"] for d in data5m]
    lows = [d["low"] for d in data5m]
    volumes = [d["volume"] for d in data5m]

    # --- Indicator Calculation ---
    latest_rsi = get_last_valid_value(calc_rsi(closes, 14))
    macd_obj = calc_macd(closes, 12, 26, 9)
    boll = calc_bollinger(closes, 20, 2)
    atr = calc_atr(highs, lows, closes, 14)
    latest_cci = calc_cci(highs, lows, closes, 20)
    vol_profile_scores = calc_vol_profile(closes, highs, lows, volumes)
    latest_ema50 = get_last_valid_value(calc_ema(closes, 50))
    latest_macd_hist = macd_obj.get("histogram")

    # --- Data Quality Check ---
    if any(v is None for v in [latest_rsi, latest_cci, boll.get("lower"), latest_macd_hist]):
        print(f"    - Bypassing {symbol} due to insufficient indicator data.")
        return None

    # ==================================================================
    # ### ENHANCED LOGGING & ANALYSIS ###
    # ==================================================================
    
    analysis_log = {}
    downgrade_reasons = []

    # --- 1. Initial Scoring ---
    buy_score = 0
    sell_score = 0

    if current_price <= boll["lower"]: buy_score += 35
    if latest_rsi <= 30: buy_score += 30
    elif 30 < latest_rsi <= 40: buy_score += 15
    if latest_cci >= 100: buy_score += 25

    if current_price >= boll["upper"]: sell_score += 35
    if latest_rsi >= 70: sell_score += 30
    elif 60 <= latest_rsi < 70: sell_score += 15
    if latest_cci <= -100: sell_score += 25

    if market_trend <= -5:
        sell_score += 15
        buy_score -= 10
    elif market_trend >= 5:
        buy_score += 15
        sell_score -= 10
    
    analysis_log['buy_score'] = round(buy_score)
    analysis_log['sell_score'] = round(sell_score)

    # --- 2. Gating, Vetoes & Detailed Logging ---
    signal_type = "Neutral"
    is_strong = False
    BASE_SCORE_THRESHOLD = 20

    if buy_score > sell_score and buy_score > 0:
        signal_type = "Buy"
        analysis_log['initial_signal'] = "Buy"

        passes_base_score = buy_score >= BASE_SCORE_THRESHOLD
        passes_confluence = ((latest_rsi <= 30) + (current_price <= boll["lower"]) + (latest_cci >= 100)) >= 2
        passes_vol_profile = vol_profile_scores["bullish_score"] > 0
        passes_market_trend = market_trend >= 0

        analysis_log['base_score_ok'] = bool(passes_base_score)
        analysis_log['confluence_ok'] = bool(passes_confluence)
        analysis_log['vol_profile_ok'] = bool(passes_vol_profile)
        analysis_log['market_trend_ok'] = bool(passes_market_trend)


        if passes_base_score and passes_confluence and passes_vol_profile and passes_market_trend:
            is_strong = True
            signal_type = "Strong Buy"
            analysis_log['initial_signal'] = "Strong Buy"

    elif sell_score > buy_score and sell_score > 0:
        signal_type = "Sell"
        analysis_log['initial_signal'] = "Sell"

        passes_base_score = sell_score >= BASE_SCORE_THRESHOLD
        passes_confluence = ((latest_rsi >= 70) + (current_price >= boll["upper"]) + (latest_cci <= -100)) >= 2
        passes_vol_profile = vol_profile_scores["bearish_score"] > 0
        passes_market_trend = market_trend <= 0
        passes_macd_conflict = latest_macd_hist <= 0

        analysis_log['base_score_ok'] = bool(passes_base_score)
        analysis_log['confluence_ok'] = bool(passes_confluence)
        analysis_log['vol_profile_ok'] = bool(passes_vol_profile)
        analysis_log['market_trend_ok'] = bool(passes_market_trend)
        analysis_log['macd_conflict_ok'] = bool(passes_macd_conflict)


        if passes_base_score and passes_confluence and passes_vol_profile and passes_market_trend and passes_macd_conflict:
            is_strong = True
            signal_type = "Strong Sell"
            analysis_log['initial_signal'] = "Strong Sell"

    # --- 3. Risk Management & Profit Veto ---
    tp_factor = 1.8
    sl_factor = 1.8
    leverage = 5
    effective_atr = atr if atr and atr > 0 else current_price * 0.002
    
    tp, sl = current_price, current_price
    if "Buy" in signal_type:
        tp = current_price + (effective_atr * tp_factor)
        sl = current_price - (effective_atr * sl_factor)
    elif "Sell" in signal_type:
        tp = current_price - (effective_atr * tp_factor)
        sl = current_price + (effective_atr * sl_factor)
    
    profit_pct = abs(((tp - current_price) / current_price) * 100 * leverage) if current_price > 0 else 0
    passes_profit_ceiling = profit_pct <= 5.0
    analysis_log['profit_ceiling_ok'] = passes_profit_ceiling

    if is_strong and not passes_profit_ceiling:
        signal_type = signal_type.replace("Strong ", "")
        is_strong = False
        downgrade_reasons.append("Profit Ceiling Veto")

    # --- 4. Populate Downgrade Reason ---
    # Find the first check that failed to provide the primary reason
    if analysis_log.get('initial_signal', 'Neutral') == 'Strong Buy' and not is_strong:
        if not analysis_log.get('base_score_ok'): downgrade_reasons.append("Failed Base Score")
        if not analysis_log.get('confluence_ok'): downgrade_reasons.append("Failed Confluence")
        if not analysis_log.get('vol_profile_ok'): downgrade_reasons.append("Failed Vol Profile")
        if not analysis_log.get('market_trend_ok'): downgrade_reasons.append("Failed Market Trend")
    elif analysis_log.get('initial_signal', 'Neutral') == 'Strong Sell' and not is_strong:
        if not analysis_log.get('base_score_ok'): downgrade_reasons.append("Failed Base Score")
        if not analysis_log.get('confluence_ok'): downgrade_reasons.append("Failed Confluence")
        if not analysis_log.get('vol_profile_ok'): downgrade_reasons.append("Failed Vol Profile")
        if not analysis_log.get('market_trend_ok'): downgrade_reasons.append("Failed Market Trend")
        if not analysis_log.get('macd_conflict_ok'): downgrade_reasons.append("Failed MACD Conflict")
        
    analysis_log['downgrade_reason'] = ", ".join(downgrade_reasons) if downgrade_reasons else "N/A"

    # --- 5. POP Score & Final Leverage Calculation ---
    pop = 50
    if "Buy" in signal_type:
        pop = min(100, round((buy_score / ((buy_score + abs(sell_score)) or 1)) * 100))
    elif "Sell" in signal_type:
        pop = min(100, round((sell_score / ((abs(buy_score) + sell_score) or 1)) * 100))
    pop = max(0, pop)

    ema_boost_applied = False
    if is_strong:
        if "Buy" in signal_type and current_price > latest_ema50:
            pop = min(100, round(pop * 1.10))
            ema_boost_applied = True
        elif "Sell" in signal_type and current_price < latest_ema50:
            pop = min(100, round(pop * 1.10))
            ema_boost_applied = True
    
    if pop >= 80: leverage = 9
    elif pop >= 65: leverage = 7
    elif pop >= 50: leverage = 6

    # --- 6. Final Return Object ---
    return {
        "coin": symbol,
        "price": round(current_price, 4),
        "tp": round(tp, 4),
        "sl": round(sl, 4),
        "leverage": f"{leverage}x",
        "pop": pop,
        "signal": signal_type,
        "ema_boost_applied": ema_boost_applied,
        "estimated_profit": f"{profit_pct:.2f}%",
        "analysis_log": analysis_log, # <-- NEW LOG OBJECT
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













