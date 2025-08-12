# run_automation.py (V4 - Proxy Enabled)
import pandas as pd
import requests
import json
from datetime import datetime
import time
import os

# --- PROXY CONFIGURATION ---
# Replace these placeholders with your actual proxy details. Keep the quotes.
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

# Format for the requests library
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {
    "http": proxy_url,
    "https": proxy_url,
}

# --- General Configuration ---
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"

# --- Indicator Calculation Functions ---
def calc_ema(values, period):
    if not values or len(values) < period: return [None] * len(values)
    k = 2 / (period + 1)
    ema_array = [sum(values[:period]) / period]
    for i in range(period, len(values)):
        if ema_array[-1] is not None:
            ema = (values[i] * k) + (ema_array[-1] * (1 - k))
            ema_array.append(ema)
        else: # Handle gaps in data
            ema_array.append(None)
    return [None] * (len(values) - len(ema_array)) + ema_array

def calc_rsi(values, period=14):
    if len(values) < period + 1: return [None] * len(values)
    series = pd.Series(values)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).tolist()

def calc_macd(values, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(values, fast)
    ema_slow = calc_ema(values, slow)
    macd_line = [f - s if f is not None and s is not None else None for f, s in zip(ema_fast, ema_slow)]
    
    valid_macd_line = [v for v in macd_line if v is not None]
    if not valid_macd_line: return {'histogram': None}
    
    signal_line = calc_ema(valid_macd_line, signal)
    if not signal_line or signal_line[-1] is None: return {'histogram': None}
    
    latest_macd = valid_macd_line[-1]
    latest_signal = signal_line[-1]
    return {'histogram': latest_macd - latest_signal}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period: return {'upper': None, 'lower': None}
    series = pd.Series(values)
    mean = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    return {'upper': mean + (mult * std), 'lower': mean - (mult * std)}

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period: return None
    tp_series = pd.Series([(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)])
    mean = tp_series.rolling(window=period).mean().iloc[-1]
    
    # This line is updated to use the modern calculation for Mean Absolute Deviation
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

# --- Data Fetching Functions ---
def fetch_top_volume_coins(limit=70):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        usdt_pairs = [t for t in data if t['symbol'].endswith('USDT')]
        return [c['symbol'] for c in sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]]
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def fetch_binance_data(symbol, timeframe='5m', limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [(float(d[2]), float(d[3]), float(d[4])) for d in data]
    except Exception as e:
        print(f"  - Could not fetch data for {symbol}: {e}")
        return []

# --- Core Analysis Function ---
def analyze_data(symbol, data5m, market_trend):
    if len(data5m) < 50: return None
    highs, lows, closes = [list(c) for c in zip(*data5m)]
    current_price = closes[-1]

    latest_rsi5m = calc_rsi(closes)[-1]
    latest_boll5m = calc_bollinger(closes)
    latest_cci5m = calc_cci(highs, lows, closes)
    latest_macd_hist5m = calc_macd(closes)['histogram']

    buy_score, sell_score, veto_applied = 0, 0, False

    if latest_boll5m['lower'] and current_price <= latest_boll5m['lower']: buy_score += 35
    if latest_rsi5m and latest_rsi5m <= 30: buy_score += 30
    elif latest_rsi5m and 30 < latest_rsi5m <= 40: buy_score += 15
    if latest_cci5m and latest_cci5m >= 100: buy_score += 25

    if latest_boll5m['upper'] and current_price >= latest_boll5m['upper']: sell_score += 35
    if latest_rsi5m and latest_rsi5m >= 70: sell_score += 30
    elif latest_rsi5m and 60 <= latest_rsi5m < 70: sell_score += 15
    if latest_cci5m and latest_cci5m <= -100: sell_score += 25

    if market_trend <= -10: buy_score = -999; veto_applied = True
    elif market_trend >= 10: sell_score = -999; veto_applied = True
    elif market_trend <= -5: sell_score += 15; buy_score -= 10
    elif market_trend >= 5: buy_score += 15; sell_score -= 10
    
    if latest_macd_hist5m and latest_macd_hist5m > 0 and sell_score > buy_score:
        sell_score -= 45; veto_applied = True

    total_score = buy_score - sell_score
    signal_type = "Neutral"
    if total_score >= 25: signal_type = "Strong Buy"
    elif total_score <= -25: signal_type = "Strong Sell"
    
    if veto_applied and "Strong" in signal_type:
        signal_type = "Buy" if signal_type == "Strong Buy" else "Sell"
    
    return {"coin": symbol, "price": current_price, "signal": signal_type}

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting automated data fetch...")
    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting.")
        exit()
    print(f"Found {len(top_coins)} coins to analyze.")
    
    btc_data = fetch_binance_data("BTCUSDT")
    market_trend = calc_market_trend([d[2] for d in btc_data])
    print(f"Market Trend determined: {market_trend}")

    all_results, strong_signals = [], []
    for coin in top_coins:
        print(f" - Analyzing {coin}...")
        time.sleep(0.2)
        data_5m = fetch_binance_data(coin)
        if not data_5m: continue
        result = analyze_data(coin, data_5m, market_trend)
        if result:
            all_results.append(result)
            if "Strong" in result['signal']:
                strong_signals.append(result)

    if strong_signals:
        print(f"\nFound {len(strong_signals)} strong signals. Saving file...")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_filename = f"signals_{timestamp}.json"
        
        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        archive_filepath = os.path.join(ARCHIVE_FOLDER, archive_filename)
        
        with open(archive_filepath, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Archive file saved to {archive_filepath}")
        
        with open(LIVE_FILENAME, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Live data file saved as {LIVE_FILENAME}")
    else:

        print("\nNo strong signals found. No file will be saved.")
