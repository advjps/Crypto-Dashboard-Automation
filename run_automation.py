# run_automation.py (V3 - FINAL)
import pandas as pd
import requests
import json
from datetime import datetime
import time
import os

# --- Configuration ---
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"

# --- Indicator Calculation Functions ---
def calc_ema(values, period):
    if not values or len(values) < period: return [None] * len(values)
    k = 2 / (period + 1)
    ema_array = [sum(values[:period]) / period]
    for i in range(period, len(values)):
        ema = (values[i] * k) + (ema_array[-1] * (1 - k))
        ema_array.append(ema)
    return [None] * (len(values) - len(ema_array)) + ema_array

def calc_rsi(values, period=14):
    if len(values) < period + 1: return [None] * len(values)
    deltas = pd.Series(values).diff()
    gains = deltas.where(deltas > 0, 0)
    losses = -deltas.where(deltas < 0, 0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean().tolist()
    avg_loss = losses.rolling(window=period, min_periods=period).mean().tolist()
    rs = [g / l if l != 0 else float('inf') for g, l in zip(avg_gain, avg_loss)]
    rsi = [100 - (100 / (1 + r)) for r in rs]
    return rsi

def calc_macd(values, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(values, fast)
    ema_slow = calc_ema(values, slow)
    macd_line = [f - s if f is not None and s is not None else None for f, s in zip(ema_fast, ema_slow)]
    signal_line = calc_ema([v for v in macd_line if v is not None], signal)
    if not macd_line or not signal_line: return {'histogram': None}
    latest_macd = next((v for v in reversed(macd_line) if v is not None), None)
    latest_signal = next((v for v in reversed(signal_line) if v is not None), None)
    return {'histogram': latest_macd - latest_signal if latest_macd is not None and latest_signal is not None else None}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period: return {'upper': None, 'lower': None}
    series = pd.Series(values)
    rolling_mean = series.rolling(window=period).mean().iloc[-1]
    rolling_std = series.rolling(window=period).std().iloc[-1]
    return {'upper': rolling_mean + (mult * rolling_std), 'lower': rolling_mean - (mult * rolling_std)}

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period: return None
    tp_series = pd.Series([(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)])
    mean_dev = tp_series.rolling(window=period).apply(lambda x: pd.Series(x).mad(), raw=False).iloc[-1]
    rolling_mean = tp_series.rolling(window=period).mean().iloc[-1]
    if mean_dev == 0: return 0
    return (tp_series.iloc[-1] - rolling_mean) / (0.015 * mean_dev)

def calc_market_trend(closes):
    if len(closes) < 50: return 0
    ema20, ema50 = calc_ema(closes, 20)[-1], calc_ema(closes, 50)[-1]
    last_close = closes[-1]
    if ema20 is None or ema50 is None: return 0
    if ema20 > ema50 and last_close > ema20: return 10
    if ema20 > ema50: return 5
    if ema20 < ema50 and last_close < ema20: return -10
    if ema20 < ema50: return -5
    return 0

# --- Data Fetching ---
def fetch_top_volume_coins(limit=70):
    """Fetches the top coins by 24h trading volume from Binance Futures."""
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # This will catch HTTP errors like 403 or 451
        data = response.json()

        # Check if the data is in the expected list-of-dictionaries format
        if not isinstance(data, list) or not data or 'symbol' not in data[0]:
            print(f"Error: Binance API returned an unexpected data format.")
            return [] # Return an empty list to avoid crashing

        usdt_pairs = [t for t in data if t['symbol'].endswith('USDT')]
        sorted_coins = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        return [coin['symbol'] for coin in sorted_coins[:limit]]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching top coins (Network issue): {e}")
        return [] # Return an empty list
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from Binance. Response was not valid JSON.")
        return [] # Return an empty list

def fetch_binance_data(symbol, timeframe='5m', limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
    data = requests.get(url, timeout=10).json()
    return [(float(d[2]), float(d[3]), float(d[4])) for d in data] # high, low, close

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

    # 1. Core Mean-Reversion Scoring
    if latest_boll5m['lower'] and current_price <= latest_boll5m['lower']: buy_score += 35
    if latest_rsi5m <= 30: buy_score += 30
    elif 30 < latest_rsi5m <= 40: buy_score += 15

    if latest_boll5m['upper'] and current_price >= latest_boll5m['upper']: sell_score += 35
    if latest_rsi5m >= 70: sell_score += 30
    elif 60 <= latest_rsi5m < 70: sell_score += 15
    
    if latest_cci5m and latest_cci5m <= -100: sell_score += 25
    if latest_cci5m and latest_cci5m >= 100: buy_score += 25

    # 2. Advanced Filters
    if market_trend <= -10:
        buy_score = -999; veto_applied = True
    elif market_trend >= 10:
        sell_score = -999; veto_applied = True
    elif market_trend <= -5:
        sell_score += 15; buy_score -= 10
    elif market_trend >= 5:
        buy_score += 15; sell_score -= 10
    
    if latest_macd_hist5m and latest_macd_hist5m > 0 and sell_score > buy_score:
        sell_score -= 45; veto_applied = True
    
    # 3. Final Signal Determination
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
    btc_data = fetch_binance_data("BTCUSDT")
    market_trend = calc_market_trend([d[2] for d in btc_data])
    print(f"Market Trend determined: {market_trend}")

    all_results, strong_signals = [], []
    for coin in top_coins:
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
