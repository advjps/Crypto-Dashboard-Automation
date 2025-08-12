# run_automation.py (V8 - Futures API Final)
import pandas as pd
import requests
import json
from datetime import datetime
import time
import os

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

# --- Indicator Calculation Functions (No Changes) ---
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
def calc_macd(values, fast=12, slow=26, signal=9):
    ema_fast = pd.Series(values).ewm(span=fast, adjust=False).mean()
    ema_slow = pd.Series(values).ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = (macd_line - signal_line).tolist()
    return {'macd': macd_line.iloc[-1], 'signal': signal_line.iloc[-1], 'histogram': histogram[-1]}
def calc_bollinger(values, period=20, mult=2):
    if len(values) < period: return {'upper': None, 'middle': None, 'lower': None}
    series = pd.Series(values)
    mean = series.rolling(window=period).mean().iloc[-1]
    std = series.rolling(window=period).std().iloc[-1]
    return {'upper': mean + (mult * std), 'middle': mean, 'lower': mean - (mult * std)}
def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1: return None
    high_low = pd.Series(highs) - pd.Series(lows)
    high_close = (pd.Series(highs) - pd.Series(closes).shift()).abs()
    low_close = (pd.Series(lows) - pd.Series(closes).shift()).abs()
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

# --- Data Fetching Functions ---
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

def fetch_binance_data(symbol, timeframe='5m', limit=100):
    try:
        # **UPDATED to use the Futures API endpoint**
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
        response = requests.get(url, proxies=proxies, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [(float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])) for d in data] # open, high, low, close, volume
    except Exception as e:
        print(f"  - Could not fetch data for {symbol}: {e}")
        return []

# --- Core Analysis Function (No Changes) ---
def analyze_data(symbol, data5m, market_trend):
    # This entire function remains the same as the previous version
    if len(data5m) < 50: return None
    opens, highs, lows, closes, volumes = [list(c) for c in zip(*data5m)]
    current_price = closes[-1]
    latest_rsi = calc_rsi(closes)[-1]
    latest_boll = calc_bollinger(closes)
    latest_cci = calc_cci(highs, lows, closes)
    latest_macd_obj = calc_macd(closes)
    latest_atr = calc_atr(highs, lows, closes)
    latest_vol_profile = calc_vol_profile(closes, highs, lows, volumes)
    latest_ema50 = calc_ema(closes, 50)[-1]
    buy_score, sell_score, veto_applied = 0, 0, False
    if latest_boll.get('lower') and current_price <= latest_boll['lower']: buy_score += 35
    if latest_rsi and latest_rsi <= 30: buy_score += 30
    elif latest_rsi and 30 < latest_rsi <= 40: buy_score += 15
    if latest_cci and latest_cci >= 100: buy_score += 25
    if latest_boll.get('upper') and current_price >= latest_boll['upper']: sell_score += 35
    if latest_rsi and latest_rsi >= 70: sell_score += 30
    elif latest_rsi and 60 <= latest_rsi < 70: sell_score += 15
    if latest_cci and latest_cci <= -100: sell_score += 25
    if latest_ema50 and current_price > latest_ema50: buy_score += 1
    if latest_ema50 and current_price < latest_ema50: sell_score += 1
    if market_trend <= -10: buy_score = -999; veto_applied = True
    elif market_trend >= 10: sell_score = -999; veto_applied = True
    elif market_trend <= -5: sell_score += 15; buy_score -= 10
    elif market_trend >= 5: buy_score += 15; sell_score -= 10
    vol_profile_score = latest_vol_profile['bullish_score'] if buy_score > sell_score else latest_vol_profile['bearish_score']
    if vol_profile_score == 0:
        buy_score -= 50; sell_score -= 50; veto_applied = True
    if latest_macd_obj.get('histogram') and latest_macd_obj['histogram'] > 0 and sell_score > buy_score:
        sell_score -= 45; veto_applied = True
    signal_type = "Neutral"
    STRONG_THRESHOLD = 25
    if buy_score > sell_score:
        if buy_score >= STRONG_THRESHOLD: signal_type = "Strong Buy"
        elif buy_score > 0: signal_type = "Buy"
    elif sell_score > buy_score:
        if sell_score >= STRONG_THRESHOLD: signal_type = "Strong Sell"
        elif sell_score > 0: signal_type = "Sell"
    if veto_applied and "Strong" in signal_type:
        signal_type = "Buy" if signal_type == "Strong Buy" else "Sell"
    pop = 50
    if "Buy" in signal_type: pop = min(100, round((buy_score / (buy_score + abs(sell_score) or 1)) * 100))
    elif "Sell" in signal_type: pop = min(100, round((sell_score / (abs(buy_score) + sell_score or 1)) * 100))
    leverage = 5
    if pop >= 80: leverage = 9
    elif pop >= 65: leverage = 7
    elif pop >= 50: leverage = 6
    sl_factor, tp_factor = 1.5, 1.5
    effective_atr = latest_atr if latest_atr and latest_atr > 0 else current_price * 0.002
    tp, sl = current_price, current_price
    if "Buy" in signal_type:
        tp, sl = current_price + (effective_atr * tp_factor), current_price - (effective_atr * sl_factor)
    elif "Sell" in signal_type:
        tp, sl = current_price - (effective_atr * tp_factor), current_price + (effective_atr * sl_factor)
    profit_pct = abs(((tp - current_price) / current_price) * 100 * leverage) if current_price > 0 else 0
    if profit_pct > 7.0 and "Strong" in signal_type:
        signal_type = "Neutral"
    return {
      "coin": symbol, "price": current_price, "tp": f"{tp:.4f}", "sl": f"{sl:.4f}",
      "leverage": f"{leverage}x", "pop": max(0, pop), "rsi": latest_rsi,
      "estimated_profit": f"{profit_pct:.2f}%", "signal": signal_type,
      "volProfile": latest_vol_profile,
      "indicators": { "rsi5m": latest_rsi, "macd5m": latest_macd_obj, "boll5m": latest_boll,
                      "cci5m": latest_cci, "marketTrend": market_trend, "volProfile": latest_vol_profile,
                      "ema50_5m": latest_ema50 }
    }

# --- Main Execution Block (Corrected for Futures Symbols) ---
if __name__ == "__main__":
    print("Starting automated data fetch...")
    
    top_coins = fetch_top_volume_coins()
    if not top_coins:
        print("Could not fetch top coins. Exiting."); exit()
    
    print(f"Found {len(top_coins)} coins to analyze.")
    
    # Use BTC Futures data for market trend for consistency
    btc_data = fetch_binance_data("BTCUSDT")
    market_trend = calc_market_trend([d[3] for d in btc_data]) # index 3 is close
    print(f"Market Trend determined: {market_trend}")

    all_results, strong_signals = [], []
    for coin in top_coins:
        # This section is now corrected. It uses the original `coin` name directly.
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
        timestamp = datetime.now().strftime("%Y-m-%d_%H-%M-%S")
        
        file_suffix = "_STRONG" if strong_signals else ""
        archive_filename = f"signals_{timestamp}{file_suffix}.json"
        
        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        archive_filepath = os.path.join(ARCHIVE_FOLDER, archive_filename)
        
        with open(archive_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Archive file saved to {archive_filepath}")
        
        with open(LIVE_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Live data file saved as {LIVE_FILENAME}")
    else:
        print("\nNo strong signals found. No file will be saved.")
