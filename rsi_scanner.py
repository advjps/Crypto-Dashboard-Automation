import requests
import pandas as pd
import time
import hmac
import hashlib
from datetime import datetime

# Proxy configuration (used if public API fails)
proxy = {
    'http': 'http://NQOgprvOa4fgcWw:Nx8gIuzPunYu7P1@217.180.42.139:48642',
    'https': 'http://NQOgprvOa4fgcWw:Nx8gIuzPunYu7P1@217.180.42.139:48642'
}

# Function to calculate RSI (14-period)
def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = 0
    losses = 0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gain = max(diff, 0)
        loss = max(-diff, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100 if avg_gain > 0 else None
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Function to fetch data with retries
def fetch_with_retries(url, params=None, use_proxy=False, retries=3, delay=2):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, proxies=proxy if use_proxy else None, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Attempt {attempt} failed for {url}: {e}")
            if attempt < retries and response.status_code == 429:
                time.sleep(delay * attempt)
            else:
                print(f"Max retries reached for {url}")
                return None

# Fetch market data
def fetch_market_data():
    url = "https://api.coindcx.com/exchange/v1/market_details"
    data = fetch_with_retries(url)
    if not data:
        print("Trying with proxy...")
        data = fetch_with_retries(url, use_proxy=True)
    if data:
        print(f"Raw market data: {data[:5]}")
        markets = [pair for pair in data if 'pair' in pair and '_USDT' in pair['pair'] and pair.get('status') == 'Active']
        print(f"Filtered {len(markets)} USDT pairs: {[p['pair'] for p in markets]}")
        return markets[:5]  # Limit to 5 for testing
    return []

# Fetch ticker data
def fetch_ticker_data():
    url = "https://api.coindcx.com/exchange/ticker"
    data = fetch_with_retries(url)
    if not data:
        print("Trying with proxy...")
        data = fetch_with_retries(url, use_proxy=True)
    return data or []

# Fetch candlestick data
def fetch_candles(pair, timeframe, limit=50):
    url = "https://public.coindcx.com/market_data/candles"
    params = {"pair": pair, "interval": timeframe, "endTime": int(time.time() * 1000), "limit": limit}
    data = fetch_with_retries(url, params)
    if not data:
        print(f"Trying with proxy for {pair} ({timeframe})...")
        data = fetch_with_retries(url, params, use_proxy=True)
    return data[::-1] if data else None  # Reverse for chronological order

# Main function
def main():
    # Fetch market and ticker data
    markets = fetch_market_data()
    if not markets:
        print("No USDT pairs found")
        return
    ticker_data = fetch_ticker_data()

    results = []
    for i, market in enumerate(markets, 1):
        print(f"Processing pair {i}/{len(markets)}: {market['pair']}...")
        pair = market['pair']
        symbol = market.get('target_currency_short_name', pair.split('_')[0])

        # Fetch candlestick data
        timeframes = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '1d': '1d'}
        candles = {}
        volume = 0
        for key, interval in timeframes.items():
            data = fetch_candles(pair, interval)
            if data and len(data) > 14:
                candles[key] = data
                if key == '1m':
                    volume = float(data[-1].get('volume', 0))
            else:
                candles[key] = None
            time.sleep(2)  # Avoid rate limits

        # Calculate RSI
        rsi_values = {}
        for key, data in candles.items():
            if data:
                closes = [float(candle['close']) for candle in data]
                rsi_values[key] = calculate_rsi(closes)
            else:
                rsi_values[key] = None

        # Filter for RSI (1-min) > 60 or < 30
        rsi_1m = rsi_values.get('1m')
        if rsi_1m is not None and (rsi_1m > 60 or rsi_1m < 30):
            ticker = next((t for t in ticker_data if t['market'] == pair), {})
            results.append({
                'Name': symbol,
                'Current Price (USDT)': float(ticker.get('last_price', 0)) if ticker else None,
                'Volume (1-min)': volume,
                '24h Price Change (%)': float(ticker.get('change_24_hour', 0)) if ticker else None,
                'RSI (1-min)': rsi_1m,
                'RSI (5-min)': rsi_values.get('5m'),
                'RSI (15-min)': rsi_values.get('15m'),
                'RSI (1-hour)': rsi_values.get('1h'),
                'RSI (1-day)': rsi_values.get('1d')
            })

    # Create and save CSV
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by='RSI (1-min)', ascending=False)
        df.to_csv('rsi_dashboard.csv', index=False, float_format='%.2f')
        print(f"Saved {len(df)} pairs to rsi_dashboard.csv")
    else:
        print("No pairs with RSI (1-min) > 60 or < 30 found")
        # Create empty CSV to avoid workflow errors
        pd.DataFrame(columns=[
            'Name', 'Current Price (USDT)', 'Volume (1-min)', '24h Price Change (%)',
            'RSI (1-min)', 'RSI (5-min)', 'RSI (15-min)', 'RSI (1-hour)', 'RSI (1-day)'
        ]).to_csv('rsi_dashboard.csv', index=False)

if __name__ == "__main__":
    main()
