# rsi_scanner.py

import requests
import pandas as pd
import pandas_ta as ta
import time
import os
import sys  # Import sys to use the exit function

# Create a Session object to persist headers and connection settings
session = requests.Session()

# Update headers to more closely mimic a real browser session
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://coindcx.com',
    'Referer': 'https://coindcx.com/'
})

# Securely get proxy from environment variables set by GitHub Actions
proxy_url = os.getenv('HTTP_PROXY')
if proxy_url:
    session.proxies = {
        'http': proxy_url,
        'https': proxy_url
    }

def fetch_with_retries(url, params=None, retries=3, delay=5):
    """Fetches data from a URL with a retry mechanism using the session."""
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None

def get_ticker_data():
    """Fetches ticker data for all futures markets."""
    url = "https://public.coindcx.com/market_data/ticker"
    return fetch_with_retries(url)

def get_rsi(pair, interval):
    """Fetches candles and calculates the latest RSI value for a given pair and interval."""
    url = "https://public.coindcx.com/market_data/candles"
    params = {'pair': pair, 'interval': interval, 'limit': 100}
    data = fetch_with_retries(url, params=params)

    if not data or len(data) < 15:
        print(f"Not enough data for {pair} on {interval} timeframe.")
        return None, 0

    data.reverse()
    df = pd.DataFrame(data)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # --- THIS IS THE FIX FOR RSI MISMATCH ---
    # Use the RMA smoothing method, which matches TradingView and CoinDCX.
    df.ta.rsi(length=14, append=True, ma='RMA')
    
    latest_rsi = df['RSI_14'].iloc[-1]
    latest_volume = df['volume'].iloc[-1]
    
    return latest_rsi, latest_volume

def main():
    """Main function to execute the scanner."""
    print("Fetching ticker data for all markets to filter by volume...")
    all_tickers = get_ticker_data()

    # --- THIS IS THE FIX TO STOP THE SCRIPT ON FAILURE ---
    if not all_tickers:
        print("Could not fetch ticker data. Halting workflow to save minutes.")
        sys.exit(1)  # Exit with an error code to stop the GitHub Action
        
    MIN_VOLUME_USDT = 100_000_000
    high_volume_markets = []
    ticker_map = {}

    for ticker in all_tickers:
        market = ticker.get('market')
        if market and market.startswith('B-') and market.endswith('_USDT'):
            try:
                volume_in_base = float(ticker.get('volume', 0))
                last_price = float(ticker.get('last_price', 0))
                volume_in_usdt = volume_in_base * last_price
                
                if volume_in_usdt >= MIN_VOLUME_USDT:
                    high_volume_markets.append(market)
                    ticker_map[market] = ticker

            except (ValueError, TypeError):
                continue

    if not high_volume_markets:
        print(f"No coins found with 24h volume >= ${MIN_VOLUME_USDT:,}")
        return
    
    results = []
    total_markets = len(high_volume_markets)
    print(f"Found {total_markets} markets with volume over ${MIN_VOLUME_USDT:,}. Starting scan...")

    for i, pair in enumerate(high_volume_markets, 1):
        print(f"[{i}/{total_markets}] Scanning {pair}...")
        
        rsi_1m, volume_1m = get_rsi(pair, '1m')

        if rsi_1m is not None and (rsi_1m >= 65 or rsi_1m <= 30):
            print(f"  -> HIT on {pair} with 1-min RSI: {rsi_1m:.2f}. Fetching other timeframes...")
            rsi_5m, _ = get_rsi(pair, '5m')
            rsi_15m, _ = get_rsi(pair, '15m')
            rsi_1d, _ = get_rsi(pair, '1d')

            ticker_info = ticker_map.get(pair, {})
            results.append({
                'Coin name': pair.replace('B-', '').replace('_USDT', ''),
                'current price': float(ticker_info.get('last_price', 0)),
                'volume': volume_1m,
                '24 hours change': float(ticker_info.get('change_24_hour', 0)),
                'RSI 14 on 1 min Candles': rsi_1m,
                'RSI 14 on 5 min Candles': rsi_5m,
                'RSI 14 on 15 min Candles': rsi_15m,
                'RSI 14 on 1 day candles': rsi_1d
            })
            time.sleep(1)

    if not results:
        print("Scan complete. No high-volume coins matched the RSI criteria.")
        df = pd.DataFrame(columns=['Coin name', 'current price', 'volume', '24 hours change', 'RSI 14 on 1 min Candles', 'RSI 14 on 5 min Candles', 'RSI 14 on 15 min Candles', 'RSI 14 on 1 day candles'])
    else:
        print(f"Scan complete. Found {len(results)} matching coins.")
        df = pd.DataFrame(results)
        df = df.sort_values(by='RSI 14 on 1 min Candles', ascending=False)

    df.to_csv('rsi_dashboard.csv', index=False, float_format='%.2f')
    print("CSV file 'rsi_dashboard.csv' has been created/updated.")

if __name__ == "__main__":
    main()
