# rsi_scanner.py

import requests
import pandas as pd
import pandas_ta as ta
import time
import os

# Securely get proxy from environment variables set by GitHub Actions
proxy_url = os.getenv('HTTP_PROXY')
proxies = {
    'http': proxy_url,
    'https': proxy_url
} if proxy_url else None

def fetch_with_retries(url, params=None, retries=3, delay=5):
    """Fetches data from a URL with a retry mechanism."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, proxies=proxies, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None

def get_futures_markets():
    """Fetches all active B-USDT futures markets from CoinDCX."""
    url = "https://api.coindcx.com/exchange/v1/derivatives/futures/data/active_instruments"
    data = fetch_with_retries(url)
    if data:
        # Filter for pairs that are strings and match the desired format
        return [market for market in data if isinstance(market, str) and market.startswith('B-') and market.endswith('_USDT')]
    return []

def get_ticker_data():
    """Fetches ticker data for all futures markets."""
    url = "https://api.coindcx.com/exchange/v1/derivatives/futures/ticker"
    data = fetch_with_retries(url)
    # Create a dictionary for quick lookups: {'B-BTC_USDT': {...ticker_info...}}
    return {item['market']: item for item in data} if data else {}

def get_rsi(pair, interval):
    """Fetches candles and calculates the latest RSI value for a given pair and interval."""
    # CoinDCX API limit is 1000 candles, which is more than enough for RSI 14
    url = "https://public.coindcx.com/market_data/candles"
    params = {'pair': pair, 'interval': interval, 'limit': 100}
    data = fetch_with_retries(url, params=params)

    if not data or len(data) < 15:
        print(f"Not enough data for {pair} on {interval} timeframe.")
        return None, 0 # Return RSI and Volume

    df = pd.DataFrame(data)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # Calculate RSI using pandas-ta
    df.ta.rsi(length=14, append=True)
    
    latest_rsi = df['RSI_14'].iloc[-1]
    latest_volume = df['volume'].iloc[-1]
    
    return latest_rsi, latest_volume

def main():
    """Main function to execute the scanner."""
    print("Fetching active futures markets...")
    markets = get_futures_markets()
    if not markets:
        print("Could not fetch futures markets. Exiting.")
        return

    print("Fetching ticker data for all markets...")
    tickers = get_ticker_data()
    
    results = []
    total_markets = len(markets)
    print(f"Found {total_markets} markets. Starting scan...")

    for i, pair in enumerate(markets, 1):
        print(f"[{i}/{total_markets}] Scanning {pair}...")
        
        rsi_1m, volume_1m = get_rsi(pair, '1m')

        if rsi_1m is not None and (rsi_1m >= 65 or rsi_1m <= 30):
            print(f"  -> HIT on {pair} with 1-min RSI: {rsi_1m:.2f}. Fetching other timeframes...")
            rsi_5m, _ = get_rsi(pair, '5m')
            rsi_15m, _ = get_rsi(pair, '15m')
            rsi_1d, _ = get_rsi(pair, '1d')

            ticker_info = tickers.get(pair, {})
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
            time.sleep(1) # Small delay after a hit to avoid being rate-limited

    if not results:
        print("Scan complete. No coins matched the RSI criteria.")
        # Create an empty CSV if no results are found to ensure the file exists
        df = pd.DataFrame(columns=['Coin name', 'current price', 'volume', '24 hours change', 'RSI 14 on 1 min Candles', 'RSI 14 on 5 min Candles', 'RSI 14 on 15 min Candles', 'RSI 14 on 1 day candles'])
    else:
        print(f"Scan complete. Found {len(results)} matching coins.")
        df = pd.DataFrame(results)
        df = df.sort_values(by='RSI 14 on 1 min Candles', ascending=False)

    df.to_csv('rsi_dashboard.csv', index=False, float_format='%.2f')
    print("CSV file 'rsi_dashboard.csv' has been created/updated.")

if __name__ == "__main__":
    main()
