# backtest.py (V10 - Enhanced Logging & Proxy Enabled)
import pandas as pd
import requests
from datetime import datetime, timedelta
import re
import pytz
import time
import os
import json
import math # Ensure math is imported

# --- GITHUB CONFIGURATION ---
GITHUB_REPO_URL = "https://github.com/advjps/Crypto-Dashboard-Automation"
REPORTS_FOLDER = "backtest_reports"

# --- PROXY CONFIGURATION ---
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url}

# --- General Configuration ---
HOURS_TO_CHECK = 3

def get_github_archive_urls():
    """Fetches the download URLs for all JSON files in the data_archive folder via proxy."""
    try:
        parts = GITHUB_REPO_URL.strip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/data_archive"
        
        response = requests.get(api_url, proxies=proxies, timeout=20)
        response.raise_for_status()
        files = response.json()
        
        json_urls = [file['download_url'] for file in files if file.get('name', '').endswith('.json')]
        if not json_urls:
            print("No JSON files found in the 'data_archive' folder on GitHub.")
            return []
        return json_urls
    except Exception as e:
        print(f"Error fetching file list from GitHub: {e}")
        return []

def fetch_binance_klines(symbol, start_time, end_time):
    """Fetches kline data from the Futures API via proxy."""
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&startTime={int(start_time.timestamp() * 1000)}&endTime={int(end_time.timestamp() * 1000)}&limit=1000"
    try:
        response = requests.get(url, proxies=proxies, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data: return None
        df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        for col in ['open_time', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        return df
    except Exception as e:
        print(f"    - WARNING: Could not fetch kline data for {symbol}. Error: {e}")
        return None

def parse_timestamp_from_filename(filepath):
    """Extracts and converts the IST timestamp from the filename to UTC."""
    try:
        filename = os.path.basename(filepath)
        match = re.search(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', filename)
        if not match:
            return None

        ist_tz = pytz.timezone("Asia/Kolkata")
        dt_naive = datetime.strptime(match.group(1), '%Y-%m-%d_%H-%M-%S')
        dt_ist = ist_tz.localize(dt_naive)
        dt_utc = dt_ist.astimezone(pytz.utc)
        
        return dt_utc
    except Exception as e:
        print(f"Error parsing timestamp: {e}")
        return None

def analyze_trade_journey(signal, klines_df, start_time):
    """Performs the 'what-if' analysis on a trade's price action."""
    tp = float(signal['tp'])
    sl = float(signal['sl'])
    outcome, duration_min, did_tp_hit_later = "Inconclusive", None, "No"
    
    max_profit_price = float(signal['price'])
    max_drawdown_price = float(signal['price'])
    
    is_buy = 'Buy' in signal['signal']

    for _, row in klines_df.iterrows():
        if is_buy:
            max_profit_price = max(max_profit_price, row['high'])
            max_drawdown_price = min(max_drawdown_price, row['low'])
        else:
            max_profit_price = min(max_profit_price, row['low'])
            max_drawdown_price = max(max_drawdown_price, row['high'])

        if outcome == "Inconclusive":
            if (is_buy and row['high'] >= tp) or (not is_buy and row['low'] <= tp):
                outcome = "Success"
            elif (is_buy and row['low'] <= sl) or (not is_buy and row['high'] >= sl):
                outcome = "Fail"
            
            if outcome != "Inconclusive":
                hit_time = pd.to_datetime(row['open_time'], unit='ms').replace(tzinfo=pytz.utc)
                duration_min = round((hit_time - start_time).total_seconds() / 60)

    if outcome == "Fail":
        if (is_buy and max_profit_price >= tp) or (not is_buy and max_profit_price <= tp):
            did_tp_hit_later = "Yes"
            
    return {
        "Outcome": outcome, "Duration(min)": duration_min, "MaxProfitPrice": max_profit_price,
        "MaxDrawdownPrice": max_drawdown_price, "Did_TP_Hit_Later": did_tp_hit_later
    }

# --- Main Execution Block ---
def run_batch_backtest():
    json_urls = get_github_archive_urls()
    if not json_urls: return

    print(f"Found {len(json_urls)} JSON files in the GitHub archive.")
    
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    for url in json_urls:
        filename = url.split('/')[-1]
        print(f"\n================ PROCESSING: {filename} ================")
        
        try:
            response = requests.get(url, proxies=proxies, timeout=20)
            response.raise_for_status()
            signals = response.json()
        except Exception as e:
            print(f"Could not download or parse JSON from {url}. Error: {e}")
            continue

        start_time_utc = parse_timestamp_from_filename(filename)
        if not start_time_utc:
            print("Could not parse timestamp from filename. Skipping.")
            continue
        
        all_trades_data = []
        for signal in signals:
            if "Neutral" in signal.get('signal', 'Neutral') or not signal.get('indicators'):
                continue
            
            print(f"- Analyzing {signal['coin']} ({signal['signal']})...")
            klines = fetch_binance_klines(signal['coin'], start_time_utc, start_time_utc + timedelta(hours=HOURS_TO_CHECK))
            if klines is None or klines.empty:
                print("    - Skipping, no kline data.")
                continue

            journey_analysis = analyze_trade_journey(signal, klines, start_time_utc)
            
            # --- NEW: Unpack the analysis_log for detailed reporting ---
            analysis_log = signal.get('analysis_log', {})

            trade_data = {
                'Coin': signal.get('coin'),
                'Signal': signal.get('signal'),
                'POP': signal.get('pop'),
                'EMA_Boost_Applied': signal.get('ema_boost_applied', False),
                **journey_analysis,
                'Buy_Score': analysis_log.get('buy_score'),
                'Sell_Score': analysis_log.get('sell_score'),
                'Base_Score_OK': analysis_log.get('base_score_ok'),
                'Confluence_OK': analysis_log.get('confluence_ok'),
                'Vol_Profile_OK': analysis_log.get('vol_profile_ok'),
                'Market_Trend_OK': analysis_log.get('market_trend_ok'),
                'MACD_Conflict_OK': analysis_log.get('macd_conflict_ok'),
                'Profit_Ceiling_OK': analysis_log.get('profit_ceiling_ok'),
                'Downgrade_Reason': analysis_log.get('downgrade_reason', 'N/A')
            }
            all_trades_data.append(trade_data)
        
        if not all_trades_data:
            print("No valid signals to process in this file.")
            continue
            
        report_df = pd.DataFrame(all_trades_data)
        report_filename = f"backtest_{start_time_utc.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        report_filepath = os.path.join(REPORTS_FOLDER, report_filename)
        
        with open(report_filepath, "w", encoding='utf-8') as f:
            f.write(f"===== BACKTEST REPORT FOR: {filename} =====\n")
            f.write(f"Signal Generation Time (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for signal_type in ["Strong Buy", "Strong Sell", "Buy", "Sell"]:
                df_filtered = report_df[report_df['Signal'] == signal_type]
                if not df_filtered.empty:
                    f.write(f"--- {signal_type.upper()} SIGNALS ---\n")
                    # Use to_string() for better formatting of many columns
                    f.write(df_filtered.to_string(index=False))
                    f.write("\n\n")

        print(f"SUCCESS! Report saved as '{report_filepath}'")

if __name__ == "__main__":
    run_batch_backtest()
