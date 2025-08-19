# backtest.py (6th-ready diagnostics) — Proxy + GitHub fetch + 5th-Amendment fields
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import re
import pytz
import time
import os
import json
import math

# --- GITHUB CONFIGURATION ---
GITHUB_REPO_URL = "https://github.com/advjps/Crypto-Dashboard-Automation"
REPORTS_FOLDER = "backtest_reports"

# --- PROXY CONFIGURATION (same as your current file) ---
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url}

# --- General Configuration ---
HOURS_TO_CHECK = 3  # look ahead window for backtest

# ---------------------------
# Helpers: GitHub + filename time
# ---------------------------

def get_github_archive_urls():
    """Fetch download URLs for all JSON files in the data_archive folder via proxy."""
    try:
        parts = GITHUB_REPO_URL.strip('/').split('/')
        owner, repo = parts[-2], parts[-1]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/data_archive"
        resp = requests.get(api_url, proxies=proxies, timeout=30)
        resp.raise_for_status()
        files = resp.json()
        json_urls = [f['download_url'] for f in files if f.get('name', '').endswith('.json')]
        if not json_urls:
            print("No JSON files found in the 'data_archive' folder on GitHub.")
            return []
        return sorted(json_urls)
    except Exception as e:
        print(f"Error fetching file list from GitHub: {e}")
        return []

FN_TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})')

def parse_timestamp_from_filename(filename: str):
    """
    Extract IST timestamp from e.g. 'signals_2025-08-18_01-16-59.json' and convert to UTC.
    Keeps your current behavior for continuity with reports.
    """
    try:
        m = FN_TS_RE.search(filename)
        if not m:
            return None
        ist_tz = pytz.timezone("Asia/Kolkata")
        dt_naive = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        dt_ist = ist_tz.localize(dt_naive)
        dt_utc = dt_ist.astimezone(pytz.utc)
        return dt_utc
    except Exception as e:
        print(f"Error parsing timestamp from filename '{filename}': {e}")
        return None

# ---------------------------
# Binance klines via proxy
# ---------------------------

def fetch_binance_klines(symbol: str, start_dt_utc: datetime, end_dt_utc: datetime, max_retries: int = 4):
    """
    Fetch 1m futures klines from Binance between start and end UTC times via proxy.
    Returns a pandas DataFrame or None.
    """
    start_ms = int(start_dt_utc.timestamp() * 1000)
    end_ms = int(end_dt_utc.timestamp() * 1000)
    url = (
        "https://fapi.binance.com/fapi/v1/klines"
        f"?symbol={symbol.upper()}&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, proxies=proxies, timeout=25)
            r.raise_for_status()
            data = r.json()
            if not data:
                return None
            df = pd.DataFrame(data, columns=[
                'open_time','open','high','low','close','volume',
                'close_time','quote_asset_volume','number_of_trades',
                'taker_buy_base_asset_volume','taker_buy_quote_asset_volume','ignore'
            ])
            for col in ['open_time','high','low','close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # keep rows strictly within window
            df = df[(df['open_time'] >= start_ms) & (df['open_time'] <= end_ms)]
            return df
        except Exception as e:
            last_err = str(e)
            time.sleep(attempt)  # simple backoff
    print(f"    - WARNING: Could not fetch klines for {symbol}. Last error: {last_err}")
    return None

# ---------------------------
# 5th-Amendment field adapters
# ---------------------------

def compute_percent_b(price, boll):
    """%B: (price - lower) / (upper - lower). < 0 below lower band; > 1 above upper band."""
    try:
        lower = float(boll["lower"])
        upper = float(boll["upper"])
        rng = (upper - lower) if (upper - lower) != 0 else 1e-9
        return (float(price) - lower) / rng
    except Exception:
        return None

def pop_proxy_from_scores(sig: str, buy_score, sell_score):
    """Legacy POP proxy (0–100) from score ratio—useful for comparing against new confidence."""
    try:
        b = float(buy_score or 0.0)
        s = float(sell_score or 0.0)
        if sig and "Buy" in sig:
            denom = (b + abs(s)) or 1.0
            return int(round(max(0.0, min(100.0, (b / denom) * 100))))
        elif sig and "Sell" in sig:
            denom = (abs(b) + s) or 1.0
            return int(round(max(0.0, min(100.0, (s / denom) * 100))))
    except Exception:
        pass
    return None

def extract_fields_5th(entry: dict):
    """
    Normalize fields from a 5th-Amendment JSON entry.
    Backward-compatible: will return None for missing legacy keys.
    """
    sig  = entry.get("signal")  # Strong Buy / Buy / Sell / Strong Sell / Neutral
    conf = entry.get("confidence")  # 0–100 in 5th Amendment
    price = entry.get("price")
    alog = entry.get("analysis_log", {}) or {}
    inds = entry.get("indicators", {}) or {}
    boll = inds.get("boll5m") or {}

    base_ok   = alog.get("base_threshold_ok")
    num_conf  = alog.get("num_confluence_met")
    confl_ok  = (num_conf is not None and num_conf >= 2)
    vol_ok    = alog.get("vol_profile_ok")
    min_ok    = alog.get("min_profit_ok")
    ceil_ok   = alog.get("profit_ceiling_ok")
    buy_s     = alog.get("buy_score")
    sell_s    = alog.get("sell_score")

    # Legacy POP or proxy
    pop = entry.get("pop")
    if pop is None:
        pop = pop_proxy_from_scores(sig, buy_s, sell_s)

    mkt_trend = inds.get("marketTrend")
    macd_hist = inds.get("macd_hist5m")
    rsi       = inds.get("rsi5m")
    cci       = inds.get("cci5m")
    pct_b     = compute_percent_b(price, boll)

    bb_touch     = alog.get("bb_touch")
    rsi_extreme  = alog.get("rsi_extreme")
    cci_extreme  = alog.get("cci_extreme")
    initial_sig  = alog.get("initial_signal")
    downgrade    = alog.get("downgrade_reason") or "N/A"

    return {
        "Signal": sig,
        "Confidence": conf,
        "POP": pop,
        "Buy_Score": buy_s,
        "Sell_Score": sell_s,
        "Base_Score_OK": base_ok,
        "Num_Conf": num_conf,
        "Confluence_OK": confl_ok,
        "Vol_Profile_OK": vol_ok,
        "Min_Profit_OK": min_ok,
        "Profit_Ceiling_OK": ceil_ok,
        "MarketTrend": mkt_trend,
        "MACD_Hist": macd_hist,
        "RSI": rsi,
        "CCI": cci,
        "%B": pct_b,
        "BB_Touch": bb_touch,
        "RSI_Extreme": rsi_extreme,
        "CCI_Extreme": cci_extreme,
        "Initial_Signal": initial_sig,
        "Downgrade_Reason": downgrade,
    }

# ---------------------------
# Outcome logic (first-touch)
# ---------------------------

def analyze_trade_journey(signal_entry: dict, klines_df: pd.DataFrame, start_time_utc: datetime):
    """
    Determines first-touch outcome and basic diagnostics from the next 3 hours of 1m klines.
    """
    try:
        tp = float(signal_entry['tp'])
        sl = float(signal_entry['sl'])
        side_is_buy = 'Buy' in signal_entry['signal']
        cur_price = float(signal_entry.get('price', 0.0))
    except Exception:
        return {"Outcome": "Inconclusive", "Duration(min)": None,
                "MaxProfitPrice": None, "MaxDrawdownPrice": None, "Did_TP_Hit_Later": "No"}

    outcome = "Inconclusive"
    duration_min = None
    did_tp_later = "No"

    max_profit_px = cur_price
    max_drawdown_px = cur_price

    for _, row in klines_df.iterrows():
        high = float(row['high'])
        low  = float(row['low'])

        if side_is_buy:
            # track extremes
            max_profit_px   = max(max_profit_px, high)
            max_drawdown_px = min(max_drawdown_px, low)
            # first-touch check
            if outcome == "Inconclusive":
                if high >= tp:
                    outcome = "Success"
                elif low <= sl:
                    outcome = "Fail"
        else:
            max_profit_px   = min(max_profit_px, low)
            max_drawdown_px = max(max_drawdown_px, high)
            if outcome == "Inconclusive":
                if low <= tp:
                    outcome = "Success"
                elif high >= sl:
                    outcome = "Fail"

        if outcome != "Inconclusive":
            hit_time = pd.to_datetime(row['open_time'], unit='ms', utc=True)
            duration_min = int(round((hit_time - start_time_utc).total_seconds() / 60.0))
            break

    # If failed first, check if TP would have hit later anyway (diagnostic only)
    if outcome == "Fail":
        for _, row in klines_df.iterrows():
            high = float(row['high'])
            low  = float(row['low'])
            if (side_is_buy and high >= tp) or ((not side_is_buy) and low <= tp):
                did_tp_later = "Yes"
                break

    return {
        "Outcome": outcome,
        "Duration(min)": duration_min,
        "MaxProfitPrice": max_profit_px,
        "MaxDrawdownPrice": max_drawdown_px,
        "Did_TP_Hit_Later": did_tp_later
    }

# ---------------------------
# Reporting
# ---------------------------

REPORT_COLUMNS = [
    "Coin", "Signal", "Confidence", "POP", "Outcome", "Duration(min)",
    "MaxProfitPrice", "MaxDrawdownPrice", "Did_TP_Hit_Later",
    "Buy_Score", "Sell_Score",
    "Base_Score_OK", "Num_Conf", "Confluence_OK", "Vol_Profile_OK",
    "Min_Profit_OK", "Profit_Ceiling_OK",
    "MarketTrend", "MACD_Hist",
    "RSI", "CCI", "%B",
    "BB_Touch", "RSI_Extreme", "CCI_Extreme",
    "Initial_Signal",
    "Downgrade_Reason"
]

def format_header():
    return "  " + "  ".join([c.ljust(15) for c in REPORT_COLUMNS])

def format_row(row_dict):
    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, float):
            # keep 6 decimals for prices, 0 decimals for integers
            return f"{x:.6f}" if any(k in row_dict for k in ["MaxProfitPrice","MaxDrawdownPrice"]) else str(round(x, 4))
        return str(x)
    return "  " + "  ".join([fmt(row_dict.get(col, "")).ljust(15) for col in REPORT_COLUMNS])

# ---------------------------
# Main batch backtest
# ---------------------------

def run_batch_backtest():
    urls = get_github_archive_urls()
    if not urls:
        return

    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    print(f"Found {len(urls)} JSON files in the GitHub archive.")

    for url in urls:
        filename = url.split("/")[-1]
        print(f"\n================ PROCESSING: {filename} ================")

        try:
            r = requests.get(url, proxies=proxies, timeout=40)
            r.raise_for_status()
            signals = r.json()
        except Exception as e:
            print(f"Could not download/parse JSON from {url}. Error: {e}")
            continue

        start_time_utc = parse_timestamp_from_filename(filename)
        if not start_time_utc:
            print("Could not parse timestamp from filename. Skipping.")
            continue

        end_time_utc = start_time_utc + timedelta(hours=HOURS_TO_CHECK)

        rows_by_type = {
            "Strong Buy": [], "Buy": [],
            "Strong Sell": [], "Sell": []
        }

        for sig in signals:
            label = sig.get("signal", "Neutral")
            if label == "Neutral":
                continue
            if not sig.get("indicators"):
                continue

            coin = sig.get("coin")
            print(f"- Analyzing {coin} ({label})...")

            kl = fetch_binance_klines(coin, start_time_utc, end_time_utc)
            if kl is None or kl.empty:
                print("    - Skipping, no kline data.")
                continue

            journey = analyze_trade_journey(sig, kl, start_time_utc)
            fields = extract_fields_5th(sig)

            row = {
                "Coin": coin,
                "Signal": fields["Signal"],
                "Confidence": fields["Confidence"],
                "POP": fields["POP"],
                "Outcome": journey["Outcome"],
                "Duration(min)": journey["Duration(min)"],
                "MaxProfitPrice": journey["MaxProfitPrice"],
                "MaxDrawdownPrice": journey["MaxDrawdownPrice"],
                "Did_TP_Hit_Later": journey["Did_TP_Hit_Later"],
                "Buy_Score": fields["Buy_Score"],
                "Sell_Score": fields["Sell_Score"],
                "Base_Score_OK": fields["Base_Score_OK"],
                "Num_Conf": fields["Num_Conf"],
                "Confluence_OK": fields["Confluence_OK"],
                "Vol_Profile_OK": fields["Vol_Profile_OK"],
                "Min_Profit_OK": fields["Min_Profit_OK"],
                "Profit_Ceiling_OK": fields["Profit_Ceiling_OK"],
                "MarketTrend": fields["MarketTrend"],
                "MACD_Hist": fields["MACD_Hist"],
                "RSI": fields["RSI"],
                "CCI": fields["CCI"],
                "%B": (round(fields["%B"], 4) if isinstance(fields["%B"], (int, float)) else ""),
                "BB_Touch": fields["BB_Touch"],
                "RSI_Extreme": fields["RSI_Extreme"],
                "CCI_Extreme": fields["CCI_Extreme"],
                "Initial_Signal": fields["Initial_Signal"],
                "Downgrade_Reason": fields["Downgrade_Reason"],
            }

            # Normalize confidence to int if numeric
            try:
                if row["Confidence"] is not None:
                    row["Confidence"] = int(round(float(row["Confidence"])))
            except Exception:
                pass

            # Bucket rows by label
            if "Strong Buy" == label:
                rows_by_type["Strong Buy"].append(row)
            elif "Buy" == label:
                rows_by_type["Buy"].append(row)
            elif "Strong Sell" == label:
                rows_by_type["Strong Sell"].append(row)
            elif "Sell" == label:
                rows_by_type["Sell"].append(row)
            else:
                # Unknown label; skip
                continue

        # Write report
        out_name = f"backtest_{start_time_utc.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        out_path = os.path.join(REPORTS_FOLDER, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"===== BACKTEST REPORT FOR: {filename} =====\n")
            f.write(f"Signal Generation Time (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for section in ["Strong Buy", "Buy", "Strong Sell", "Sell"]:
                f.write(f"--- {section.upper()} SIGNALS ---\n")
                rows = rows_by_type[section]
                if rows:
                    f.write(format_header() + "\n")
                    for r in rows:
                        f.write(format_row(r) + "\n")
                else:
                    f.write("(None)\n")
                f.write("\n")

        print(f"SUCCESS! Report saved as '{out_path}'")

if __name__ == "__main__":
    run_batch_backtest()
