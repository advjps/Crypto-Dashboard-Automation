# backtest.py (Updated for signal_time_utc compatibility)

import os
import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz

# --- CONFIG ---
DATA_ARCHIVE = "data_archive"
REPORTS_DIR = "backtest_reports"
ANALYTICS_DIR = "analytics"
LOOKAHEAD_MINUTES = 300  # configurable (3h=180, 5h=300)
BINANCE_FAPI = "https://fapi.binance.com"

# Ensure output dirs exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)


def extract_signal_time(signal_entry, filename):
    """
    Extracts the signal time for a given signal entry.
    Priority:
    1. Use 'signal_time_utc' if present in the entry.
    2. Fall back to filename timestamp (legacy).
    Returns a timezone-aware datetime (UTC).
    """
    # Case 1: New field present
    if "signal_time_utc" in signal_entry:
        try:
            return datetime.strptime(
                signal_entry["signal_time_utc"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
        except Exception:
            pass

    # Case 2: Legacy fallback to filename
    filename_time = (
        filename.split("signals_")[-1].split(".json")[0].replace("_STRONG", "")
    )
    try:
        return datetime.strptime(filename_time, "%Y-%m-%d_%H-%M-%S").replace(
            tzinfo=timezone.utc
        )
    except Exception:
        return None


def fetch_binance_1m(symbol, start_time, end_time):
    """Fetch 1m klines between start_time and end_time."""
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": int(start_time.timestamp() * 1000),
        "endTime": int(end_time.timestamp() * 1000),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "time": int(d[0]),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
            }
            for d in data
        ]
    except Exception as e:
        print(f"[ERROR] Could not fetch Binance 1m data for {symbol}: {e}")
        return []


def analyze_signal_outcome(signal, filename):
    """Check if TP or SL was hit first within LOOKAHEAD_MINUTES."""
    symbol = signal["coin"]
    price = signal["price"]
    tp = signal["tp"]
    sl = signal["sl"]

    signal_time = extract_signal_time(signal, filename)
    if not signal_time:
        return None

    start_time = signal_time
    end_time = signal_time + timedelta(minutes=LOOKAHEAD_MINUTES)

    klines = fetch_binance_1m(symbol, start_time, end_time)
    if not klines:
        return None

    outcome = "Inconclusive"
    duration = LOOKAHEAD_MINUTES
    did_tp_hit = False

    for i, k in enumerate(klines):
        high, low = k["high"], k["low"]

        if "Buy" in signal["signal"]:
            if low <= sl:
                outcome = "Fail"
                duration = i + 1
                break
            if high >= tp:
                outcome = "Success"
                duration = i + 1
                did_tp_hit = True
                break

        elif "Sell" in signal["signal"]:
            if high >= sl:
                outcome = "Fail"
                duration = i + 1
                break
            if low <= tp:
                outcome = "Success"
                duration = i + 1
                did_tp_hit = True
                break

    return {
        "coin": symbol,
        "signal": signal["signal"],
        "confidence": signal.get("confidence"),
        "outcome": outcome,
        "duration_min": duration,
        "did_tp_hit_later": did_tp_hit,
        "estimated_profit": signal.get("estimated_profit"),
        "analysis_log": signal.get("analysis_log", {}),
    }


def run_backtest():
    print("[INFO] Starting backtest run...")

    json_files = sorted(
        [f for f in os.listdir(DATA_ARCHIVE) if f.endswith(".json")]
    )
    if not json_files:
        print("[WARN] No JSON files found in data_archive/")
        return

    for jf in json_files:
        filepath = os.path.join(DATA_ARCHIVE, jf)
        print(f"[INFO] Processing {jf}...")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                signals = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read {jf}: {e}")
            continue

        all_results = []
        for sig in signals:
            result = analyze_signal_outcome(sig, jf)
            if result:
                all_results.append(result)

        # Save per-file TXT report
        ist_tz = pytz.timezone("Asia/Kolkata")
        ts = datetime.now(ist_tz).strftime("%Y-%m-%d_%H-%M-%S")
        txt_name = f"backtest_{ts}.txt"
        txt_path = os.path.join(REPORTS_DIR, txt_name)

        with open(txt_path, "w", encoding="utf-8") as outf:
            outf.write(f"===== BACKTEST REPORT FOR: {jf} =====\n")
            outf.write(f"Signal Generation Time (UTC): {extract_signal_time(signals[0], jf) if signals else 'N/A'}\n\n")
            for r in all_results:
                outf.write(json.dumps(r, indent=2))
                outf.write("\n\n")

        print(f"[INFO] Saved report: {txt_path}")

        # Save per-file CSV for analytics
        if all_results:
            df = pd.DataFrame(all_results)
            csv_name = jf.replace(".json", ".csv")
            csv_path = os.path.join(ANALYTICS_DIR, csv_name)
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Saved analytics CSV: {csv_path}")


if __name__ == "__main__":
    run_backtest()
