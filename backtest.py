#!/usr/bin/env python3
# backtest.py  (10th Amendment compatible)
# Usage: python backtest.py
# Produces per-json CSVs in analytics/ and .txt reports in backtest_reports/

import os
import json
import time
import math
import glob
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# ---------- CONFIG ----------
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

DATA_ARCHIVE = "data_archive"
BACKTEST_REPORTS = "backtest_reports"
ANALYTICS_DIR = "analytics"
BINANCE_FAPI = "https://fapi.binance.com"
LOOKAHEAD_MINUTES = 300  # user set; change if needed
LOOKAHEAD_MS = LOOKAHEAD_MINUTES * 60 * 1000
KLINE_LIMIT = 1000
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 0.8

os.makedirs(BACKTEST_REPORTS, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# ---------- UTIL ----------
def request_with_retries(url, params=None, proxies_local=proxies, timeout=REQUEST_TIMEOUT):
    last_exc = None
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, params=params, proxies=proxies_local, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(REQUEST_BACKOFF * (1 + attempt))
    raise last_exc

def ms_from_iso(ts):
    # accepts ISO formatted string with tzinfo; returns milliseconds since epoch
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        # fallback: try parse manually
        try:
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

def to_ist_iso(utc_iso):
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(tz=timezone(timedelta(hours=5, minutes=30)))
        return ist.isoformat()
    except Exception:
        return ""

# Flatten nested dict (one level) for CSV columns
def flatten_indicator_dict(indicators):
    flat = {}
    def norm_key(k):
        return k.replace(" ", "_").replace("-", "_")
    for k, v in (indicators or {}).items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{norm_key(k)}__{norm_key(str(subk))}"] = json.dumps(subv) if not isinstance(subv, (int,float,str)) and subv is not None else subv
        else:
            flat[norm_key(k)] = v
    return flat

# ---------- Binance 1m klines fetch ----------
def fetch_binance_klines_1m(symbol, startTime=None, endTime=None, limit=KLINE_LIMIT):
    """
    Fetch 1m klines from Binance futures with start/end time in ms.
    Returns list of dict with keys: open_time, open, high, low, close, volume, close_time
    """
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": limit}
        if startTime is not None:
            params["startTime"] = int(startTime)
        if endTime is not None:
            params["endTime"] = int(endTime)
        resp = request_with_retries(url, params=params)
        data = resp.json()
        out = []
        for d in data:
            out.append({
                "open_time": int(d[0]),
                "open": float(d[1]),
                "high": float(d[2]),
                "low": float(d[3]),
                "close": float(d[4]),
                "volume": float(d[5]),
                "close_time": int(d[6])
            })
        return out
    except Exception as e:
        # bubble up to caller
        raise

# ---------- Evaluate outcome ----------
def evaluate_signal_outcome(signal_obj):
    """
    signal_obj: dict as saved in data_archive json (one item)
    Returns: structured dict that will become one CSV row
    """
    coin = signal_obj.get("coin")
    signal_label = signal_obj.get("signal")  # "Buy"/"Strong Buy"/"Sell"/"Strong Sell"
    confidence = signal_obj.get("confidence")
    price = float(signal_obj.get("price") or 0.0)
    tp = float(signal_obj.get("tp") or 0.0)
    sl = float(signal_obj.get("sl") or 0.0)
    signal_time_utc = signal_obj.get("signal_time_utc") or signal_obj.get("timestamp") or signal_obj.get("time") or ""
    regime = signal_obj.get("regime") or ""
    analysis_log = signal_obj.get("analysis_log") or {}
    indicators = signal_obj.get("indicators") or {}
    would_be = analysis_log.get("would_be_strong_if") if isinstance(analysis_log, dict) else None

    row = {
        "Coin": coin,
        "Signal": signal_label,
        "Confidence": int(confidence) if isinstance(confidence, (int, float)) else None,
        "Price": price,
        "TP": tp,
        "SL": sl,
        "Regime": regime,
        "signal_time_utc": signal_time_utc,
        "signal_time_ist": to_ist_iso(signal_time_utc) if signal_time_utc else "",
        "would_be_strong_if": json.dumps(would_be) if would_be else "",
        # placeholders for outcome fields
        "Outcome": "Inconclusive",
        "Duration_min": None,
        "Time_to_TP_min": None,
        "Time_to_SL_min": None,
        "Estimated_profit": signal_obj.get("estimated_profit") or "",
    }

    # convert signal_time to ms for fetching 1m candles
    start_ms = ms_from_iso(signal_time_utc)
    if start_ms is None:
        # cannot evaluate if no proper signal timestamp
        row["Outcome"] = "Inconclusive"
        return row

    # compute end time
    end_ms = start_ms + LOOKAHEAD_MS

    # fetch 1m klines for the lookahead window
    try:
        klines = fetch_binance_klines_1m(coin, startTime=start_ms, endTime=end_ms, limit=1000)
    except Exception as e:
        # if fetching fails, mark as Inconclusive and include error text in row for debug
        row["Outcome"] = "Inconclusive"
        row["note"] = f"FetchError: {str(e)}"
        return row

    if not klines:
        row["Outcome"] = "Inconclusive"
        row["note"] = "No klines"
        return row

    # scan minute by minute to find first TP or SL hit
    tp_hit_time = None
    sl_hit_time = None
    max_profit_pct = None
    max_drawdown_pct = None

    # For buy: TP > price, SL < price
    # For sell: TP < price, SL > price
    for k in klines:
        t = k["close_time"]
        h = k["high"]
        l = k["low"]
        # check buy TP/SL
        if (signal_label or "").lower().find("buy") >= 0:
            # TP hit if high >= tp
            if tp and h >= tp and tp_hit_time is None:
                tp_hit_time = t
            # SL hit if low <= sl
            if sl and l <= sl and sl_hit_time is None:
                sl_hit_time = t
        elif (signal_label or "").lower().find("sell") >= 0:
            # TP for sell is price lower target; check low <= tp
            if tp and l <= tp and tp_hit_time is None:
                tp_hit_time = t
            # SL for sell is higher price; check high >= sl
            if sl and h >= sl and sl_hit_time is None:
                sl_hit_time = t

    # Determine which happened first
    first_hit = None
    if tp_hit_time and sl_hit_time:
        first_hit = "TP" if tp_hit_time <= sl_hit_time else "SL"
    elif tp_hit_time:
        first_hit = "TP"
    elif sl_hit_time:
        first_hit = "SL"
    else:
        first_hit = None

    if first_hit == "TP":
        row["Outcome"] = "Success"
        dur_min = None
        try:
            dur_min = int((tp_hit_time - start_ms) / 60000)
        except Exception:
            dur_min = None
        row["Duration_min"] = dur_min
        row["Time_to_TP_min"] = dur_min
        row["Time_to_SL_min"] = None
    elif first_hit == "SL":
        row["Outcome"] = "Fail"
        dur_min = None
        try:
            dur_min = int((sl_hit_time - start_ms) / 60000)
        except Exception:
            dur_min = None
        row["Duration_min"] = dur_min
        row["Time_to_SL_min"] = dur_min
        row["Time_to_TP_min"] = None
    else:
        row["Outcome"] = "Inconclusive"
        row["Duration_min"] = int((klines[-1]["close_time"] - start_ms) / 60000)

    # flatten indicators into CSV-friendly columns (prefix with IND_)
    flat_ind = flatten_indicator_dict(indicators)
    for k, v in flat_ind.items():
        # normalize key names for CSV
        col = f"IND__{k}"
        # ensure CSV-safe cell: convert non-primitive to JSON string
        if isinstance(v, (dict, list)):
            row[col] = json.dumps(v)
        else:
            row[col] = v

    # also flatten analysis_log indicator_scores if present
    if isinstance(analysis_log, dict):
        ind_scores = analysis_log.get("indicator_scores") or {}
        for k, v in ind_scores.items():
            col = f"SCORE__{k}"
            if isinstance(v, (dict, list)):
                row[col] = json.dumps(v)
            else:
                row[col] = v

    return row

# ---------- Build backtest report (text) ----------
def write_text_report(json_filename, rows, out_txt_path):
    # rows is list of dicts produced by evaluate_signal_outcome (one per signal in file)
    # Build similar format to previous backtest text reports
    lines = []
    header = f"BACKTEST REPORT FOR: {os.path.basename(json_filename)}"
    lines.append(header)
    lines.append(f"Processed at UTC: {datetime.utcnow().isoformat()}")
    lines.append("")
    # Group by Signal type -> Buy / Sell
    buys = [r for r in rows if (r.get("Signal") or "").lower().find("buy") >= 0]
    sells = [r for r in rows if (r.get("Signal") or "").lower().find("sell") >= 0]
    def dump_section(title, items):
        lines.append(f"--- {title} ---")
        if not items:
            lines.append(" (none)")
        else:
            df = pd.DataFrame(items)
            cols = ["Coin", "Signal", "Confidence", "Outcome", "Duration_min", "Time_to_TP_min", "Time_to_SL_min", "Estimated_profit"]
            # some columns may not exist; handle
            for idx, it in df.iterrows():
                coin = it.get("Coin")
                sig = it.get("Signal")
                conf = it.get("Confidence")
                outc = it.get("Outcome")
                dur = it.get("Duration_min")
                ttp = it.get("Time_to_TP_min")
                tsl = it.get("Time_to_SL_min")
                est = it.get("Estimated_profit")
                lines.append(f"{coin:10} {sig:15} {str(conf):>4} {outc:12} {str(dur):>6} {str(ttp):>6} {str(tsl):>6} {str(est):>10}")
        lines.append("")
    dump_section("BUY SIGNALS", buys)
    dump_section("SELL SIGNALS", sells)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------- Main ----------
def main():
    print("[INFO] Starting backtest run...")
    json_files = sorted(glob.glob(os.path.join(DATA_ARCHIVE, "*.json")))
    if not json_files:
        print("[INFO] No JSON files in data_archive/. Nothing to do.")
        return

    for jf in json_files:
        print(f"[INFO] Processing {os.path.basename(jf)}...")
        try:
            with open(jf, "r", encoding="utf-8") as f:
                signals = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not read {jf}: {e}")
            continue

        rows = []
        csv_rows = []
        for sig in signals:
            row = evaluate_signal_outcome(sig)
            rows.append(row)
            csv_rows.append(row)

        # write text summary
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        txtname = f"backtest_{os.path.basename(jf).replace('.json','')}_{timestamp}.txt"
        txtpath = os.path.join(BACKTEST_REPORTS, txtname)
        write_text_report(jf, rows, txtpath)
        print(f"[OK] Wrote backtest report {txtpath}")

        # write per-json analytics CSV (flattened)
        csvname = f"{os.path.basename(jf).replace('.json','')}.csv"
        csvpath = os.path.join(ANALYTICS_DIR, csvname)
        try:
            df = pd.DataFrame(csv_rows)
            # Ensure stable ordering of columns; fill missing columns
            df.to_csv(csvpath, index=False)
            print(f"[OK] Wrote analytics CSV {csvpath} with {len(df)} rows.")
        except Exception as e:
            print(f"[WARN] Could not write CSV for {jf}: {e}")

    print("[INFO] Backtest run complete.")

if __name__ == "__main__":
    main()
