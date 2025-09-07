#!/usr/bin/env python3
# backtest.py  (10th/11th Amendment compatible)
# Produces per-json CSVs in analytics/ and .txt reports in backtest_reports/

import os
import json
import time
import math
import glob
import requests
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
import pytz

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
        # fallback parsing
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except Exception:
                continue
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

def sanitize_col_name(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z]+', '_', str(s)).strip('_')

def normalize_value(v):
    """Convert numpy scalars and pandas scalars to native Python, otherwise return primitive or JSON-stringable."""
    # None
    if v is None:
        return None
    # native primitives
    if isinstance(v, (bool, int, float, str)):
        return v
    # numpy/pandas scalar (has item())
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    # dict/list -> leave as is (caller may json.dumps)
    if isinstance(v, (dict, list)):
        return v
    # last resort convert to string
    try:
        return str(v)
    except Exception:
        return None

def flatten_indicator_dict(indicators):
    flat = {}
    def norm_key(k):
        return sanitize_col_name(k)
    for k, v in (indicators or {}).items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                key = f"{norm_key(k)}__{norm_key(str(subk))}"
                val = normalize_value(subv)
                if isinstance(val, (dict, list)):
                    flat[key] = json.dumps(val, default=str)
                else:
                    flat[key] = val
        else:
            flat[norm_key(k)] = normalize_value(v)
    return flat

# ---------- Binance 1m klines fetch ----------
def fetch_binance_klines_1m(symbol, startTime=None, endTime=None, limit=KLINE_LIMIT):
    """
    Fetch 1m klines from Binance futures with start/end time in ms.
    Returns list of dict with keys: open_time, open, high, low, close, volume, close_time
    """
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1m", "limit": min(limit, 1000)}
        if startTime is not None:
            params["startTime"] = int(startTime)
        if endTime is not None:
            params["endTime"] = int(endTime)
        resp = request_with_retries(url, params=params, proxies_local=proxies)
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
    try:
        price = float(signal_obj.get("price") or 0.0)
    except Exception:
        price = None
    try:
        tp = float(signal_obj.get("tp") or 0.0)
    except Exception:
        tp = None
    try:
        sl = float(signal_obj.get("sl") or 0.0)
    except Exception:
        sl = None

    signal_time_utc = signal_obj.get("signal_time_utc") or signal_obj.get("timestamp") or signal_obj.get("time") or ""
    regime = signal_obj.get("regime") or ""
    analysis_log = signal_obj.get("analysis_log") or {}
    indicators = signal_obj.get("indicators") or {}
    would_be = analysis_log.get("would_be_strong_if") if isinstance(analysis_log, dict) else None

    # base row
    row = {
        "Coin": coin,
        "Signal": signal_label,
        "Confidence": int(normalize_value(confidence)) if confidence is not None else None,
        "Price": price,
        "TP": tp,
        "SL": sl,
        "Regime": regime,
        "signal_time_utc": signal_time_utc,
        "signal_time_ist": to_ist_iso(signal_time_utc) if signal_time_utc else "",
        "would_be_strong_if": json.dumps(would_be, default=str) if would_be else "",
        # placeholders for outcome fields
        "Outcome": "Inconclusive",
        "Duration_min": None,
        "Time_to_TP_min": None,
        "Time_to_SL_min": None,
        "Estimated_profit": signal_obj.get("estimated_profit") or "",
        # debug note
        "note": ""
    }

    # convert signal_time to ms for fetching 1m candles
    start_ms = ms_from_iso(signal_time_utc)
    if start_ms is None:
        row["Outcome"] = "Inconclusive"
        row["note"] = "BadTimestamp"
        return row

    # compute end time
    end_ms = start_ms + LOOKAHEAD_MS

    # fetch 1m klines for the lookahead window
    try:
        klines = fetch_binance_klines_1m(coin, startTime=start_ms, endTime=end_ms, limit=1000)
    except Exception as e:
        row["Outcome"] = "Inconclusive"
        row["note"] = f"FetchError: {str(e)}"
        return row

    if not klines:
        row["Outcome"] = "Inconclusive"
        row["note"] = "NoKlines"
        return row

    # scan minute by minute to find first TP or SL hit
    tp_hit_time = None
    sl_hit_time = None

    for k in klines:
        t = k["close_time"]
        h = k["high"]
        l = k["low"]
        sl_local = sl
        tp_local = tp
        label = (signal_label or "")
        if label.lower().find("buy") >= 0:
            # TP hit if high >= tp
            if tp_local and h >= tp_local and tp_hit_time is None:
                tp_hit_time = t
            # SL hit if low <= sl
            if sl_local and l <= sl_local and sl_hit_time is None:
                sl_hit_time = t
        elif label.lower().find("sell") >= 0:
            # TP for sell is price lower target; check low <= tp
            if tp_local and l <= tp_local and tp_hit_time is None:
                tp_hit_time = t
            # SL for sell is higher price; check high >= sl
            if sl_local and h >= sl_local and sl_hit_time is None:
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
        try:
            dur_min = int((tp_hit_time - start_ms) / 60000)
        except Exception:
            dur_min = None
        row["Duration_min"] = dur_min
        row["Time_to_TP_min"] = dur_min
        row["Time_to_SL_min"] = None
    elif first_hit == "SL":
        row["Outcome"] = "Fail"
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

    # flatten indicators into CSV-friendly columns (prefix with IND__)
    flat_ind = flatten_indicator_dict(indicators)
    for k, v in flat_ind.items():
        col = f"IND__{sanitize_col_name(k)}"
        if isinstance(v, (dict, list)):
            try:
                row[col] = json.dumps(v, default=str)
            except Exception:
                row[col] = str(v)
        else:
            row[col] = normalize_value(v)

    # flatten analysis_log.components -> COMP__
    if isinstance(analysis_log, dict):
        comps = analysis_log.get("components", {}) or {}
        if isinstance(comps, dict):
            for k, v in comps.items():
                col = f"COMP__{sanitize_col_name(k)}"
                row[col] = normalize_value(v)

    # flatten analysis_log.indicator_scores -> SCORE__
    if isinstance(analysis_log, dict):
        ind_scores = analysis_log.get("indicator_scores", {}) or {}
        if isinstance(ind_scores, dict):
            for k, v in ind_scores.items():
                col = f"SCORE__{sanitize_col_name(k)}"
                val = normalize_value(v)
                # convert booleans to ints, numpy bools handled in normalize_value
                if isinstance(val, bool):
                    val = int(val)
                row[col] = val

    # confluence_flags -> string + we'll return list for binary flag creation
    flags = []
    if isinstance(analysis_log, dict):
        raw_flags = analysis_log.get("confluence_flags", None)
        if isinstance(raw_flags, str):
            # try parse JSON like string
            try:
                parsed = json.loads(raw_flags.replace("'", '"'))
                if isinstance(parsed, list):
                    flags = [str(x) for x in parsed if x]
                else:
                    flags = [raw_flags]
            except Exception:
                # fallback splitting
                flags = [p.strip() for p in re.split(r'[;|,]', raw_flags) if p.strip()]
        elif isinstance(raw_flags, list):
            flags = [str(x) for x in raw_flags if x]
        else:
            flags = []
    row["CONFLUENCE_FLAGS"] = ";".join(flags) if flags else ""

    # would_be_strong_if flatten
    wbs = analysis_log.get("would_be_strong_if") if isinstance(analysis_log, dict) else None
    if isinstance(wbs, dict):
        row["WBS_missing_points"] = normalize_value(wbs.get("missing_points"))
        tmc = wbs.get("top_missing_components", [])
        if isinstance(tmc, list):
            for i in range(3):
                row[f"WBS_top_missing_{i+1}"] = tmc[i]["component"] if i < len(tmc) and isinstance(tmc[i], dict) else (tmc[i] if i < len(tmc) else None)
        else:
            row["WBS_top_missing_1"] = str(tmc)

    return row, flags

# ---------- Build backtest report (text) ----------
def write_text_report(json_filename, rows, out_txt_path):
    lines = []
    header = f"BACKTEST REPORT FOR: {os.path.basename(json_filename)}"
    lines.append(header)
    lines.append(f"Processed at UTC: {datetime.utcnow().isoformat()}")
    lines.append("")
    buys = [r for r in rows if (r.get("Signal") or "").lower().find("buy") >= 0]
    sells = [r for r in rows if (r.get("Signal") or "").lower().find("sell") >= 0]

    def dump_section(title, items):
        lines.append(f"--- {title} ---")
        if not items:
            lines.append(" (none)")
        else:
            for it in items:
                coin = it.get("Coin")
                sig = it.get("Signal")
                conf = it.get("Confidence")
                outc = it.get("Outcome")
                dur = it.get("Duration_min")
                ttp = it.get("Time_to_TP_min")
                tsl = it.get("Time_to_SL_min")
                est = it.get("Estimated_profit")
                lines.append(f"{coin:12} {sig:15} {str(conf):>4} {outc:12} dur:{str(dur):>4} ttp:{str(ttp):>4} tsl:{str(tsl):>4} est:{str(est):>8}")
        lines.append("")

    dump_section("BUY SIGNALS", buys)
    dump_section("SELL SIGNALS", sells)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------- Main ----------
def process_json_file(jf):
    basename = os.path.basename(jf)
    print(f"[INFO] Processing {basename} ...")
    try:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read {jf}: {e}")
        return

    # data may be list of signals or dict containing list
    if isinstance(data, dict) and "signals" in data and isinstance(data["signals"], list):
        signals = data["signals"]
    elif isinstance(data, list):
        signals = data
    else:
        # single signal object
        signals = [data]

    rows = []
    csv_rows = []
    flags_seen = set()
    for sig in signals:
        try:
            row, flags = evaluate_signal_outcome(sig)
        except Exception as e:
            print(f"[WARN] Error evaluating signal in {basename}: {e}")
            continue
        csv_rows.append(row)
        rows.append(row)
        for fl in flags:
            flags_seen.add(fl)

    # write text report
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    txtname = f"backtest_{os.path.splitext(basename)[0]}_{timestamp}.txt"
    txtpath = os.path.join(BACKTEST_REPORTS, txtname)
    write_text_report(basename, rows, txtpath)
    print(f"[OK] Wrote backtest report {txtpath}")

    # write per-json analytics CSV (flattened)
    csvname = f"{os.path.splitext(basename)[0]}.csv"
    csvpath = os.path.join(ANALYTICS_DIR, csvname)
    try:
        df = pd.DataFrame(csv_rows)
        # Add binary FLAG__ columns for flags seen in this file
        for fl in sorted(flags_seen):
            col = "FLAG__" + sanitize_col_name(fl)
            if col not in df.columns:
                df[col] = df["CONFLUENCE_FLAGS"].apply(lambda x: 1 if fl in (x or "") else 0)
        # ensure numeric columns for obvious fields
        for c in df.columns:
            if c.startswith("SCORE__") or c.startswith("COMP__") or c in ("WBS_missing_points", "Confidence", "Price", "TP", "SL"):
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                except Exception:
                    pass
        df.to_csv(csvpath, index=False)
        print(f"[OK] Wrote analytics CSV {csvpath} with {len(df)} rows.")
    except Exception as e:
        print(f"[WARN] Could not write CSV for {jf}: {e}")

def main():
    print("[INFO] Starting backtest run...")
    json_files = sorted(glob.glob(os.path.join(DATA_ARCHIVE, "*.json")))
    if not json_files:
        print("[INFO] No JSON files in data_archive/. Nothing to do.")
        return

    for jf in json_files:
        try:
            process_json_file(jf)
        except Exception as e:
            print(f"[ERROR] Processing {jf}: {e}")

    print("[INFO] Backtest run complete.")

if __name__ == "__main__":
    main()
