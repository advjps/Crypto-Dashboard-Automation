#!/usr/bin/env python3
# backtest.py  (10B-compatible)
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

# lookahead & fetch config
LOOKAHEAD_MINUTES = 600  # user-configurable
LOOKAHEAD_MS = LOOKAHEAD_MINUTES * 60 * 1000
KLINE_LIMIT = 1000
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 0.8

# ensure dirs
os.makedirs(DATA_ARCHIVE, exist_ok=True)
os.makedirs(BACKTEST_REPORTS, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)


# ---------- HELPERS ----------
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
    """Accept ISO timestamp string with tzinfo and return ms since epoch, or None."""
    if not ts:
        return None
    try:
        # Python 3.7+: fromisoformat supports offsets like +00:00
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        try:
            # fallback parsing common format
            dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f%z")
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

def to_ist_iso(utc_iso):
    """Convert UTC ISO timestamp to IST ISO (string)."""
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(tz=timezone(timedelta(hours=5, minutes=30)))
        return ist.isoformat()
    except Exception:
        return ""

def flatten_indicator_dict(indicators):
    """
    Flatten nested 'indicators' dict to single-level mapping suitable for CSV.
    nested dict keys become IND__{key}__{subkey} or IND__{key}
    Non-primitive values (lists/dicts) are JSON-dumped.
    """
    flat = {}
    if not isinstance(indicators, dict):
        return flat
    def norm(k):
        return str(k).replace(" ", "_").replace("-", "_")
    for k, v in indicators.items():
        key = norm(k)
        if isinstance(v, dict):
            for subk, subv in v.items():
                col = f"IND__{key}__{norm(subk)}"
                if isinstance(subv, (dict, list)):
                    flat[col] = json.dumps(subv)
                else:
                    flat[col] = subv
        else:
            col = f"IND__{key}"
            if isinstance(v, (dict, list)):
                flat[col] = json.dumps(v)
            else:
                flat[col] = v
    return flat

def safe_col_name(s):
    if s is None:
        return "None"
    s = str(s)
    # replace non-alnum by underscore
    return "".join([c if c.isalnum() else "_" for c in s])

def flatten_analysis_scores(analysis_log):
    """
    Extract indicator_scores and components from analysis_log into flat mapping.
    - SCORE__{name} for indicator_scores
    - COMP__{name} for components
    - CONFL__flags (semicolon joined)
    - WOULD_BE_STRONG -> JSON string
    - HMA_GATEKEEPER__... fields flattened if present
    - SUPERVISOR__... fields flattened if present
    """
    flat = {}
    if not isinstance(analysis_log, dict):
        # ensure canonical fields exist to avoid missing columns later
        flat.setdefault("ANALYSIS__engine", None)
        flat.setdefault("ANALYSIS__confidence", None)
        flat.setdefault("CONFL__flags", "")
        flat.setdefault("WOULD_BE_STRONG", "")
        flat.setdefault("HMA_GATEKEEPER__applied", None)
        flat.setdefault("SUPERVISOR__applied", None)
        return flat

    # components
    comps = analysis_log.get("components") or {}
    if isinstance(comps, dict):
        for k, v in comps.items():
            col = f"COMP__{str(k)}"
            flat[col] = v

    # indicator_scores
    ind_scores = analysis_log.get("indicator_scores") or {}
    if isinstance(ind_scores, dict):
        for k, v in ind_scores.items():
            col = f"SCORE__{str(k)}"
            if isinstance(v, (dict, list)):
                flat[col] = json.dumps(v)
            else:
                flat[col] = v

    # confluence flags
    flags = analysis_log.get("confluence_flags") or []
    if isinstance(flags, (list, tuple)):
        try:
            joined = ";".join([str(x) for x in flags]) if flags else ""
            flat["CONFL__flags"] = joined
        except Exception:
            flat["CONFL__flags"] = ""
        # also create individual binary columns for each flag for easier analytics (prefix CONFL_FLAG__)
        try:
            for f in flags:
                fname = safe_col_name(f)
                col = f"CONFL_FLAG__{fname}"
                flat[col] = 1
        except Exception:
            pass
    else:
        flat["CONFL__flags"] = ""

    # would_be_strong_if
    wbs = analysis_log.get("would_be_strong_if")
    flat["WOULD_BE_STRONG"] = json.dumps(wbs) if wbs is not None else ""

    # hma_gatekeeper info
    hma = analysis_log.get("hma_gatekeeper")
    if isinstance(hma, dict):
        for k, v in hma.items():
            col = f"HMA_GATEKEEPER__{str(k)}"
            flat[col] = v
    else:
        # ensure explicit columns exist (avoid missing columns later)
        flat.setdefault("HMA_GATEKEEPER__applied", None)
        flat.setdefault("HMA_GATEKEEPER__before_confidence", None)
        flat.setdefault("HMA_GATEKEEPER__after_confidence", None)
        flat.setdefault("HMA_GATEKEEPER__reason", None)
        flat.setdefault("HMA_GATEKEEPER__hma_slope", None)
        flat.setdefault("HMA_GATEKEEPER__hma_slope_pct", None)

    # supervisor info (10B)
    sup = analysis_log.get("supervisor")
    if isinstance(sup, dict):
        for k, v in sup.items():
            col = f"SUPERVISOR__{str(k)}"
            flat[col] = v
    else:
        # default supervisor cols to avoid missing columns
        flat.setdefault("SUPERVISOR__applied", None)
        flat.setdefault("SUPERVISOR__before_confidence", None)
        flat.setdefault("SUPERVISOR__after_confidence", None)
        flat.setdefault("SUPERVISOR__hist_norm", None)
        flat.setdefault("SUPERVISOR__hma_slope_pct", None)
        flat.setdefault("SUPERVISOR__cvd_support", None)
        flat.setdefault("SUPERVISOR__rule", None)
        flat.setdefault("SUPERVISOR__reason", None)

    # keep original confidence & engine
    try:
        flat["ANALYSIS__engine"] = analysis_log.get("engine")
        flat["ANALYSIS__confidence"] = analysis_log.get("confidence")
    except Exception:
        flat.setdefault("ANALYSIS__engine", None)
        flat.setdefault("ANALYSIS__confidence", None)

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
    price = float(signal_obj.get("price") or 0.0)
    tp = float(signal_obj.get("tp") or 0.0)
    sl = float(signal_obj.get("sl") or 0.0)
    signal_time_utc = signal_obj.get("signal_time_utc") or signal_obj.get("timestamp") or signal_obj.get("time") or ""
    regime = signal_obj.get("regime") or ""
    analysis_log = signal_obj.get("analysis_log") or {}
    indicators = signal_obj.get("indicators") or {}
    # prepare base row
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
        row["note"] = "Missing or invalid signal_time"
        # still flatten available info so analytics can use it
        flat_ind = flatten_indicator_dict(indicators)
        row.update(flat_ind)
        row.update(flatten_analysis_scores(analysis_log))
        return row

    # compute end time
    end_ms = start_ms + LOOKAHEAD_MS

    # fetch 1m klines for the lookahead window
    try:
        klines = fetch_binance_klines_1m(coin, startTime=start_ms, endTime=end_ms, limit=KLINE_LIMIT)
    except Exception as e:
        row["Outcome"] = "Inconclusive"
        row["note"] = f"FetchError: {str(e)}"
        # flatten indicator & analysis_log for debugging
        flat_ind = flatten_indicator_dict(indicators)
        row.update(flat_ind)
        row.update(flatten_analysis_scores(analysis_log))
        return row

    if not klines:
        row["Outcome"] = "Inconclusive"
        row["note"] = "No klines"
        flat_ind = flatten_indicator_dict(indicators)
        row.update(flat_ind)
        row.update(flatten_analysis_scores(analysis_log))
        return row

    # scan minute by minute to find first TP or SL hit
    tp_hit_time = None
    sl_hit_time = None

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
    row.update(flat_ind)

    # flatten analysis_log components/scores/confluence/would_be/hma_gatekeeper/supervisor
    row.update(flatten_analysis_scores(analysis_log))

    return row


# ---------- Build backtest report (text) ----------
def write_text_report(json_filename, rows, out_txt_path):
    # rows is list of dicts produced by evaluate_signal_outcome (one per signal in file)
    lines = []
    header = f"===== BACKTEST REPORT FOR: {os.path.basename(json_filename)} ====="
    lines.append(header)
    lines.append(f"Signal Generation Time (UTC): {datetime.utcnow().isoformat()}")
    lines.append("")
    # Group by Signal type -> Buy / Sell (include strong variants)
    buys = [r for r in rows if (r.get("Signal") or "").lower().find("buy") >= 0]
    sells = [r for r in rows if (r.get("Signal") or "").lower().find("sell") >= 0]

    def dump_section(title, items):
        lines.append(f"--- {title} ---")
        if not items:
            lines.append("    (none)")
        else:
            # format each row nicely
            for it in items:
                coin = it.get("Coin") or ""
                sig = it.get("Signal") or ""
                conf = it.get("Confidence")
                outcome = it.get("Outcome") or ""
                dur = it.get("Duration_min")
                ttp = it.get("Time_to_TP_min")
                tsl = it.get("Time_to_SL_min")
                est = it.get("Estimated_profit") or ""
                # include would_be_strong missing_points if present
                wbs = it.get("WOULD_BE_STRONG") or ""
                missing = ""
                try:
                    if wbs:
                        parsed = json.loads(wbs)
                        mp = parsed.get("missing_points")
                        if mp is not None:
                            missing = f" missing_pts={mp}"
                except Exception:
                    missing = ""
                # include brief hma_gatekeeper/supervisor markers
                hg = it.get("HMA_GATEKEEPER__applied")
                sup = it.get("SUPERVISOR__applied")
                extras = ""
                try:
                    if hg:
                        extras += " HMA_gate"
                    if sup:
                        extras += " SUP"
                except Exception:
                    pass
                lines.append(f"{coin:10} {sig:15} Conf:{str(conf):>3} Outcome:{outcome:12} Dur(min):{str(dur):>4} TTP:{str(ttp):>4} TSL:{str(tsl):>4} Est:{str(est):>8}{missing}{extras}")
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
        for sig in signals:
            try:
                row = evaluate_signal_outcome(sig)
                rows.append(row)
            except Exception as e:
                print(f"[WARN] Error evaluating signal for {sig.get('coin','?')}: {e}")
                continue

        # write text summary
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        txtname = f"backtest_{os.path.basename(jf).replace('.json','')}_{timestamp}.txt"
        txtpath = os.path.join(BACKTEST_REPORTS, txtname)
        try:
            write_text_report(jf, rows, txtpath)
            print(f"[OK] Wrote backtest report {txtpath}")
        except Exception as e:
            print(f"[WARN] Could not write text report: {e}")

        # write per-json analytics CSV (flattened)
        csvname = f"{os.path.basename(jf).replace('.json','')}.csv"
        csvpath = os.path.join(ANALYTICS_DIR, csvname)
        try:
            df = pd.DataFrame(rows)
            # ensure boolean/None -> numeric where possible: replace NaN with empty
            df.to_csv(csvpath, index=False)
            print(f"[OK] Wrote analytics CSV {csvpath} with {len(df)} rows.")
        except Exception as e:
            print(f"[WARN] Could not write CSV for {jf}: {e}")

    print("[INFO] Backtest run complete.")


if __name__ == "__main__":
    main()
