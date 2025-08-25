# backtest.py (8b.1) — Reads signals_*.json, simulates next 3h, outputs TXT + CSV with 'deserving_strong'
import os
import json
import math
import time
import glob
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import pytz
import pandas as pd
import requests

# ==============================
# CONFIG
# ==============================
DATA_ARCHIVE_DIR = "data_archive"
REPORTS_DIR = "backtest_reports"
ANALYTICS_DIR = "analytics"

BINANCE_FAPI = "https://fapi.binance.com"
# 1-minute klines
FUTURES_INTERVAL = "1m"
LOOKAHEAD_MINUTES = 300  # 5 hours

# Reuse the same proxy config as automation
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
PROXIES = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

REQUEST_TIMEOUT = 30

# ==============================
# UTILS
# ==============================

def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(ANALYTICS_DIR, exist_ok=True)

def parse_float(x, default=None):
    try:
        if isinstance(x, str) and x.endswith("%"):
            return float(x[:-1])
        return float(x)
    except Exception:
        return default

def parse_signal_time_from_filename(fname: str) -> Optional[datetime]:
    """
    Try to parse IST timestamp from filename like:
    signals_2025-08-22_00-23-42.json (or ..._STRONG.json)
    We treat this as IST time (to match automation naming), then convert to UTC.
    """
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    # Find YYYY-MM-DD_HH-MM-SS inside
    import re
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", stem)
    if not m:
        return None
    ist_str = m.group(0)
    try:
        ist_tz = pytz.timezone("Asia/Kolkata")
        ist_dt = ist_tz.localize(datetime.strptime(ist_str, "%Y-%m-%d_%H-%M-%S"))
        return ist_dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        return None

def fetch_klines_1m(symbol: str, start_time_ms: int, end_time_ms: int) -> List[List[Any]]:
    """
    Fetch 1m klines for [start_time_ms, end_time_ms] from Binance Futures.
    Returns raw kline rows.
    """
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": FUTURES_INTERVAL,
        "startTime": start_time_ms,
        "endTime": end_time_ms,
        "limit": 1000  # enough for 3 hours of 1m data
    }
    r = requests.get(url, params=params, proxies=PROXIES, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def first_hit(tp: float, sl: float, klines: List[List[Any]], side: str) -> Dict[str, Any]:
    """
    Given TP/SL levels and 1m klines AFTER signal time, return which level hit first.
    Kline format indexes:
      0 open time (ms), 1 open, 2 high, 3 low, 4 close, 5 volume, 6 close time (ms), ...
    Returns:
      {
        "outcome": "Success"|"Fail"|"Inconclusive",
        "duration_min": int or None,
        "tp_later": "Yes"|"No"|"N/A"
      }
    """
    hit_tp_at = None
    hit_sl_at = None

    for row in klines:
        k_open_ms = int(row[0])
        k_high = float(row[2])
        k_low = float(row[3])

        if side == "Buy":
            # For buys: price going up hits TP; going down hits SL
            if k_high >= tp and hit_tp_at is None:
                hit_tp_at = k_open_ms  # first moment in this minute bar
            if k_low <= sl and hit_sl_at is None:
                hit_sl_at = k_open_ms
        else:
            # For sells: price going down hits TP; up hits SL
            if k_low <= tp and hit_tp_at is None:
                hit_tp_at = k_open_ms
            if k_high >= sl and hit_sl_at is None:
                hit_sl_at = k_open_ms

        # Determine first hit strictly by time
        if hit_tp_at is not None and hit_sl_at is not None:
            break

    if hit_tp_at is None and hit_sl_at is None:
        return {"outcome": "Inconclusive", "duration_min": None, "tp_later": "N/A"}

    # If both exist, earlier one wins
    if hit_tp_at is not None and (hit_sl_at is None or hit_tp_at <= hit_sl_at):
        # TP first = Success
        duration = int(round((hit_tp_at - int(klines[0][0])) / 60000.0)) if klines else None
        return {"outcome": "Success", "duration_min": duration, "tp_later": "N/A"}

    if hit_sl_at is not None and (hit_tp_at is None or hit_sl_at < hit_tp_at):
        # SL first = Fail, but check if TP later (after SL) within window
        tp_later = "No"
        if hit_tp_at is not None and hit_tp_at > hit_sl_at:
            tp_later = "Yes"
        duration = int(round((hit_sl_at - int(klines[0][0])) / 60000.0)) if klines else None
        return {"outcome": "Fail", "duration_min": duration, "tp_later": tp_later}

    return {"outcome": "Inconclusive", "duration_min": None, "tp_later": "N/A"}

def safe_get(dct: Dict[str, Any], path: List[str], default=None):
    cur = dct
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

# ==============================
# MAIN BACKTEST
# ==============================
def backtest_one_file(json_path: str) -> Dict[str, Any]:
    """
    For a single signals_*.json, create a TXT report and a CSV in analytics/.
    Returns summary counters.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            signals = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not parse {json_path}: {e}")
            return {"file": os.path.basename(json_path), "total": 0, "success": 0, "fail": 0, "inconclusive": 0}

    # Establish signal time from filename (IST → UTC)
    signal_utc = parse_signal_time_from_filename(json_path)
    if signal_utc is None:
        # fallback: use current UTC minus 3h (worst-case)
        signal_utc = datetime.utcnow() - timedelta(hours=3)

    start_ms = int(signal_utc.timestamp() * 1000)
    end_ms = int((signal_utc + timedelta(minutes=LOOKAHEAD_MINUTES)).timestamp() * 1000)

    rows = []
    succ = fail = inconc = 0

    for entry in signals:
        try:
            symbol = entry.get("coin") or entry.get("symbol") or ""
            if not symbol:
                continue

            signal_label = entry.get("signal", "Neutral")
            side = "Buy" if "Buy" in signal_label else "Sell" if "Sell" in signal_label else "Neutral"

            price = parse_float(entry.get("price"), None)
            tp = parse_float(entry.get("tp"), None)
            sl = parse_float(entry.get("sl"), None)
            conf = int(entry.get("confidence", 0)) if entry.get("confidence") is not None else None
            est_profit_str = entry.get("estimated_profit", None)
            est_profit_pct = parse_float(est_profit_str, None)

            # NEW: deserving_strong in 8b / 8b.1
            deserving_strong = bool(entry.get("deserving_strong", False))

            # Grab useful indicators/analysis fields when available
            rsi = parse_float(safe_get(entry, ["indicators", "rsi5m"]), None)
            percentB = parse_float(safe_get(entry, ["indicators", "percentB"]), None)
            macd_hist = parse_float(safe_get(entry, ["indicators", "macd_hist5m"]), None)
            market_trend = parse_float(safe_get(entry, ["indicators", "marketTrend"]), None)

            # From analysis_log (may vary by amendment)
            num_conf_buy = safe_get(entry, ["analysis_log", "num_confluence_buy"], None)
            num_conf_sell = safe_get(entry, ["analysis_log", "num_confluence_sell"], None)
            # pick the active side count or fallback to generic
            if side == "Buy":
                num_conf = num_conf_buy if num_conf_buy is not None else safe_get(entry, ["analysis_log", "num_confluence_met"], None)
            elif side == "Sell":
                num_conf = num_conf_sell if num_conf_sell is not None else safe_get(entry, ["analysis_log", "num_confluence_met"], None)
            else:
                num_conf = safe_get(entry, ["analysis_log", "num_confluence_met"], None)

            vol_ok = safe_get(entry, ["analysis_log", "vol_profile_ok"], None)
            overshoot_ok = safe_get(entry, ["analysis_log", "overshoot_ok"], None)
            min_profit_ok = safe_get(entry, ["analysis_log", "min_profit_ok"], None)

            # If any critical value missing or non-tradable signal, mark inconclusive
            if side == "Neutral" or price is None or tp is None or sl is None:
                outcome = "Inconclusive"
                duration_min = None
                tp_later = "N/A"
            else:
                # fetch 1m klines for the lookahead window and simulate
                try:
                    klines = fetch_klines_1m(symbol, start_ms, end_ms)
                    if not isinstance(klines, list) or len(klines) == 0:
                        outcome = "Inconclusive"
                        duration_min = None
                        tp_later = "N/A"
                    else:
                        hit = first_hit(tp=tp, sl=sl, klines=klines, side=side)
                        outcome = hit["outcome"]
                        duration_min = hit["duration_min"]
                        tp_later = hit["tp_later"]
                except Exception as e:
                    print(f"[WARN] fetch klines failed for {symbol}: {e}")
                    outcome = "Inconclusive"
                    duration_min = None
                    tp_later = "N/A"

            if outcome == "Success":
                succ += 1
            elif outcome == "Fail":
                fail += 1
            else:
                inconc += 1

            rows.append({
                "UTC_Time": signal_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "Coin": symbol,
                "Signal": signal_label,
                "Confidence": conf,
                "Estimated_Profit(%)": est_profit_pct,
                "Deserving_Strong": deserving_strong,   # <--- NEW

                "RSI": rsi,
                "%B": percentB,
                "MACD_Hist": macd_hist,
                "Regime": market_trend,
                "Num_Conf": num_conf,
                "Vol_Profile_OK": vol_ok,
                "Overshoot_OK": overshoot_ok,
                "Min_Profit_OK": min_profit_ok,

                "Price": price,
                "TP": tp,
                "SL": sl,

                "Outcome": outcome,
                "Duration(min)": duration_min,
                "Did_TP_Hit_Later": tp_later
            })

        except Exception as e:
            print(f"[WARN] Error processing entry in {json_path}: {e}")

    # Write TXT report
    ist_tz = pytz.timezone("Asia/Kolkata")
    ist_now = datetime.now(timezone.utc).astimezone(ist_tz)
    out_txt = os.path.join(REPORTS_DIR, f"backtest_{ist_now.strftime('%Y-%m-%d_%H-%M-%S')}.txt")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"===== BACKTEST REPORT FOR: {os.path.basename(json_path)} =====\n")
        f.write(f"Signal Generation Time (UTC): {signal_utc.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Split by side for readability
        df_rows = pd.DataFrame(rows)
        for side_name, side_df in [("BUY SIGNALS", df_rows[df_rows["Signal"].str.contains("Buy", na=False)]),
                                   ("SELL SIGNALS", df_rows[df_rows["Signal"].str.contains("Sell", na=False)])]:
            f.write(f"--- {side_name} ---\n")
            if side_df.empty:
                f.write("  (none)\n\n")
                continue

            # Print a compact table
            printable = side_df[[
                "Coin","Signal","Confidence","Deserving_Strong","Outcome","Duration(min)",
                "Estimated_Profit(%)","TP","SL","RSI","%B","Num_Conf","Vol_Profile_OK","Overshoot_OK","Min_Profit_OK",
                "Did_TP_Hit_Later"
            ]].copy()

            # Ensure sane text formatting
            printable.to_string(f, index=False)
            f.write("\n\n")

        # Totals
        f.write("Summary:\n")
        f.write(f"  Total: {len(rows)}  Success: {succ}  Fail: {fail}  Inconclusive: {inconc}\n")

    # Write CSV to analytics/
    out_csv = os.path.join(ANALYTICS_DIR, f"{os.path.splitext(os.path.basename(out_txt))[0]}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[OK] Wrote: {out_txt}")
    print(f"[OK] Wrote: {out_csv}")

    return {"file": os.path.basename(json_path), "total": len(rows), "success": succ, "fail": fail, "inconclusive": inconc}

def main():
    ensure_dirs()

    json_files = sorted(glob.glob(os.path.join(DATA_ARCHIVE_DIR, "signals_*.json")))
    if not json_files:
        print(f"[INFO] No JSON files found in {DATA_ARCHIVE_DIR}/")
        return

    grand_total = grand_succ = grand_fail = grand_inconc = 0

    for jp in json_files:
        print(f"[RUN] Backtesting {os.path.basename(jp)} ...")
        try:
            res = backtest_one_file(jp)
            grand_total += res["total"]
            grand_succ += res["success"]
            grand_fail += res["fail"]
            grand_inconc += res["inconclusive"]
            # Gentle pacing to respect rate limits
            time.sleep(0.25)
        except Exception as e:
            print(f"[WARN] Failed on {jp}: {e}")

    print("\n=== GRAND TOTALS ===")
    print(f"Total rows: {grand_total}")
    print(f"Success: {grand_succ}  Fail: {grand_fail}  Inconclusive: {grand_inconc}")
    wr = (100.0 * grand_succ / grand_total) if grand_total > 0 else 0.0
    print(f"Overall Win Rate: {wr:.2f}%")

if __name__ == "__main__":
    main()
