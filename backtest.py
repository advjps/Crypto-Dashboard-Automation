# backtest.py — 8th Amendment aligned (TXT + CSV analytics)
# - Reads JSONs from data_archive/
# - Uses proxy to call Binance Futures 1m klines
# - 3h outcome window: Success(FIRST TP), Fail(FIRST SL), Inconclusive
# - For Fails, also mark if TP hit later in the window
# - Outputs: backtest_reports/*.txt and analytics/*.csv
# - Includes Estimated_Profit(%) from JSON; removes MaxProfitPrice/MaxDrawdownPrice

import os
import json
import math
import time
import glob
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import pytz

# ========= Proxy & API =========
PROXY_IP   = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

BINANCE_FAPI = "https://fapi.binance.com"
KLINE_ENDPOINT = f"{BINANCE_FAPI}/fapi/v1/klines"

# ========= Folders =========
ARCHIVE_DIR      = "data_archive"
REPORTS_DIR      = "backtest_reports"
ANALYTICS_DIR    = "analytics"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)

# ========= Backtest window =========
WINDOW_MINUTES = 180  # 3 hours
KLINE_INTERVAL = "1m"

# ========= Helpers =========
def _parse_estimated_profit(val) -> Optional[float]:
    """Accept '3.95%' or 3.95, return float percent or None."""
    if val is None:
        return None
    try:
        if isinstance(val, str):
            s = val.strip().replace("%", "")
            return float(s)
        return float(val)
    except Exception:
        return None

def _safe_float(x, default=None):
    try:
        if x is None: 
            return default
        return float(x)
    except Exception:
        return default

def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default

def _pct_b(price: float, boll: Dict[str,float]) -> Optional[float]:
    try:
        lower = float(boll.get("lower"))
        upper = float(boll.get("upper"))
        rng = (upper - lower)
        if rng == 0:
            return None
        return (price - lower) / rng
    except Exception:
        return None

def parse_ist_timestamp_from_filename(fname: str) -> Optional[datetime]:
    """
    Expects filenames like: signals_YYYY-MM-DD_HH-MM-SS[_STRONG].json  (IST)
    Returns a timezone-aware UTC datetime.
    """
    base = os.path.basename(fname)
    # extract "YYYY-MM-DD_HH-MM-SS" between 'signals_' and optional suffix
    try:
        tag = base.split("signals_")[1].split(".json")[0]
        tag = tag.replace("_STRONG", "")
        ist = pytz.timezone("Asia/Kolkata")
        dt_ist = datetime.strptime(tag, "%Y-%m-%d_%H-%M-%S")
        dt_ist = ist.localize(dt_ist)
        dt_utc = dt_ist.astimezone(timezone.utc)
        return dt_utc
    except Exception:
        return None

def fetch_klines_1m(symbol: str, start_utc: datetime, minutes: int = WINDOW_MINUTES) -> List[Dict[str, Any]]:
    """
    Fetch 1m klines from Binance Futures from start_utc to start_utc + minutes.
    Returns a list of dicts with open_time(ms), open, high, low, close.
    """
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms   = int((start_utc + timedelta(minutes=minutes)).timestamp() * 1000)
    out = []
    fetch_start = start_ms

    # Binance returns max ~1500 bars per query; 3h at 1m = 180 bars, so one query suffices,
    # but keep a loop in case we adjust mins later.
    while fetch_start < end_ms:
        try:
            params = {
                "symbol": symbol,
                "interval": KLINE_INTERVAL,
                "startTime": fetch_start,
                "endTime": end_ms,
                "limit": 1000
            }
            r = requests.get(KLINE_ENDPOINT, params=params, proxies=proxies, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data:
                break

            for d in data:
                # kline format:
                # [0] open time(ms), [1] open, [2] high, [3] low, [4] close, [5] volume, [6] close time(ms), ...
                ot = int(d[0])
                if ot >= end_ms:
                    break
                out.append({
                    "open_time": ot,
                    "open": float(d[1]),
                    "high": float(d[2]),
                    "low":  float(d[3]),
                    "close":float(d[4]),
                })
            # advance: last close time + 1ms
            last_close = int(data[-1][6])
            fetch_start = last_close + 1
            if len(data) < 1000:
                break
            time.sleep(0.05)
        except Exception as e:
            print(f"  - Kline fetch error for {symbol}: {e}")
            break

    # ensure sorted by time
    out.sort(key=lambda x: x["open_time"])
    return out

def backtest_signal(symbol: str, signal: str, price: float, tp: float, sl: float,
                    start_utc: datetime) -> Dict[str, Any]:
    """
    Simulate from start_utc for up to WINDOW_MINUTES using 1m candles.
    Returns outcome dict with Outcome, Duration(min), Did_TP_Hit_Later.
    """
    klines = fetch_klines_1m(symbol, start_utc, WINDOW_MINUTES)
    if not klines:
        return {"Outcome": "Inconclusive", "Duration(min)": 0, "Did_TP_Hit_Later": False}

    # Determine first touch: TP or SL
    # For SELL: price moves down to TP is profit; up to SL is loss
    # For BUY:  price moves up to TP is profit; down to SL is loss
    first_hit = None
    first_hit_time = None

    is_buy = "Buy" in signal
    is_sell = "Sell" in signal

    for i, c in enumerate(klines):
        h = c["high"]
        l = c["low"]
        # Touch logic must consider wicks
        if is_buy:
            hit_tp = (h >= tp)
            hit_sl = (l <= sl)
        else:  # sell
            hit_tp = (l <= tp)
            hit_sl = (h >= sl)

        if hit_tp and hit_sl:
            # If both in same candle, consider "first" by which boundary is closer to open?
            # Simpler: choose the smaller distance in price from open.
            dist_tp = abs((tp - c["open"]) / c["open"])
            dist_sl = abs((sl - c["open"]) / c["open"])
            if dist_tp <= dist_sl:
                first_hit = "TP"
            else:
                first_hit = "SL"
            first_hit_time = c["open_time"]
            break
        elif hit_tp:
            first_hit = "TP"; first_hit_time = c["open_time"]; break
        elif hit_sl:
            first_hit = "SL"; first_hit_time = c["open_time"]; break

    if first_hit is None:
        return {"Outcome": "Inconclusive", "Duration(min)": len(klines), "Did_TP_Hit_Later": False}

    # Duration = minutes until first hit (approx 1m per bar)
    start_ms = int(start_utc.timestamp() * 1000)
    dur_min = max(0, int(round((first_hit_time - start_ms) / 60000.0)))

    if first_hit == "TP":
        return {"Outcome": "Success", "Duration(min)": dur_min, "Did_TP_Hit_Later": False}

    # If first hit SL → outcome Fail, but check if TP was hit later within window
    # Search after the first-hit candle
    tp_later = False
    after = False
    for c in klines:
        if c["open_time"] == first_hit_time:
            after = True
            continue
        if not after:
            continue
        h = c["high"]; l = c["low"]
        if is_buy and h >= tp:
            tp_later = True; break
        if is_sell and l <= tp:
            tp_later = True; break

    return {"Outcome": "Fail", "Duration(min)": dur_min, "Did_TP_Hit_Later": tp_later}

def extract_indicators(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Safely pull indicators from a signal entry (handles old/new JSONs)."""
    inds = entry.get("indicators", {}) or {}
    # RSI
    rsi = _safe_float(inds.get("rsi5m"))
    # CCI
    cci = _safe_float(inds.get("cci5m"))
    # MACD hist
    macd_hist = _safe_float(inds.get("macd_hist5m"))
    # ADX15m
    adx15 = _safe_float(inds.get("adx15m"))
    # %B (new: percentB, else compute)
    percent_b = inds.get("percentB")
    if percent_b is None:
        boll = inds.get("boll5m") or {}
        price = _safe_float(entry.get("price"))
        if price is not None and isinstance(boll, dict):
            try:
                lower = _safe_float(boll.get("lower"))
                upper = _safe_float(boll.get("upper"))
                if lower is not None and upper is not None and upper != lower:
                    percent_b = (price - lower) / (upper - lower)
            except Exception:
                percent_b = None
    percent_b = _safe_float(percent_b)

    # Regime
    regime = inds.get("marketRegime")
    regime_score = _safe_float(inds.get("regimeScore"))
    alog = entry.get("analysis_log", {}) or {}
    if regime is None:
        regime = alog.get("regime")
    if regime_score is None:
        regime_score = _safe_float(alog.get("regime_score"))

    # Confluence, gates (present in newer JSONs)
    num_conf = _safe_int(alog.get("num_confluence_met"))
    vol_ok = bool(alog.get("vol_profile_ok")) if "vol_profile_ok" in alog else None
    overshoot_ok = bool(alog.get("overshoot_ok")) if "overshoot_ok" in alog else None
    min_profit_ok = bool(alog.get("min_profit_ok")) if "min_profit_ok" in alog else None

    buy_score  = _safe_int(alog.get("buy_score"))
    sell_score = _safe_int(alog.get("sell_score"))

    return {
        "RSI": rsi, "CCI": cci, "MACD_Hist": macd_hist, "ADX15m": adx15,
        "%B": percent_b, "Regime": regime, "RegimeScore": regime_score,
        "Num_Conf": num_conf, "Vol_Profile_OK": vol_ok, "Overshoot_OK": overshoot_ok,
        "Min_Profit_OK": min_profit_ok, "Buy_Score": buy_score, "Sell_Score": sell_score
    }

def build_row(coin: str, entry: Dict[str, Any], outcome: Dict[str, Any], signal_utc: datetime) -> Dict[str, Any]:
    """Flatten a single signal into a report row."""
    sig = entry.get("signal", "Neutral")
    confidence = _safe_int(entry.get("confidence"))
    pop = _safe_int(entry.get("pop"))  # might be None in newer JSONs
    price = _safe_float(entry.get("price"))
    tp = _safe_float(entry.get("tp"))
    sl = _safe_float(entry.get("sl"))

    est_profit = _parse_estimated_profit(entry.get("estimated_profit"))

    # indicators & context
    ind = extract_indicators(entry)
    r = {
        "UTC_Time": signal_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "Coin": coin,
        "Signal": sig,
        "Confidence": confidence,
        "POP": pop,
        "Outcome": outcome.get("Outcome"),
        "Duration(min)": outcome.get("Duration(min)"),
        "Did_TP_Hit_Later": outcome.get("Did_TP_Hit_Later", False),
        "Estimated_Profit(%)": est_profit,
        "Price": price,
        "TP": tp,
        "SL": sl,
        # Scores / gates
        "Buy_Score": ind.get("Buy_Score"),
        "Sell_Score": ind.get("Sell_Score"),
        "Base_Score_OK": None,          # not computed here
        "Confluence_OK": (ind.get("Num_Conf") is not None and ind.get("Num_Conf") >= 2),
        "Vol_Profile_OK": ind.get("Vol_Profile_OK"),
        "Min_Profit_OK": ind.get("Min_Profit_OK"),
        # Indicators
        "RSI": ind.get("RSI"),
        "CCI": ind.get("CCI"),
        "%B": ind.get("%B"),
        "ADX15m": ind.get("ADX15m"),
        "MACD_Hist": ind.get("MACD_Hist"),
        # Regime info
        "Regime": ind.get("Regime"),
        "RegimeScore": ind.get("RegimeScore"),
        "Num_Conf": ind.get("Num_Conf"),
        "Overshoot_OK": ind.get("Overshoot_OK"),
    }
    return r

def write_txt_report(filepath: str, signal_utc: datetime, rows: List[Dict[str, Any]]):
    """Write human-readable TXT report."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"===== BACKTEST REPORT FOR: {os.path.basename(filepath).replace('.txt','.json')} =====\n")
        f.write(f"Signal Generation Time (UTC): {signal_utc.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Split by signal type for readability
        buckets = [("BUY SIGNALS", "Buy"), ("SELL SIGNALS", "Sell")]
        for title, key in buckets:
            f.write(f"--- {title} ---\n")
            header = [
                "Coin","Signal","Confidence","Outcome","Duration(min)","Did_TP_Hit_Later",
                "Estimated_Profit(%)","Buy_Score","Sell_Score","Confluence_OK",
                "Vol_Profile_OK","Min_Profit_OK","RSI","CCI","%B","ADX15m","MACD_Hist",
            ]
            f.write("    " + "  ".join(h.ljust(12) for h in header) + "\n")
            for r in rows:
                if key in r["Signal"]:
                    line = [
                        r.get("Coin",""),
                        r.get("Signal",""),
                        str(r.get("Confidence","")),
                        str(r.get("Outcome","")),
                        str(r.get("Duration(min)","")),
                        "Yes" if r.get("Did_TP_Hit_Later") else "No",
                        f"{r.get('Estimated_Profit(%)'):.2f}%" if isinstance(r.get("Estimated_Profit(%)"), (int,float)) else "",
                        str(r.get("Buy_Score","")),
                        str(r.get("Sell_Score","")),
                        str(r.get("Confluence_OK","")),
                        str(r.get("Vol_Profile_OK","")),
                        str(r.get("Min_Profit_OK","")),
                        f"{r.get('RSI'):.2f}" if isinstance(r.get("RSI"), (int,float)) else "",
                        f"{r.get('CCI'):.2f}" if isinstance(r.get("CCI"), (int,float)) else "",
                        f"{r.get('%B'):.3f}" if isinstance(r.get("%B"), (int,float)) else "",
                        f"{r.get('ADX15m'):.2f}" if isinstance(r.get("ADX15m"), (int,float)) else "",
                        f"{r.get('MACD_Hist'):.5f}" if isinstance(r.get("MACD_Hist"), (int,float)) else "",
                    ]
                    f.write("    " + "  ".join(s.ljust(12) for s in line) + "\n")
            f.write("\n")

def write_csv_analytics(csv_path: str, rows: List[Dict[str, Any]]):
    """Write per-file analytics CSV."""
    cols = [
        "UTC_Time","Coin","Signal","Confidence","POP","Outcome","Duration(min)","Did_TP_Hit_Later",
        "Estimated_Profit(%)",
        "Price","TP","SL",
        "Buy_Score","Sell_Score","Base_Score_OK","Confluence_OK","Vol_Profile_OK","Min_Profit_OK",
        "RSI","CCI","%B","ADX15m","MACD_Hist",
        "Regime","RegimeScore","Num_Conf","Overshoot_OK"
    ]
    df = pd.DataFrame(rows)
    # Ensure all cols exist
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df.to_csv(csv_path, index=False)

def process_json_file(json_path: str):
    """Process a single signals JSON into TXT + CSV."""
    # Determine UTC signal time from filename (IST → UTC)
    signal_utc = parse_ist_timestamp_from_filename(json_path) or datetime.now(timezone.utc)

    # Read JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    rows = []
    base = os.path.basename(json_path)
    report_name = base.replace(".json", ".txt")
    csv_name = base.replace(".json", ".csv")

    print(f"[INFO] Backtesting {base} ({len(data)} signals) ...")

    for entry in data:
        coin = entry.get("coin") or entry.get("symbol")
        if not coin:
            continue

        price = _safe_float(entry.get("price"))
        tp    = _safe_float(entry.get("tp"))
        sl    = _safe_float(entry.get("sl"))
        sig   = entry.get("signal","Neutral")

        if not price or not tp or not sl or "Neutral" in sig:
            # Skip neutral or malformed
            continue

        # Evaluate next 3h
        outcome = backtest_signal(coin, sig, price, tp, sl, signal_utc)
        row = build_row(coin, entry, outcome, signal_utc)
        rows.append(row)
        time.sleep(0.05)  # gentle pacing

    # Write TXT and CSV
    txt_path = os.path.join(REPORTS_DIR, report_name)
    write_txt_report(txt_path, signal_utc, rows)

    csv_path = os.path.join(ANALYTICS_DIR, csv_name)
    write_csv_analytics(csv_path, rows)

    print(f"[OK] Wrote TXT: {txt_path}")
    print(f"[OK] Wrote CSV: {csv_path}")

def main():
    json_files = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "signals_*.json")))
    if not json_files:
        print(f"[INFO] No JSON files found under {ARCHIVE_DIR}/")
        return

    for jp in json_files:
        process_json_file(jp)

if __name__ == "__main__":
    main()
