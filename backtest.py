# backtest.py (Updated - exports full indicator-level columns + indicator_scores)
import os
import json
import requests
import pandas as pd
from datetime import datetime, timezone
import time
import pytz

# ---------- CONFIG ----------
DATA_ARCHIVE = "data_archive"
REPORTS_DIR = "backtest_reports"
ANALYTICS_DIR = "analytics"

LOOKAHEAD_MINUTES = 300  # 5 hours (configurable)

# Proxy (use your proxy details or set PROXIES = None)
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
PROXIES = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

# ---------- Ensure folders exist ----------
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)
os.makedirs(DATA_ARCHIVE, exist_ok=True)

# ---------- Helpers ----------
def parse_iso_to_utc_ms(s):
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return int(dt_utc.timestamp() * 1000)
    except Exception:
        try:
            # fallback: try parsing without timezone
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            return None

def fetch_binance_1m(symbol, start_ms, end_ms):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": "1m", "startTime": start_ms, "endTime": end_ms, "limit": 1500}
    try:
        r = requests.get(url, params=params, proxies=PROXIES, timeout=25)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[ERR] Could not fetch 1m data for {symbol}: {e}")
        return []

def evaluate_signal_with_klines(signal, klines, start_ms):
    try:
        tp = float(signal.get("tp"))
        sl = float(signal.get("sl"))
    except Exception:
        return {"Outcome": "Inconclusive", "Duration(min)": None, "Did_TP_Hit": False}

    if not klines:
        return {"Outcome": "Inconclusive", "Duration(min)": None, "Did_TP_Hit": False}

    outcome = "Inconclusive"
    did_tp = False
    duration_min = None

    for k in klines:
        try:
            ts = int(k[0]); high = float(k[2]); low = float(k[3])
        except Exception:
            continue
        minutes = (ts - start_ms) // 60000
        side = signal.get("signal", "")
        if "Buy" in side:
            if high >= tp:
                outcome = "Success"; did_tp = True; duration_min = int(minutes); break
            if low <= sl:
                outcome = "Fail"; duration_min = int(minutes); break
        elif "Sell" in side:
            if low <= tp:
                outcome = "Success"; did_tp = True; duration_min = int(minutes); break
            if high >= sl:
                outcome = "Fail"; duration_min = int(minutes); break
    return {"Outcome": outcome, "Duration(min)": duration_min, "Did_TP_Hit": did_tp}

def flatten_indicator_columns(sig):
    """
    From a signal dict, extract indicator-level primitives and indicator_scores into a flat dict of columns.
    """
    out = {}
    # Top-level indicators object
    inds = sig.get("indicators") or {}
    # Flatten common indicator numeric fields (if present)
    # rsi, willr, cci, cmf, stc, tsi, macd histogram, boll/keltner components, hma/alma (categorical)
    if "rsi5m" in inds:
        out["rsi5m"] = inds.get("rsi5m")
    if "williamsR" in inds:
        out["williamsR"] = inds.get("williamsR")
    if "cci5m" in inds:
        out["cci5m"] = inds.get("cci5m")
    if "cmf5m" in inds:
        out["cmf5m"] = inds.get("cmf5m")
    if "stc5m" in inds:
        out["stc5m"] = inds.get("stc5m")
    if "tsi5m" in inds:
        out["tsi5m"] = inds.get("tsi5m")
    if "cvd5m" in inds:
        out["cvd5m"] = inds.get("cvd5m")
    if "hma5m" in inds:
        out["hma5m"] = inds.get("hma5m")
    if "alma5m" in inds:
        out["alma5m"] = inds.get("alma5m")
    # macd subfields
    macd = inds.get("macd5m") or {}
    out["macd5m_macd"] = macd.get("macd")
    out["macd5m_signal"] = macd.get("signal")
    out["macd5m_hist"] = macd.get("histogram")
    # boll subfields
    boll = inds.get("boll5m") or {}
    out["boll_upper"] = boll.get("upper"); out["boll_middle"] = boll.get("middle"); out["boll_lower"] = boll.get("lower")
    # keltner
    kelt = inds.get("keltner5m") or {}
    out["keltner_upper"] = kelt.get("upper"); out["keltner_middle"] = kelt.get("middle"); out["keltner_lower"] = kelt.get("lower")
    # ema
    if "ema50_5m" in inds:
        out["ema50_5m"] = inds.get("ema50_5m")
    # marketTrend
    if "marketTrend" in inds:
        out["marketTrend"] = inds.get("marketTrend")

    # Next: indicator_scores (analysis_log.indicator_scores)
    a_log = sig.get("analysis_log") or {}
    ind_scores = a_log.get("indicator_scores") or {}
    # iterate and flatten nested indicator score dicts
    for key, val in ind_scores.items():
        # normalize key: replace spaces / punctuation with underscores and uppercase the metric role (keep original case)
        col_key = key
        # if val is a dict and contains 'score', extract numeric score
        if isinstance(val, dict) and "score" in val:
            try:
                out[f"{col_key}_score"] = float(val.get("score"))
            except Exception:
                out[f"{col_key}_score"] = val.get("score")
        else:
            # if val itself is numeric, store as x_score for legacy
            try:
                out[f"{col_key}_score"] = float(val)
            except Exception:
                out[f"{col_key}_score"] = val

    return out

def backtest_file(json_path):
    fname = os.path.basename(json_path)
    print(f"[INFO] Processing {fname}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERR] Could not read {json_path}: {e}"); return

    rows = []
    for sig in data:
        if not sig.get("coin") or not sig.get("signal"):
            continue
        # determine start_ms
        start_ms = None
        if sig.get("signal_time_utc"):
            start_ms = parse_iso_to_utc_ms(sig.get("signal_time_utc"))
        elif sig.get("signal_time_ist"):
            start_ms = parse_iso_to_utc_ms(sig.get("signal_time_ist"))
        elif sig.get("signal_time"):
            start_ms = parse_iso_to_utc_ms(sig.get("signal_time"))

        if start_ms:
            end_ms = start_ms + LOOKAHEAD_MINUTES * 60 * 1000
            klines = fetch_binance_1m(sig["coin"], start_ms, end_ms)
            outcome_obj = evaluate_signal_with_klines(sig, klines, start_ms)
        else:
            outcome_obj = {"Outcome": "Inconclusive", "Duration(min)": None, "Did_TP_Hit": False}

        # canonical fields
        row = {
            "SourceFile": fname,
            "Coin": sig.get("coin"),
            "Signal": sig.get("signal"),
            "Confidence": sig.get("confidence") if sig.get("confidence") is not None else sig.get("pop") or None,
            "Outcome": outcome_obj.get("Outcome"),
            "Duration(min)": outcome_obj.get("Duration(min)"),
            "Did_TP_Hit": outcome_obj.get("Did_TP_Hit"),
            "Buy_Score": (sig.get("analysis_log") or {}).get("total_buy_score") or (sig.get("analysis_log") or {}).get("buy_score"),
            "Sell_Score": (sig.get("analysis_log") or {}).get("total_sell_score") or (sig.get("analysis_log") or {}).get("sell_score"),
            "Num_Confluence_Buy": (sig.get("analysis_log") or {}).get("num_confluence_met", {}).get("buy_overshoot") if isinstance((sig.get("analysis_log") or {}).get("num_confluence_met"), dict) else (sig.get("analysis_log") or {}).get("num_conf_buy"),
            "Num_Confluence_Sell": (sig.get("analysis_log") or {}).get("num_confluence_met", {}).get("sell_overshoot") if isinstance((sig.get("analysis_log") or {}).get("num_confluence_met"), dict) else (sig.get("analysis_log") or {}).get("num_conf_sell"),
            "Regime": sig.get("regime") or (sig.get("analysis_log") or {}).get("regime_bias", {}).get("btc_trend"),
            "Estimated_Profit": sig.get("estimated_profit"),
            "DeservedStrongBuy": (sig.get("analysis_log") or {}).get("deserved_strong_buy") or (1 if (sig.get("analysis_log") or {}).get("deserved_strong") == "buy" else 0),
            "DeservedStrongSell": (sig.get("analysis_log") or {}).get("deserved_strong_sell") or (1 if (sig.get("analysis_log") or {}).get("deserved_strong") == "sell" else 0),
            "SignalTimeUTC": sig.get("signal_time_utc") or None,
            "SignalTimeIST": sig.get("signal_time_ist") or None
        }

        # append flattened indicators & indicator_scores
        row.update(flatten_indicator_columns(sig))

        rows.append(row)
        time.sleep(0.02)  # gentle

    # write CSV to analytics
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(ANALYTICS_DIR, f"{fname.replace('.json', '')}.csv")
        try:
            df.to_csv(csv_path, index=False)
            print(f"[OK] Wrote analytics CSV: {csv_path}")
        except Exception as e:
            print(f"[ERR] Could not write CSV {csv_path}: {e}")

    # write human readable txt report (short)
    txt_path = os.path.join(REPORTS_DIR, f"backtest_{fname.replace('.json', '.txt')}")
    try:
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(f"===== BACKTEST REPORT FOR: {fname} =====\n")
            tf.write(f"Signals processed: {len(rows)}\n\n")
            if rows:
                df_show = pd.DataFrame(rows)
                buys = df_show[df_show["Signal"].str.contains("Buy", na=False)]
                sells = df_show[df_show["Signal"].str.contains("Sell", na=False)]
                tf.write("--- BUY SIGNALS ---\n")
                tf.write(buys.to_string(index=False) + "\n\n" if not buys.empty else "None\n\n")
                tf.write("--- SELL SIGNALS ---\n")
                tf.write(sells.to_string(index=False) + "\n\n" if not sells.empty else "None\n\n")
    except Exception as e:
        print(f"[WARN] Could not write txt {txt_path}: {e}")

    print(f"[OK] Done {fname}")

def main():
    files = sorted([os.path.join(DATA_ARCHIVE, f) for f in os.listdir(DATA_ARCHIVE) if f.endswith(".json")])
    if not files:
        print("[INFO] No JSON files in data_archive/")
        return
    for jf in files:
        backtest_file(jf)

if __name__ == "__main__":
    main()
