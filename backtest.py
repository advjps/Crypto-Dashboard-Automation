#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
from datetime import datetime, timedelta, timezone

import requests
import pandas as pd
import pytz

# =========================
# CONFIG
# =========================
GITHUB_REPO_URL = "https://github.com/advjps/Crypto-Dashboard-Automation"
DATA_DIR_IN_REPO = "data_archive"
REPORTS_FOLDER = "backtest_reports"
ANALYTICS_DIR = "analytics"          # NEW: CSV output folder
HOURS_TO_CHECK = 3                   # backtest horizon (in hours)
BINANCE_FAPI = "https://fapi.binance.com"

# --- PROXY (same as automation) ---
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url}

# =========================
# HELPERS
# =========================
FN_TS_RE = re.compile(r"signals_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")

def parse_timestamp_from_filename(filename: str) -> datetime:
    """
    Files are named with IST timestamps like 'signals_2025-08-18_01-16-59.json'.
    Convert to UTC for Binance queries.
    """
    m = FN_TS_RE.search(filename)
    if not m:
        # fallback: file mtime UTC
        return datetime.fromtimestamp(os.path.getmtime(filename), tz=timezone.utc).replace(microsecond=0)
    date_part, time_part = m.group(1), m.group(2)
    ist = pytz.timezone("Asia/Kolkata")
    dt_ist = ist.localize(datetime.strptime(f"{date_part}_{time_part}", "%Y-%m-%d_%H-%M-%S"))
    return dt_ist.astimezone(pytz.utc)

def utc_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def get_github_archive_urls():
    """List raw download URLs for all JSON files in data_archive/ via GitHub API (through proxy)."""
    try:
        parts = GITHUB_REPO_URL.strip("/").split("/")
        owner, repo = parts[-2], parts[-1]
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{DATA_DIR_IN_REPO}"
        resp = requests.get(api_url, proxies=proxies, timeout=30)
        resp.raise_for_status()
        files = resp.json()
        urls = [f["download_url"] for f in files if f.get("name", "").endswith(".json")]
        return sorted(urls)
    except Exception as e:
        print(f"[ERROR] GitHub list error: {e}")
        return []

# =========================
# BINANCE DATA
# =========================
def fetch_future_klines_1m(symbol: str, start_dt: datetime, minutes: int = 180, max_retries: int = 4):
    """
    Get 1m futures klines for [start, start+minutes] via proxy.
    Returns list of klines: [ [openTime, open, high, low, close, volume, closeTime, ...], ... ]
    """
    start_ms = utc_ms(start_dt)
    end_ms = utc_ms(start_dt + timedelta(minutes=minutes))
    params = {
        "symbol": symbol.upper(),
        "interval": "1m",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1000
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines", params=params, proxies=proxies, timeout=25)
            if r.status_code == 200:
                data = r.json()
                return [k for k in data if start_ms <= int(k[0]) <= end_ms]
            else:
                last_err = f"HTTP {r.status_code} {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(attempt)  # backoff
    print(f"[WARN] Kline fetch failed for {symbol}: {last_err}")
    return []

# =========================
# FIELD EXTRACTION (6th)
# =========================
def compute_percent_b(price, boll):
    """%B: (price - lower) / (upper - lower). <0 below lower; >1 above upper."""
    try:
        lower = float(boll["lower"])
        upper = float(boll["upper"])
        rng = (upper - lower) if (upper - lower) != 0 else 1e-9
        return (float(price) - lower) / rng
    except Exception:
        return None

def pop_proxy_from_scores(sig: str, buy_score, sell_score):
    """Legacy POP proxy from score ratios (for visibility while transitioning)."""
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

def extract_fields(entry: dict):
    """
    Normalize 6th-Amendment JSON to a flat dict for reporting.
    Backward compatible with older JSONs (missing keys -> None/"").
    """
    sig = entry.get("signal")  # Strong Buy / Buy / Sell / Strong Sell / Neutral
    conf = entry.get("confidence")
    price = entry.get("price")
    alog = entry.get("analysis_log", {}) or {}
    inds = entry.get("indicators", {}) or {}

    buy_s = alog.get("buy_score")
    sell_s = alog.get("sell_score")

    base_ok = alog.get("base_threshold_ok")
    num_conf = alog.get("num_confluence_met")
    confluence_ok = (num_conf is not None and num_conf >= 2)
    vol_ok = alog.get("vol_profile_ok")
    min_ok = alog.get("min_profit_ok")
    ceil_ok = alog.get("profit_ceiling_ok")
    vetoes = alog.get("vetoes_passed")
    is_aligned = alog.get("is_regime_aligned")

    boll = inds.get("boll5m") or {}
    pct_b = compute_percent_b(price, boll)

    market_regime = inds.get("marketRegime")
    regime_score = inds.get("regimeScore")
    adx15 = inds.get("adx15m")
    macd_hist = inds.get("macd_hist5m")
    rsi = inds.get("rsi5m")
    cci = inds.get("cci5m")

    pop = entry.get("pop")
    if pop is None:
        pop = pop_proxy_from_scores(sig, buy_s, sell_s)

    return {
        "Signal": sig,
        "Confidence": conf,
        "POP": pop,
        "Buy_Score": buy_s,
        "Sell_Score": sell_s,
        "Base_Score_OK": base_ok,
        "Num_Conf": num_conf,
        "Confluence_OK": confluence_ok,
        "Vol_Profile_OK": vol_ok,
        "Min_Profit_OK": min_ok,
        "Profit_Ceiling_OK": ceil_ok,
        "Vetoes_Passed": ",".join(vetoes) if isinstance(vetoes, list) else (vetoes or ""),
        "Regime": market_regime,
        "RegimeScore": regime_score,
        "MACD_Hist": macd_hist,
        "ADX15m": adx15,
        "RSI": rsi,
        "CCI": cci,
        "%B": pct_b,
        "Initial_Signal": alog.get("initial_signal"),
        "BB_Touch": alog.get("bb_touch"),
        "RSI_Extreme": alog.get("rsi_extreme"),
        "CCI_Extreme": alog.get("cci_extreme"),
        "Overshoot_OK": alog.get("overshoot_ok"),
        "EMA_Gap_ATR": alog.get("ema_gap_atr"),
        "Aligned": is_aligned,
        "Downgrade_Reason": alog.get("downgrade_reason") or "N/A",
    }

# =========================
# OUTCOME EVALUATION
# =========================
def first_touch_outcome(side: str, tp: float, sl: float, klines):
    """
    Evaluate first-touch TP/SL on 1m klines for the next HOURS_TO_CHECK hours.
    Returns tuple: (Outcome, duration_min, max_profit_price, max_drawdown_price, did_tp_later)
    """
    if not klines:
        return ("Inconclusive", 0, None, None, "No")

    max_high = float("-inf")
    min_low = float("inf")
    hit_tp_index = None
    hit_sl_index = None

    for i, k in enumerate(klines):
        high = float(k[2]); low = float(k[3])
        if high > max_high: max_high = high
        if low < min_low:  min_low = low

        if side == "Buy":
            if low <= sl and hit_sl_index is None:  hit_sl_index = i
            if high >= tp and hit_tp_index is None: hit_tp_index = i
        else:
            if high >= sl and hit_sl_index is None: hit_sl_index = i
            if low <= tp and hit_tp_index is None:  hit_tp_index = i

        if hit_tp_index is not None and hit_sl_index is not None:
            break

    if hit_tp_index is None and hit_sl_index is None:
        outcome = "Inconclusive"; duration = 0; did_tp_later = "No"
    elif hit_tp_index is not None and (hit_sl_index is None or hit_tp_index < hit_sl_index):
        outcome = "Success"; duration = hit_tp_index + 1; did_tp_later = "Yes"
    elif hit_sl_index is not None and (hit_tp_index is None or hit_sl_index < hit_tp_index):
        outcome = "Fail"; duration = hit_sl_index + 1
        did_tp_later = "Yes" if (hit_tp_index is not None and hit_tp_index > hit_sl_index) else "No"
    else:
        outcome = "Inconclusive"; duration = 0; did_tp_later = "No"

    max_profit_price = max_high if side == "Buy" else min_low
    max_drawdown_price = min_low if side == "Buy" else max_high
    return (outcome, duration, max_profit_price, max_drawdown_price, did_tp_later)

# =========================
# REPORTING (TXT)
# =========================
REPORT_COLUMNS = [
    "Coin", "Signal", "Confidence", "POP", "Outcome", "Duration(min)",
    "MaxProfitPrice", "MaxDrawdownPrice", "Did_TP_Hit_Later",
    "Buy_Score", "Sell_Score",
    "Base_Score_OK", "Num_Conf", "Confluence_OK", "Vol_Profile_OK",
    "Min_Profit_OK", "Profit_Ceiling_OK",
    "Regime", "RegimeScore", "MACD_Hist", "ADX15m",
    "RSI", "CCI", "%B", "Overshoot_OK",
    "BB_Touch", "RSI_Extreme", "CCI_Extreme",
    "Initial_Signal", "Aligned",
    "Vetoes_Passed",
    "Downgrade_Reason"
]

def format_header():
    return "  " + "  ".join([c.ljust(15) for c in REPORT_COLUMNS])

def format_row(row):
    def fmt(x):
        if x is None:
            return ""
        if isinstance(x, float):
            return f"{x:.6f}" if x > 1000 else f"{x:.4f}"
        return str(x)
    return "  " + "  ".join([fmt(row.get(c, "")).ljust(15) for c in REPORT_COLUMNS])

# =========================
# ANALYTICS CSV (NEW)
# =========================
CSV_COLUMNS = [
    "FileTimestampUTC","Coin","Signal","Outcome","Duration(min)",
    "Confidence","POP","Buy_Score","Sell_Score",
    "Num_Conf","Confluence_OK","Vol_Profile_OK",
    "Min_Profit_OK","Profit_Ceiling_OK",
    "Regime","RegimeScore","ADX15m","MACD_Hist",
    "RSI","CCI","PercentB","Overshoot_OK","Aligned",
    "Initial_Signal","Vetoes_Passed",
    "MaxProfitPrice","MaxDrawdownPrice","Did_TP_Hit_Later",
    "DeservedStrongBuy","DeservedStrongSell"
]

def compute_deserved_flags(row):
    """
    Analysis-only: who *should* have been Strong under queued tweaks?
    - Sell: Success AND Num_Conf>=2 AND (PercentB>=1.00 OR (ADX15m>=20 and MACD_Hist>0)) AND Confidence>=60
    - Buy : Success AND Num_Conf>=2 AND (PercentB<=-0.05 OR RSI<=28) AND Confidence>=60
    """
    try:
        outcome   = (row.get("Outcome") or "").strip()
        num_conf  = int(row.get("Num_Conf") or 0)
        pctb      = float(row.get("PercentB")) if row.get("PercentB") not in ("", None) else None
        adx       = float(row.get("ADX15m")) if row.get("ADX15m") not in ("", None) else None
        macd_hist = float(row.get("MACD_Hist")) if row.get("MACD_Hist") not in ("", None) else None
        rsi       = float(row.get("RSI")) if row.get("RSI") not in ("", None) else None
        conf      = float(row.get("Confidence")) if row.get("Confidence") not in ("", None) else 0.0
        sig       = (row.get("Signal") or "")

        ok_base   = (outcome == "Success" and num_conf >= 2 and conf >= 60)

        deserved_sell = False
        deserved_buy  = False

        if ok_base and "Sell" in sig:
            cond_os = (pctb is not None and pctb >= 1.00) or ((adx is not None and adx >= 20) and (macd_hist is not None and macd_hist > 0))
            deserved_sell = bool(cond_os)

        if ok_base and "Buy" in sig:
            cond_ob = ((pctb is not None and pctb <= -0.05) or (rsi is not None and rsi <= 28))
            deserved_buy = bool(cond_ob)

        return int(deserved_buy), int(deserved_sell)
    except Exception:
        return 0, 0

def row_for_csv(file_ts_utc_str, r):
    out = {
        "FileTimestampUTC": file_ts_utc_str,
        "Coin": r.get("Coin",""),
        "Signal": r.get("Signal",""),
        "Outcome": r.get("Outcome",""),
        "Duration(min)": r.get("Duration(min)",""),
        "Confidence": r.get("Confidence",""),
        "POP": r.get("POP",""),
        "Buy_Score": r.get("Buy_Score",""),
        "Sell_Score": r.get("Sell_Score",""),
        "Num_Conf": r.get("Num_Conf",""),
        "Confluence_OK": r.get("Confluence_OK",""),
        "Vol_Profile_OK": r.get("Vol_Profile_OK",""),
        "Min_Profit_OK": r.get("Min_Profit_OK",""),
        "Profit_Ceiling_OK": r.get("Profit_Ceiling_OK",""),
        "Regime": r.get("Regime",""),
        "RegimeScore": r.get("RegimeScore",""),
        "ADX15m": r.get("ADX15m",""),
        "MACD_Hist": r.get("MACD_Hist",""),
        "RSI": r.get("RSI",""),
        "CCI": r.get("CCI",""),
        "PercentB": r.get("%B",""),
        "Overshoot_OK": r.get("Overshoot_OK",""),
        "Aligned": r.get("Aligned",""),
        "Initial_Signal": r.get("Initial_Signal",""),
        "Vetoes_Passed": r.get("Vetoes_Passed",""),
        "MaxProfitPrice": r.get("MaxProfitPrice",""),
        "MaxDrawdownPrice": r.get("MaxDrawdownPrice",""),
        "Did_TP_Hit_Later": r.get("Did_TP_Hit_Later",""),
        "DeservedStrongBuy": "",
        "DeservedStrongSell": ""
    }
    b, s = compute_deserved_flags(out)
    out["DeservedStrongBuy"] = b
    out["DeservedStrongSell"] = s
    return out

# =========================
# CORE BACKTEST
# =========================
def build_row(entry, start_time_utc):
    """Extract fields, evaluate outcome, and return a single row dict."""
    label = entry.get("signal", "Neutral")
    if label == "Neutral":
        return None, None

    coin = entry.get("coin") or entry.get("symbol")
    if not coin:
        return None, None

    tp = entry.get("tp"); sl = entry.get("sl")
    if tp is None or sl is None:
        return None, None
    tp = float(tp); sl = float(sl)

    side = "Buy" if "Buy" in label else "Sell"
    klines = fetch_future_klines_1m(coin, start_time_utc, minutes=HOURS_TO_CHECK*60)
    outcome, duration_min, max_profit_px, max_drawdown_px, did_tp_later = first_touch_outcome(side, tp, sl, klines)

    f = extract_fields(entry)
    row = {
        "Coin": coin,
        "Signal": f["Signal"],
        "Confidence": int(round(float(f["Confidence"]))) if f["Confidence"] not in (None, "") else "",
        "POP": f["POP"],
        "Outcome": outcome,
        "Duration(min)": duration_min,
        "MaxProfitPrice": max_profit_px,
        "MaxDrawdownPrice": max_drawdown_px,
        "Did_TP_Hit_Later": did_tp_later,
        "Buy_Score": f["Buy_Score"],
        "Sell_Score": f["Sell_Score"],
        "Base_Score_OK": f["Base_Score_OK"],
        "Num_Conf": f["Num_Conf"],
        "Confluence_OK": f["Confluence_OK"],
        "Vol_Profile_OK": f["Vol_Profile_OK"],
        "Min_Profit_OK": f["Min_Profit_OK"],
        "Profit_Ceiling_OK": f["Profit_Ceiling_OK"],
        "Regime": f["Regime"],
        "RegimeScore": f["RegimeScore"],
        "MACD_Hist": f["MACD_Hist"],
        "ADX15m": f["ADX15m"],
        "RSI": f["RSI"],
        "CCI": f["CCI"],
        "%B": round(f["%B"], 4) if isinstance(f["%B"], (int, float)) else "",
        "Overshoot_OK": f["Overshoot_OK"],
        "BB_Touch": f["BB_Touch"],
        "RSI_Extreme": f["RSI_Extreme"],
        "CCI_Extreme": f["CCI_Extreme"],
        "Initial_Signal": f["Initial_Signal"],
        "Aligned": f["Aligned"],
        "Vetoes_Passed": f["Vetoes_Passed"],
        "Downgrade_Reason": f["Downgrade_Reason"],
    }
    return label, row

def backtest_file(json_url: str):
    """Download one JSON file, evaluate all signals, and write TXT + CSV."""
    filename = json_url.split("/")[-1]
    print(f"\n===== BACKTEST REPORT FOR: {filename} =====")

    # download JSON
    try:
        r = requests.get(json_url, proxies=proxies, timeout=40)
        r.raise_for_status()
        signals = r.json()
    except Exception as e:
        print(f"[ERROR] Could not download/parse JSON: {e}")
        return None

    start_time_utc = parse_timestamp_from_filename(filename)

    rows_by_type = {"Strong Buy": [], "Buy": [], "Strong Sell": [], "Sell": []}

    for entry in signals:
        try:
            label, row = build_row(entry, start_time_utc)
            if not label:
                continue
            rows_by_type[label].append(row)
        except Exception as e:
            print(f"[WARN] Skipped entry due to error: {e}")

    # ---- NEW: write per-file CSV analytics ----
    ensure_dir(ANALYTICS_DIR)
    file_ts_str = start_time_utc.strftime("%Y-%m-%d %H:%M:%S")
    csv_rows = []
    for key in ["Strong Buy","Buy","Strong Sell","Sell"]:
        for r in rows_by_type[key]:
            csv_rows.append(row_for_csv(file_ts_str, r))

    csv_name = f"backtest_{start_time_utc.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    csv_path = os.path.join(ANALYTICS_DIR, csv_name)
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for rr in csv_rows:
                writer.writerow(rr)
        print(f"[OK] Analytics CSV saved: {csv_path}")
    except Exception as e:
        print(f"[WARN] Could not write analytics CSV: {e}")

    # ---- TXT report (as before) ----
    ensure_dir(REPORTS_FOLDER)
    out_path = os.path.join(REPORTS_FOLDER, f"backtest_{start_time_utc.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"===== BACKTEST REPORT FOR: {filename} =====\n")
        f.write(f"Signal Generation Time (UTC): {start_time_utc.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for section in ["STRONG BUY SIGNALS", "BUY SIGNALS", "STRONG SELL SIGNALS", "SELL SIGNALS"]:
            f.write(f"--- {section} ---\n")
            key = "Strong Buy" if "STRONG BUY" in section else \
                  "Strong Sell" if "STRONG SELL" in section else \
                  "Buy" if section == "BUY SIGNALS" else "Sell"
            rows = rows_by_type[key]
            if rows:
                f.write(format_header() + "\n")
                for r in rows:
                    f.write(format_row(r) + "\n")
            else:
                f.write("(None)\n")
            f.write("\n")

    print(f"[OK] TXT report saved: {out_path}")
    return out_path

# =========================
# MAIN
# =========================
def main():
    ensure_dir(REPORTS_FOLDER)
    ensure_dir(ANALYTICS_DIR)

    urls = get_github_archive_urls()
    if not urls:
        print("[INFO] No JSON files found on GitHub.")
        return

    print(f"Found {len(urls)} JSON files in repo. Backtesting {HOURS_TO_CHECK}h horizon...")
    for url in urls:
        backtest_file(url)

if __name__ == "__main__":
    main()
