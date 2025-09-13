#!/usr/bin/env python3
# strongoutcome.py
# Backtest strong signals only with fixed TP/SL and trailing SL logic

import os
import json
import math
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# -------- CONFIG --------
DATA_FILE = os.path.join("analytics", "TotalStrong_*.csv")
OUT_DETAILED = os.path.join("analytics", "strongoutcome_detailed.csv")
OUT_SUMMARY = os.path.join("analytics", "strongoutcome_summary.csv")

# Proxy
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

BINANCE_FAPI = "https://fapi.binance.com"

# Trading assumptions
MARGIN = 10.0        # USDT
LEVERAGE = 7
FEE_RATE = 0.001     # 0.1% entry+exit combined

# Backtest config
LOOKAHEAD_MINUTES = 600
LOOKAHEAD_MS = LOOKAHEAD_MINUTES * 60 * 1000
KLINE_LIMIT = 1000

# -------- HELPERS --------
def request_with_retries(url, params=None, proxies_local=proxies, timeout=30, retries=3, backoff=0.8):
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, proxies=proxies_local, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            import time; time.sleep(backoff * (1 + attempt))
    raise last_exc

def ms_from_iso(ts):
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None

def fetch_klines(symbol, interval="1m", start=None, end=None, limit=KLINE_LIMIT):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start: params["startTime"] = int(start)
    if end: params["endTime"] = int(end)
    data = request_with_retries(url, params=params).json()
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

def calc_atr(highs, lows, closes, period=14):
    import pandas as pd
    if len(highs) < period + 1: return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def pct_to_usdt(pct, entry_price):
    """Convert % move to P/L in USDT (with leverage, margin, fees)."""
    gross = MARGIN * (pct/100.0) * LEVERAGE
    fee = MARGIN * FEE_RATE
    return gross - fee

# -------- SIMULATIONS --------
def simulate_fixed(entry, tp_pct, sl_pct, klines, direction):
    tp_hit = sl_hit = None
    tp_price = entry * (1 + tp_pct/100) if direction=="buy" else entry * (1 - tp_pct/100)
    sl_price = entry * (1 - sl_pct/100) if direction=="buy" else entry * (1 + sl_pct/100)
    for k in klines:
        if direction=="buy":
            if k["high"] >= tp_price:
                tp_hit = k["close_time"]; break
            if k["low"] <= sl_price:
                sl_hit = k["close_time"]; break
        else:
            if k["low"] <= tp_price:
                tp_hit = k["close_time"]; break
            if k["high"] >= sl_price:
                sl_hit = k["close_time"]; break
    if tp_hit:
        return "Success", tp_price, pct_to_usdt(tp_pct, entry)
    elif sl_hit:
        return "Fail", sl_price, pct_to_usdt(-sl_pct, entry)
    else:
        return "Inconclusive", klines[-1]["close"], pct_to_usdt(((klines[-1]["close"]-entry)/entry*100)*(1 if direction=="buy" else -1), entry)

def simulate_trailing(entry, atr, klines, direction):
    sl = entry - (atr*1.0) if direction=="buy" else entry + (atr*1.0)
    for k in klines:
        if direction=="buy":
            if k["low"] <= sl:
                return "Fail", sl, pct_to_usdt(((sl-entry)/entry)*100, entry)
            if k["close"] > entry:
                sl = max(sl, k["close"] - atr)
        else:
            if k["high"] >= sl:
                return "Fail", sl, pct_to_usdt(((entry-sl)/entry)*100, entry)
            if k["close"] < entry:
                sl = min(sl, k["close"] + atr)
    # survived entire window
    final = klines[-1]["close"]
    pct = ((final-entry)/entry*100) if direction=="buy" else ((entry-final)/entry*100)
    return "Inconclusive", final, pct_to_usdt(pct, entry)

# -------- MAIN --------
def main():
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] {DATA_FILE} not found")
        return
    df = pd.read_csv(DATA_FILE)
    rows = []
    for _, row in df.iterrows():
        try:
            coin = row["coin"]
            sig = row["signal"]
            entry = float(row["price"])
            start = ms_from_iso(row["signal_time_utc"])
            if not start: continue
            end = start + LOOKAHEAD_MS
            klines = fetch_klines(coin, "1m", start, end)
            if not klines: continue
            # ATR from 5m candles
            k5 = fetch_klines(coin, "5m", start-3600*1000, start)  # 1h warmup
            atr = calc_atr([x["high"] for x in k5], [x["low"] for x in k5], [x["close"] for x in k5]) or (entry*0.002)

            direction = "buy" if "buy" in sig.lower() else "sell"

            # Fixed 0.72/3
            res1, exit1, pl1 = simulate_fixed(entry, 0.72, 3.0, klines, direction)
            # Fixed 1/2
            res2, exit2, pl2 = simulate_fixed(entry, 1.0, 2.0, klines, direction)
            # Trailing 1xATR
            res3, exit3, pl3 = simulate_trailing(entry, atr, klines, direction)

            rows.append({
                "coin": coin, "signal": sig, "entry": entry,
                "fixed_072_3_outcome": res1, "fixed_072_3_exit": exit1, "fixed_072_3_PL": pl1,
                "fixed_1_2_outcome": res2, "fixed_1_2_exit": exit2, "fixed_1_2_PL": pl2,
                "trailing_outcome": res3, "trailing_exit": exit3, "trailing_PL": pl3
            })
        except Exception as e:
            print(f"[WARN] Error on row: {e}")

    outdf = pd.DataFrame(rows)
    outdf.to_csv(OUT_DETAILED, index=False)

    # summary
    summ = []
    for col in ["fixed_072_3_outcome","fixed_1_2_outcome","trailing_outcome"]:
        wins = (outdf[col]=="Success").sum()
        fails = (outdf[col]=="Fail").sum()
        summ.append({
            "Strategy": col,
            "Total": len(outdf),
            "Wins": wins, "Fails": fails,
            "WinRate%": (wins/(wins+fails)*100) if (wins+fails)>0 else 0,
            "Total_PL": outdf[col.replace("outcome","PL")].sum()
        })
    pd.DataFrame(summ).to_csv(OUT_SUMMARY, index=False)
    print(f"[OK] Wrote {OUT_DETAILED} and {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
