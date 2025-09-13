#!/usr/bin/env python3
"""
strongoutcome.py

Simulate outcomes for strong signals stored in analytics/TotalStrong_*.csv.

- Fixed TP/SL simulation: TP = 1% price move, SL = 1% price move (on price)
- Trailing SL simulation: no TP; trailing SL distance = 1.5 * ATR (ATR computed on 5m candles prior to signal)
- Uses 1m klines for lookahead and hit detection (LOOKAHEAD_MINUTES)
- Writes analytics/TotalStrong_outcomes_<IST-timestamp>.csv
"""

import os
import glob
import json
import time
import math
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import pytz

# ---------------- CONFIG ----------------
ANALYTICS_DIR = "analytics"
BINANCE_FAPI = "https://fapi.binance.com"

# Proxy (copy your proxy config here / adjust)
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url} if "YOUR_IP" not in PROXY_IP else None

# lookahead & klines
LOOKAHEAD_MINUTES = 600
LOOKAHEAD_MS = LOOKAHEAD_MINUTES * 60 * 1000
KLINE_LIMIT = 1000
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 0.8

# Fixed TP/SL (price-move percent)
TP_PRICE_MOVE_PCT = 0.01   # 1.0%
SL_PRICE_MOVE_PCT = 0.01   # 1.0%

# Trailing SL config
TRAIL_ATR_MULTIPLIER = 1.5
ATR_PERIOD = 14   # ATR periods on 5m candles

# Money / fees
MARGIN_USDT = 10.0
LEVERAGE = 7.0
FEE_RATIO = 0.001   # 0.1% on gross profit (reduced from profit; added to loss)

# Fetch intervals
KL_INTERVAL_1M = "1m"
KL_INTERVAL_5M = "5m"

os.makedirs(ANALYTICS_DIR, exist_ok=True)

# ---------------- Helpers ----------------
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
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        try:
            return int(pd.to_datetime(ts).tz_localize('UTC').timestamp() * 1000)
        except Exception:
            return None

def utc_to_ist_iso(utc_iso):
    try:
        dt = datetime.fromisoformat(utc_iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = pytz.timezone("Asia/Kolkata")
        return dt.astimezone(ist).isoformat()
    except Exception:
        return ""

def fetch_binance_klines(symbol, interval, startTime=None, endTime=None, limit=KLINE_LIMIT):
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
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

def calc_atr_from_ohlc(highs, lows, closes, period=14):
    if len(highs) < 2:
        return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    try:
        atr = float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1])
        return atr
    except Exception:
        try:
            return float(tr.tail(period).mean())
        except Exception:
            return None

def net_profit_after_fee(gross_profit):
    if gross_profit is None:
        return None
    if gross_profit == 0:
        return 0.0
    fee = abs(gross_profit) * FEE_RATIO
    return gross_profit - math.copysign(fee, gross_profit)

# ---------------- Simulation helpers ----------------
def simulate_fixed_tp_sl(entry_price, tp_price, sl_price, klines, is_buy=True):
    """
    Use 1m klines to detect first TP or SL hit. Conservative: treat TP/SL hit when candle extreme crosses target,
    and set exit price to the TP/SL price (not candle extreme).
    """
    tp_hit_time = None
    sl_hit_time = None
    for k in klines:
        h = k["high"]; l = k["low"]; t = k["close_time"]
        if is_buy:
            if tp_price is not None and h >= tp_price and tp_hit_time is None:
                tp_hit_time = t
            if sl_price is not None and l <= sl_price and sl_hit_time is None:
                sl_hit_time = t
        else:
            if tp_price is not None and l <= tp_price and tp_hit_time is None:
                tp_hit_time = t
            if sl_price is not None and h >= sl_price and sl_hit_time is None:
                sl_hit_time = t

    first_hit = None
    exit_price = None
    exit_time = None
    if tp_hit_time and sl_hit_time:
        if tp_hit_time <= sl_hit_time:
            first_hit = "TP"
            exit_time = tp_hit_time; exit_price = tp_price
        else:
            first_hit = "SL"
            exit_time = sl_hit_time; exit_price = sl_price
    elif tp_hit_time:
        first_hit = "TP"; exit_time = tp_hit_time; exit_price = tp_price
    elif sl_hit_time:
        first_hit = "SL"; exit_time = sl_hit_time; exit_price = sl_price
    else:
        first_hit = None
        last = klines[-1]
        exit_time = last["close_time"]; exit_price = last["close"]

    outcome = "Inconclusive"
    if first_hit == "TP":
        outcome = "Success"
    elif first_hit == "SL":
        outcome = "Fail"

    # compute gross profit on position notional = margin * leverage
    try:
        if is_buy:
            price_diff = (exit_price - entry_price) / entry_price
        else:
            price_diff = (entry_price - exit_price) / entry_price
        position_notional = MARGIN_USDT * LEVERAGE
        gross_profit = position_notional * price_diff
        profit_pct_on_margin = price_diff * LEVERAGE * 100.0
    except Exception:
        gross_profit = None; profit_pct_on_margin = None

    net_profit = net_profit_after_fee(gross_profit) if gross_profit is not None else None
    return {
        "FixedOutcome": outcome,
        "FixedExitPrice": exit_price,
        "FixedExitTime": exit_time,
        "FixedGrossProfitUSDT": round(gross_profit, 8) if gross_profit is not None else None,
        "FixedNetProfitUSDT": round(net_profit, 8) if net_profit is not None else None,
        "FixedProfitPctOnMargin": round(profit_pct_on_margin, 6) if profit_pct_on_margin is not None else None,
    }

def simulate_trailing_sl(entry_price, klines, is_buy=True, atr=None, trail_atr_mult=1.5):
    """
    Trailing SL simulation using 1m klines; trail distance = trail_atr_mult * ATR (ATR in price units).
    We update trail_stop conservatively by using candle extremes.
    Exit when candle extreme breaches trail_stop; exit_price set to trail_stop (conservative).
    """
    if atr is None or atr == 0:
        return {
            "TrailOutcome": "Inconclusive",
            "TrailExitPrice": None,
            "TrailExitTime": None,
            "TrailGrossProfitUSDT": None,
            "TrailNetProfitUSDT": None,
            "TrailProfitPctOnMargin": None,
            "TrailMaxFavorableMovePct": None
        }
    trail_distance = trail_atr_mult * atr
    max_price = entry_price
    min_price = entry_price
    if is_buy:
        trail_stop = entry_price - trail_distance
    else:
        trail_stop = entry_price + trail_distance

    exited = False
    exit_price = None
    exit_time = None
    max_fav_move_pct = 0.0

    for k in klines:
        h = k["high"]; l = k["low"]; t = k["close_time"]
        if is_buy:
            if h > max_price:
                max_price = h
            new_trail = max(trail_stop, max_price - trail_distance)
            trail_stop = new_trail
            fav_move_pct = (max_price - entry_price) / entry_price
            if fav_move_pct > max_fav_move_pct:
                max_fav_move_pct = fav_move_pct
            if l <= trail_stop:
                exited = True
                exit_price = trail_stop
                exit_time = t
                break
        else:
            if l < min_price:
                min_price = l
            new_trail = min(trail_stop, min_price + trail_distance)
            trail_stop = new_trail
            fav_move_pct = (entry_price - min_price) / entry_price
            if fav_move_pct > max_fav_move_pct:
                max_fav_move_pct = fav_move_pct
            if h >= trail_stop:
                exited = True
                exit_price = trail_stop
                exit_time = t
                break

    if not exited:
        last = klines[-1]
        exit_time = last["close_time"]
        exit_price = last["close"]
        outcome = "Inconclusive"
    else:
        outcome = "Success" if ((is_buy and exit_price > entry_price) or (not is_buy and exit_price < entry_price)) else "Fail"

    try:
        if is_buy:
            price_diff = (exit_price - entry_price) / entry_price
        else:
            price_diff = (entry_price - exit_price) / entry_price
        position_notional = MARGIN_USDT * LEVERAGE
        gross_profit = position_notional * price_diff
        profit_pct_on_margin = price_diff * LEVERAGE * 100.0
    except Exception:
        gross_profit = None; profit_pct_on_margin = None

    net_profit = net_profit_after_fee(gross_profit) if gross_profit is not None else None

    return {
        "TrailOutcome": outcome,
        "TrailExitPrice": exit_price,
        "TrailExitTime": exit_time,
        "TrailGrossProfitUSDT": round(gross_profit, 8) if gross_profit is not None else None,
        "TrailNetProfitUSDT": round(net_profit, 8) if net_profit is not None else None,
        "TrailProfitPctOnMargin": round(profit_pct_on_margin, 6) if profit_pct_on_margin is not None else None,
        "TrailMaxFavorableMovePct": round(max_fav_move_pct * 100.0, 6) if max_fav_move_pct is not None else None
    }

# ---------------- Discovery ----------------
def find_totalstrong_csv(input_path=None):
    if input_path:
        if os.path.exists(input_path):
            return input_path
        raise FileNotFoundError(f"{input_path} not found")
    files = sorted(glob.glob(os.path.join(ANALYTICS_DIR, "TotalStrong*.csv")), reverse=True)
    if not files:
        raise FileNotFoundError("No TotalStrong_*.csv found in analytics/")
    return files[0]

# ---------------- Main ----------------
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", help="Path to TotalStrong CSV (optional). If not provided newest analytics/TotalStrong_*.csv is used.")
    args = p.parse_args(argv)

    try:
        csv_path = find_totalstrong_csv(args.input)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print(f"[INFO] Loading strong signals from: {csv_path}")
    df = pd.read_csv(csv_path, dtype=object)

    def col_ci(name_options):
        for opt in name_options:
            for c in df.columns:
                if c.lower() == opt.lower():
                    return c
        return None

    col_coin = col_ci(["coin", "Coin"])
    col_price = col_ci(["price", "Price"])
    col_sigtime = col_ci(["signal_time_utc", "signal_time", "timestamp", "time"])
    col_signal = col_ci(["signal", "Signal"])
    col_conf = col_ci(["confidence", "Confidence"])

    if not col_coin or not col_price or not col_sigtime or not col_signal:
        print("[ERROR] Input CSV missing required columns. Found columns:", df.columns.tolist())
        return

    rows_out = []
    totals = {
        "fixed_net": 0.0, "trail_net": 0.0,
        "fixed_success": 0, "fixed_fail": 0, "fixed_incon": 0,
        "trail_success": 0, "trail_fail": 0, "trail_incon": 0
    }

    for idx, r in df.iterrows():
        try:
            coin = r[col_coin]
            entry_price = None
            try:
                entry_price = float(r[col_price])
            except Exception:
                entry_price = None
            sig_time = r[col_sigtime]
            signal_label = str(r[col_signal]) if r[col_signal] is not None else ""
            confidence = r[col_conf] if col_conf else None

            if entry_price is None:
                rows_out.append({"coin": coin, "note": "missing entry price"})
                continue

            is_buy = "buy" in signal_label.lower()

            # fixed TP/SL levels (1% / 1%)
            tp_price = entry_price * (1.0 + TP_PRICE_MOVE_PCT) if is_buy else entry_price * (1.0 - TP_PRICE_MOVE_PCT)
            sl_price = entry_price * (1.0 - SL_PRICE_MOVE_PCT) if is_buy else entry_price * (1.0 + SL_PRICE_MOVE_PCT)

            # times and klines
            start_ms = ms_from_iso(sig_time)
            if start_ms is None:
                try:
                    start_ms = int(pd.to_datetime(sig_time).tz_localize('UTC').timestamp() * 1000)
                except Exception:
                    start_ms = None
            if start_ms is None:
                rows_out.append({"coin": coin, "signal_time": sig_time, "note": "invalid signal_time"})
                continue
            end_ms = start_ms + LOOKAHEAD_MS

            # fetch 1m klines for lookahead
            try:
                kl_1m = fetch_binance_klines(coin, KL_INTERVAL_1M, startTime=start_ms, endTime=end_ms, limit=KLINE_LIMIT)
            except Exception as e:
                rows_out.append({"coin": coin, "signal_time": sig_time, "note": f"1m klines fetch failed: {e}"})
                continue
            if not kl_1m:
                rows_out.append({"coin": coin, "signal_time": sig_time, "note": "no 1m klines"})
                continue

            # compute ATR on 5m pre-entry window (prefer pre-entry; else use portion of 1m clipped into 5m)
            atr_val = None
            try:
                pre_end = start_ms - 1
                pre_limit = max(ATR_PERIOD * 3, 100)
                pre_5m = fetch_binance_klines(coin, KL_INTERVAL_5M, endTime=pre_end, limit=pre_limit)
                if pre_5m and len(pre_5m) >= ATR_PERIOD:
                    highs = [k["high"] for k in pre_5m]
                    lows = [k["low"] for k in pre_5m]
                    closes = [k["close"] for k in pre_5m]
                else:
                    # fallback: aggregate 1m klines into 5m buckets for earliest available part
                    take = min(len(kl_1m), ATR_PERIOD * 5)
                    # use earliest portion from kl_1m (pre-entry not available)
                    sample_1m = kl_1m[:take]
                    # build 5m candles by grouping every 5 bars
                    highs=[]; lows=[]; closes=[]
                    for i in range(0, len(sample_1m), 5):
                        block = sample_1m[i:i+5]
                        if not block:
                            continue
                        highs.append(max(x["high"] for x in block))
                        lows.append(min(x["low"] for x in block))
                        closes.append(block[-1]["close"])
                atr_val = calc_atr_from_ohlc(highs, lows, closes, period=ATR_PERIOD)
            except Exception:
                atr_val = None

            # Fixed TP/SL simulation
            fixed_res = simulate_fixed_tp_sl(entry_price, tp_price, sl_price, kl_1m, is_buy=is_buy)
            try:
                if fixed_res.get("FixedExitTime") is not None:
                    fixed_res["FixedTimeToExitMin"] = int((int(fixed_res["FixedExitTime"]) - start_ms) / 60000)
            except Exception:
                fixed_res["FixedTimeToExitMin"] = None

            # Trailing SL simulation (no TP)
            trail_res = simulate_trailing_sl(entry_price, kl_1m, is_buy=is_buy, atr=atr_val, trail_atr_mult=TRAIL_ATR_MULTIPLIER)
            try:
                if trail_res.get("TrailExitTime") is not None:
                    trail_res["TrailTimeToExitMin"] = int((int(trail_res["TrailExitTime"]) - start_ms) / 60000)
            except Exception:
                trail_res["TrailTimeToExitMin"] = None

            # accumulate totals
            if fixed_res.get("FixedNetProfitUSDT") is not None:
                totals["fixed_net"] += float(fixed_res["FixedNetProfitUSDT"])
            if trail_res.get("TrailNetProfitUSDT") is not None:
                totals["trail_net"] += float(trail_res["TrailNetProfitUSDT"])

            if fixed_res.get("FixedOutcome") == "Success":
                totals["fixed_success"] += 1
            elif fixed_res.get("FixedOutcome") == "Fail":
                totals["fixed_fail"] += 1
            else:
                totals["fixed_incon"] += 1

            if trail_res.get("TrailOutcome") == "Success":
                totals["trail_success"] += 1
            elif trail_res.get("TrailOutcome") == "Fail":
                totals["trail_fail"] += 1
            else:
                totals["trail_incon"] += 1

            out = {
                "source_file": os.path.basename(csv_path),
                "coin": coin,
                "signal": signal_label,
                "confidence": confidence,
                "entry_price": entry_price,
                "signal_time_utc": sig_time,
                "signal_time_ist": utc_to_ist_iso(sig_time) if sig_time else "",
                "is_buy": is_buy,
                "tp_price": round(tp_price, 12),
                "sl_price": round(sl_price, 12),
                "atr_used_5m": round(atr_val, 12) if atr_val is not None else None,
                "trail_distance": round(TRAIL_ATR_MULTIPLIER * atr_val, 12) if atr_val is not None else None
            }
            out.update(fixed_res)
            out.update(trail_res)
            rows_out.append(out)

        except Exception as e:
            rows_out.append({"coin": r.get(col_coin, None), "note": f"processing error: {e}"})
            continue

    # write CSV
    if rows_out:
        ist = datetime.now(pytz.timezone("Asia/Kolkata"))
        stamp = ist.strftime("%Y-%m-%d_%H-%M-%S")
        outname = f"TotalStrong_outcomes_{stamp}.csv"
        outpath = os.path.join(ANALYTICS_DIR, outname)
        pd.DataFrame(rows_out).to_csv(outpath, index=False)
        print(f"[OK] Wrote outcomes to {outpath}")
    else:
        print("[INFO] No strong signals processed.")

    # summary
    print("SUMMARY:")
    print(f" Fixed: total_net={totals['fixed_net']:.6f} USDT  successes={totals['fixed_success']} fails={totals['fixed_fail']} inconclusive={totals['fixed_incon']}")
    print(f" Trail: total_net={totals['trail_net']:.6f} USDT  successes={totals['trail_success']} fails={totals['trail_fail']} inconclusive={totals['trail_incon']}")

if __name__ == "__main__":
    main()
