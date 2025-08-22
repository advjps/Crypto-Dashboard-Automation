# run_automation.py (V13 – 7th Amendment)
import pandas as pd
import requests
import json
from datetime import datetime
import time
import os
import math
import pytz

# ============== PROXY CONFIG ==============
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"
proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = { "http": proxy_url, "https": proxy_url } if "YOUR_IP" not in PROXY_IP else None

# ============== GENERAL CONFIG ==============
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"
TOP_LIMIT = 70
BINANCE_FAPI = "https://fapi.binance.com"

# --- Profit evaluation basis (ROI on margin) ---
LEVERAGE_FOR_PROFIT_EVAL = 7.0       # fixed leverage used for estimated_profit (ROI on margin)
MIN_PROFIT_MARGIN = 2.0              # min % on margin to pass
PROFIT_CEILING_MARGIN = None         # 8th Amendment: ceiling veto removed (not used)

# --- Strong confidence thresholds (adaptive by regime/side) ---
STRONG_CONF_THRESHOLDS = {
    "Bearish_Sell": 70,   # lowered to promote more high-quality sells in bear regimes
    "Bullish_Buy": 78,    # buys stricter in bull regimes
    "Neutral_Sell": 75,
    "Neutral_Buy": 80
}

# --- ATR-based TP/SL clamps (as % of price) - default band ---
TP_PCT_MIN, TP_PCT_MAX = 0.008, 0.016   # 0.8% .. 1.6%
SL_PCT_MIN, SL_PCT_MAX = 0.008, 0.020   # 0.8% .. 2.0%

# --- Adaptive widening for extreme RSI contexts (keeps 1:1 R:R) ---
# Applied only when: (Sell & RSI>=70 & upper-band/%B confirm & MACD<=0 or ADX>=20)
#                 or (Buy  & RSI<=30 & lower-band/%B confirm & MACD>=0 or ADX>=20)
WIDEN_MIN_PCT = 0.012   # 1.2%
WIDEN_MAX_PCT = 0.022   # 2.2%

# --- Regime hysteresis (stickiness) ---
REGIME_HOLD_MINUTES = 60
REGIME_CONFIRM_BARS = 2

# ============== INDICATORS ==============
def calc_ema(values, period):
    if not isinstance(values, list) or len(values) < period: return [None]*len(values)
    return pd.Series(values).ewm(span=period, adjust=False).mean().tolist()

def calc_rsi(values, period=14):
    if not isinstance(values, list) or len(values) < period + 1: return [None]*len(values)
    s = pd.Series(values)
    d = s.diff()
    up = (d.where(d > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.where(d < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn
    return (100 - (100 / (1 + rs))).tolist()

def get_last_valid_value(values):
    for v in reversed(values):
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
    return None

def calc_macd(values, fast=12, slow=26, signal=9):
    s = pd.Series(values)
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig  = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return {'macd': float(macd.iloc[-1]), 'signal': float(sig.iloc[-1]), 'histogram': float(hist.iloc[-1])}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period: return {'upper': None, 'middle': None, 'lower': None}
    s = pd.Series(values)
    m = s.rolling(window=period).mean().iloc[-1]
    sd = s.rolling(window=period).std().iloc[-1]
    return {'upper': float(m + mult*sd), 'middle': float(m), 'lower': float(m - mult*sd)}

def calc_atr(highs, lows, closes, period=14):
    if len(highs) < period + 1: return None
    df = pd.DataFrame({'high': highs, 'low': lows, 'close': closes})
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1])

def calc_cci(highs, lows, closes, period=20):
    if len(highs) < period: return None
    tp = pd.Series([(h + l + c)/3 for h, l, c in zip(highs, lows, closes)])
    ma = tp.rolling(window=period).mean().iloc[-1]
    md = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean()).iloc[-1]
    if md == 0: return 0.0
    return float((tp.iloc[-1] - ma) / (0.015 * md))

def calc_adx(highs, lows, closes, period=14):
    # Lightweight ADX (Wilder)
    import numpy as np
    if len(highs) < period + 2: return None
    H, L, C = pd.Series(highs), pd.Series(lows), pd.Series(closes)
    up_move = H.diff()
    dn_move = -L.diff()
    plus_dm = up_move.where((up_move > dn_move) & (up_move > 0), 0.0)
    minus_dm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0.0)
    tr1 = H - L
    tr2 = (H - C.shift()).abs()
    tr3 = (L - C.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9) ) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return float(adx.iloc[-1])

# ============== DATA FETCH ==============
def fetch_top_volume_coins(limit=TOP_LIMIT):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
        r = requests.get(url, proxies=proxies, timeout=30)
        r.raise_for_status()
        data = r.json()
        usdt = [t for t in data if 'symbol' in t and str(t['symbol']).endswith('USDT')]
        return [c['symbol'] for c in sorted(usdt, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]]
    except Exception as e:
        print(f"Error fetching top coins: {e}")
        return []

def fetch_binance_klines(symbol, interval='5m', limit=200):
    try:
        url = f"{BINANCE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        r = requests.get(url, proxies=proxies, timeout=30)
        r.raise_for_status()
        data = r.json()
        return [
            {"open": float(d[1]), "high": float(d[2]), "low": float(d[3]),
             "close": float(d[4]), "volume": float(d[5])}
            for d in data
        ]
    except Exception as e:
        print(f"  - Could not fetch {symbol} {interval}: {e}")
        return []

def fetch_funding_rate_avg(symbols, limit=1):
    # average of latest funding rates across a slice of symbols
    vals = []
    for s in symbols[:20]:  # cap to 20 to be gentle
        try:
            url = f"{BINANCE_FAPI}/fapi/v1/fundingRate?symbol={s}&limit={limit}"
            r = requests.get(url, proxies=proxies, timeout=20)
            if r.status_code == 200:
                js = r.json()
                if js:
                    vals.append(float(js[-1].get("fundingRate", 0.0)))
            time.sleep(0.05)
        except Exception:
            pass
    if not vals:
        return 0.0
    return float(sum(vals)/len(vals))

# ============== HELPER: %B ==============
def percent_b(price, boll):
    try:
        lower = float(boll["lower"]); upper = float(boll["upper"])
        rng = (upper - lower) if (upper - lower) != 0 else 1e-9
        return float((price - lower) / rng)
    except Exception:
        return None

# ============== REGIME DETECTOR ==============
_last_regime = {"regime":"Neutral", "score":0.0, "ts":0}

def compute_market_regime(btc5, btc15, btc1h, breadth_snap, funding_avg):
    """
    Votes:
      +2/-2 : EMA stack on 15m+1h (20>50>100 bullish or bearish)
      +1/-1 : ADX(15m) >= 20 in same direction
      +1/-1 : Breadth: % of top coins above 5m EMA20
      +1/-1 : Funding avg (pos bullish / neg bearish)
    Threshold: score >= +3 -> Bullish, <= -3 -> Bearish, else Neutral.
    Hysteresis: require confirmation for REGIME_CONFIRM_BARS bars; hold for REGIME_HOLD_MINUTES.
    """
    score = 0.0
    comp = {"ema_stack":0.0, "adx15m":0.0, "breadth":0.0, "funding":0.0}

    # EMA stack 15m + 1h (20 vs 50 vs 100)
    def ema_stack_vote(data):
        closes = [d["close"] for d in data]
        ema20 = get_last_valid_value(calc_ema(closes, 20))
        ema50 = get_last_valid_value(calc_ema(closes, 50))
        ema100= get_last_valid_value(calc_ema(closes, 100))
        if None in (ema20, ema50, ema100): return 0
        if ema20 > ema50 > ema100: return +1
        if ema20 < ema50 < ema100: return -1
        return 0

    v15 = ema_stack_vote(btc15)
    v1h = ema_stack_vote(btc1h)
    ema_vote = v15 + v1h
    if ema_vote > 0: score += 2.0; comp["ema_stack"]=+2.0
    elif ema_vote < 0: score -= 2.0; comp["ema_stack"]=-2.0

    # ADX on 15m
    h15 = [d["high"] for d in btc15]; l15 = [d["low"] for d in btc15]; c15 = [d["close"] for d in btc15]
    adx15 = calc_adx(h15, l15, c15, 14)
    if adx15 is not None and adx15 >= 20:
        # direction by ema_vote sign
        if ema_vote > 0: score += 1.0; comp["adx15m"]=+1.0
        elif ema_vote < 0: score -= 1.0; comp["adx15m"]=-1.0

    # Breadth: % of coins above 5m EMA20
    breadth_pct = 0.0
    if breadth_snap:
        above = 0; total = 0
        for x in breadth_snap:
            closes = x.get("closes", [])
            ema20c = get_last_valid_value(calc_ema(closes, 20)) if closes else None
            if ema20c is not None and len(closes)>0:
                total += 1
                if closes[-1] >= ema20c: above += 1
        if total>0:
            breadth_pct = (above/total)*100.0
            if breadth_pct >= 60: score += 1.0; comp["breadth"]=+1.0
            elif breadth_pct <= 40: score -= 1.0; comp["breadth"]=-1.0

    # Funding average
    if funding_avg is not None:
        if funding_avg > 0: score += 1.0; comp["funding"]=+1.0
        elif funding_avg < 0: score -= 1.0; comp["funding"]=-1.0

    # Decide raw regime
    raw = "Neutral"
    if score >= 3.0: raw = "Bullish"
    elif score <= -3.0: raw = "Bearish"

    # Hysteresis / stickiness
    now_min = int(time.time()//60)
    global _last_regime
    hold_ok = (now_min - _last_regime.get("ts",0)) >= REGIME_HOLD_MINUTES
    if raw != _last_regime.get("regime") and not hold_ok:
        # Don't flip early
        final = _last_regime.get("regime", "Neutral")
        final_score = _last_regime.get("score", 0.0)
    else:
        final = raw
        final_score = score
        _last_regime = {"regime": final, "score": final_score, "ts": now_min}

    return {
        "regime": final,
        "score": float(final_score),
        "components": {
            "ema_stack": float(comp["ema_stack"]),
            "adx15m": float(comp["adx15m"]),
            "breadth": float(comp["breadth"]),
            "funding": float(comp["funding"]),
            "breadth_pct": float(breadth_pct),
            "adx15m_value": float(adx15) if adx15 is not None else None
        }
    }

# ============== VOLUME PROFILE (simple) ==============
def calc_vol_profile(closes, highs, lows, volumes):
    try:
        df = pd.DataFrame({'price': closes, 'volume': volumes})
        price_range = max(highs) - min(lows)
        if price_range == 0: return {'bullish_score': 0.0, 'bearish_score': 0.0}
        poc = df.groupby(pd.cut(df['price'], bins=10), observed=False)['volume'].sum().idxmax().mid
        current_price = closes[-1]
        if current_price > poc: return {'bullish_score': 3.0, 'bearish_score': 0.0}
        if current_price < poc: return {'bullish_score': 0.0, 'bearish_score': 3.0}
        return {'bullish_score': 1.0, 'bearish_score': 1.0}
    except:
        return {'bullish_score': 1.0, 'bearish_score': 1.0}

# ============== ANALYZE PER SYMBOL (7th Amendment) ==============
def analyze_data(symbol, data5m, regime_obj):
    """
    8th Amendment:
      - Remove profit ceiling veto
      - Adaptive Strong thresholds by regime
      - RSI-based confidence boosts (+5..+8; extra +3 with %B/extreme close), with MACD fail-safe
      - Adaptive SL/TP widening (1.2–2.2% per side) for extreme RSI contexts only, 1:1 maintained
      - Confluence & Overshoot kept strict (they proved protective)
      - Estimated profit = ROI on margin at fixed leverage (7x)
    """
    if not data5m or len(data5m) < 60:
        return None

    price = data5m[-1].get("close")
    if price is None:
        return None

    closes = [d["close"] for d in data5m]
    highs  = [d["high"]  for d in data5m]
    lows   = [d["low"]   for d in data5m]
    vols   = [d["volume"]for d in data5m]

    # ----- Indicators -----
    rsi   = get_last_valid_value(calc_rsi(closes, 14))
    macd  = calc_macd(closes, 12, 26, 9)
    macd_hist = macd.get("histogram") if isinstance(macd, dict) else None
    boll  = calc_bollinger(closes, 20, 2)
    atr   = calc_atr(highs, lows, closes, 14)
    cci   = calc_cci(highs, lows, closes, 20)
    ema50 = get_last_valid_value(calc_ema(closes, 50))
    # approx last ~10 hours (120x5m) for a stronger ADX read
    adx15 = calc_adx(highs[-120:], lows[-120:], closes[-120:], 14)
    volp  = calc_vol_profile(closes, highs, lows, vols)

    if any(v is None for v in [rsi, cci, boll.get("lower"), boll.get("upper")]):
        return None

    def percent_b(price, boll):
        try:
            lower = float(boll["lower"]); upper = float(boll["upper"])
            rng = (upper - lower) if (upper - lower) != 0 else 1e-9
            return float((price - lower) / rng)
        except Exception:
            return None

    pb = percent_b(price, boll)

    regime = str(regime_obj.get("regime", "Neutral"))
    regime_score = float(regime_obj.get("score", 0.0))
    comp = regime_obj.get("components", {})

    # ===== Separate scoring (same base logic as 7th) =====
    buy_score = 0.0; sell_score = 0.0

    # Buy side (mean reversion + trend support)
    if price <= boll["lower"]: buy_score += 35
    if rsi <= 30: buy_score += 30
    elif 30 < rsi <= 40: buy_score += 15
    if cci >= 100: buy_score += 15
    if macd_hist is not None and macd_hist < 0: buy_score += 5
    if ema50 is not None and price >= ema50: buy_score += 10

    # Sell side (mirror)
    if price >= boll["upper"]: sell_score += 35
    if rsi >= 70: sell_score += 30
    elif 60 <= rsi < 70: sell_score += 15
    if cci <= -100: sell_score += 15
    if macd_hist is not None and macd_hist > 0: sell_score += 5
    if ema50 is not None and price <= ema50: sell_score += 10

    # Regime bias
    if regime == "Bullish":
        buy_score += 10; sell_score -= 10
    elif regime == "Bearish":
        sell_score += 10; buy_score -= 10

    # ===== Initial direction with regime gate =====
    initial = "Neutral"
    if regime == "Bullish":
        if buy_score > 0: initial = "Buy"
    elif regime == "Bearish":
        if sell_score > 0: initial = "Sell"
    else:  # Neutral allows both
        if buy_score > sell_score and buy_score > 0: initial = "Buy"
        elif sell_score > buy_score and sell_score > 0: initial = "Sell"

    if initial == "Neutral":
        return {
            "coin": symbol, "price": round(float(price),4),
            "signal": "Neutral",
            "confidence": 0,
            "estimated_profit": "0.00%",
            "analysis_log": {
                "initial_signal": "Neutral",
                "regime": regime,
                "regime_score": regime_score,
                "regime_explain": comp
            },
            "indicators": {
                "marketRegime": regime, "regimeScore": regime_score
            }
        }

    # ===== Confluence & Overshoot (kept strict) =====
    bb_touch_buy  = price <= boll["lower"]
    bb_touch_sell = price >= boll["upper"]
    rsi_buy  = (rsi <= 30)
    rsi_sell = (rsi >= 70)
    cci_buy  = (cci >= 100)
    cci_sell = (cci <= -100)

    num_conf_buy  = int(bb_touch_buy) + int(rsi_buy) + int(cci_buy)
    num_conf_sell = int(bb_touch_sell)+ int(rsi_sell)+ int(cci_sell)

    vol_ok_buy  = volp["bullish_score"] > 0
    vol_ok_sell = volp["bearish_score"] > 0

    # Overshoot heuristic: deep band excursions
    overshoot_buy  = (pb is not None and pb <= 0.05) or rsi <= 28 or bb_touch_buy
    overshoot_sell = (pb is not None and pb >= 0.95) or rsi >= 72 or bb_touch_sell

    # Base thresholds (unchanged)
    base_buy_ok  = buy_score  >= 18
    base_sell_ok = sell_score >= 18
    conf_buy_ok  = num_conf_buy  >= (2 if regime != "Neutral" else 3)
    conf_sell_ok = num_conf_sell >= (2 if regime != "Neutral" else 3)

    # ===== Risk model & adaptive widening =====
    eff_atr = atr if atr and atr > 0 else price * 0.002
    aligned = (regime == "Bullish" and initial=="Buy") or (regime=="Bearish" and initial=="Sell")

    # base ATR factors (slight advantage if aligned)
    tp_factor = 2.2 if aligned else 2.0
    sl_factor = 1.8 if aligned else 2.0

    # Raw ATR targets
    if initial == "Buy":
        tp_raw = price + eff_atr * tp_factor
        sl_raw = price - eff_atr * sl_factor
    else:
        tp_raw = price - eff_atr * tp_factor
        sl_raw = price + eff_atr * sl_factor

    tp_pct_abs = abs((tp_raw - price) / price)
    sl_pct_abs = abs((sl_raw - price) / price)

    # Default clamps
    tp_pct_abs = min(max(tp_pct_abs, TP_PCT_MIN), TP_PCT_MAX)
    sl_pct_abs = min(max(sl_pct_abs, SL_PCT_MIN), SL_PCT_MAX)

    # Extreme-context widening (1:1 maintained)
    sell_extreme = (
        initial == "Sell" and (rsi is not None and rsi >= 70) and
        (((pb is not None) and pb >= 0.95) or (price >= boll["upper"])) and
        ((macd_hist is not None and macd_hist <= 0) or (adx15 is not None and adx15 >= 20))
    )
    buy_extreme = (
        initial == "Buy" and (rsi is not None and rsi <= 30) and
        (((pb is not None) and pb <= 0.05) or (price <= boll["lower"])) and
        ((macd_hist is not None and macd_hist >= 0) or (adx15 is not None and adx15 >= 20))
    )

    adaptive_widen_applied = False
    if sell_extreme or buy_extreme:
        pct = max(tp_pct_abs, sl_pct_abs)
        pct = min(max(pct, WIDEN_MIN_PCT), WIDEN_MAX_PCT)
        tp_pct_abs = pct
        sl_pct_abs = pct
        adaptive_widen_applied = True

    # Final TP/SL levels from pct (1:1 if widened)
    tp = price * (1 + tp_pct_abs) if initial=="Buy" else price * (1 - tp_pct_abs)
    sl = price * (1 - sl_pct_abs) if initial=="Buy" else price * (1 + sl_pct_abs)

    # Estimated profit as ROI on margin (7x)
    raw_move_pct = tp_pct_abs * 100.0
    est_profit_margin_pct = raw_move_pct * LEVERAGE_FOR_PROFIT_EVAL
    min_profit_ok = est_profit_margin_pct >= MIN_PROFIT_MARGIN
    profit_ceiling_ok = True  # ceiling removed in 8th

    # ===== Confidence calculation =====
    if initial == "Buy":
        base = buy_score
        num_conf = num_conf_buy
        vol_ok = vol_ok_buy
        overshoot_ok = overshoot_buy
    else:
        base = sell_score
        num_conf = num_conf_sell
        vol_ok = vol_ok_sell
        overshoot_ok = overshoot_sell

    base_component = max(0.0, min(1.0, base/100.0)) * 0.40
    conf_component = max(0.0, min(1.0, num_conf/3.0)) * 0.40
    veto_passes = 0
    if (base_buy_ok if initial=="Buy" else base_sell_ok): veto_passes += 1
    if vol_ok: veto_passes += 1
    if min_profit_ok and profit_ceiling_ok: veto_passes += 1
    veto_component = (veto_passes/3.0) * 0.20

    confidence = int(round((base_component + conf_component + veto_component) * 100))
    confidence = max(0, min(100, confidence))

    # ===== RSI-based confidence boosts (with MACD fail-safes) =====
    rsi_boost_points = 0
    # Sell side boost: RSI >= 70, extra +3 if %B >= 0.95 or outside upper band; skip if MACD hist > 0
    if initial == "Sell" and rsi is not None and rsi >= 70:
        if not (macd_hist is not None and macd_hist > 0):
            rsi_boost_points += 6  # base +6 (within +5..+8 range)
            if (pb is not None and pb >= 0.95) or (price >= boll["upper"]):
                rsi_boost_points += 3
    # Buy side boost: RSI <= 30, extra +3 if %B <= 0.05 or outside lower band; skip if MACD hist < 0
    if initial == "Buy" and rsi is not None and rsi <= 30:
        if not (macd_hist is not None and macd_hist < 0):
            rsi_boost_points += 6
            if (pb is not None and pb <= 0.05) or (price <= boll["lower"]):
                rsi_boost_points += 3

    if rsi_boost_points:
        confidence = max(0, min(100, confidence + rsi_boost_points))

    # ===== Final label with adaptive thresholds =====
    if regime == "Bearish":
        strong_sell_thr = STRONG_CONF_THRESHOLDS["Bearish_Sell"]
        strong_buy_thr  = STRONG_CONF_THRESHOLDS["Neutral_Buy"]  # buys generally off in bearish; keep high
    elif regime == "Bullish":
        strong_sell_thr = STRONG_CONF_THRESHOLDS["Neutral_Sell"]  # sells off in bullish; keep higher
        strong_buy_thr  = STRONG_CONF_THRESHOLDS["Bullish_Buy"]
    else:  # Neutral
        strong_sell_thr = STRONG_CONF_THRESHOLDS["Neutral_Sell"]
        strong_buy_thr  = STRONG_CONF_THRESHOLDS["Neutral_Buy"]

    final_signal = "Neutral"
    if initial == "Buy":
        if (base_buy_ok and conf_component>0 and vol_ok and overshoot_ok and min_profit_ok and confidence >= strong_buy_thr):
            final_signal = "Strong Buy"
        elif confidence >= 40:
            final_signal = "Buy"
        else:
            final_signal = "Neutral"
    else:
        if (base_sell_ok and conf_component>0 and vol_ok and overshoot_ok and min_profit_ok and confidence >= strong_sell_thr):
            final_signal = "Strong Sell"
        elif confidence >= 40:
            final_signal = "Sell"
        else:
            final_signal = "Neutral"

    # Leverage suggestion
    leverage = 7 if final_signal.startswith("Strong") else (6 if confidence >= 50 else 5)

    # ----- JSON-safe payload -----
    analysis_log = {
        "initial_signal": initial,
        "regime": regime,
        "regime_score": float(regime_score),
        "regime_explain": comp,
        "buy_score": int(round(buy_score)),
        "sell_score": int(round(sell_score)),
        "num_confluence_met": int(num_conf),
        "vol_profile_ok": bool(vol_ok),
        "overshoot_ok": bool(overshoot_ok),
        "min_profit_ok": bool(min_profit_ok),
        "profit_ceiling_ok": True,  # always true now
        "profit_eval_leverage": float(LEVERAGE_FOR_PROFIT_EVAL),
        "profit_basis": "margin_roi_percent",
        "adaptive_widen_applied": bool(adaptive_widen_applied),
        "tp_sl_pct": round(tp_pct_abs * 100.0, 3),
        "rsi_conf_boost_points": int(rsi_boost_points)
    }

    return {
        "coin": symbol,
        "price": round(float(price), 4),
        "tp": round(float(tp), 4),
        "sl": round(float(sl), 4),
        "leverage": f"{int(leverage)}x",
        "confidence": int(confidence),
        "signal": final_signal,
        "estimated_profit": f"{est_profit_margin_pct:.2f}%",
        "analysis_log": analysis_log,
        "indicators": {
            "rsi5m": float(rsi),
            "macd_hist5m": float(macd_hist) if macd_hist is not None else None,
            "boll5m": {
                "upper": float(boll["upper"]),
                "lower": float(boll["lower"]),
                "middle": float(boll["middle"])
            },
            "cci5m": float(cci),
            "ema50_5m": float(ema50) if ema50 is not None else None,
            "adx15m": float(adx15) if adx15 is not None else None,
            "percentB": float(pb) if pb is not None else None,
            "marketRegime": str(regime),
            "regimeScore": float(regime_score),
            "volProfile": {
                "bullish_score": float(volp["bullish_score"]),
                "bearish_score": float(volp["bearish_score"])
            }
        }
    }
    
# ============== MAIN ==============
if __name__ == "__main__":
    print("Starting automated data fetch...")

    symbols = fetch_top_volume_coins()
    if not symbols:
        print("Could not fetch top coins. Exiting."); exit()

    print(f"Found {len(symbols)} coins to analyze.")

    # BTC multi-timeframe for regime
    btc5  = fetch_binance_klines("BTCUSDT", "5m", 300)
    btc15 = fetch_binance_klines("BTCUSDT", "15m", 300)
    btc1h = fetch_binance_klines("BTCUSDT", "1h", 300)

    # breadth snapshot (subset for speed)
    breadth_snap = []
    for sym in symbols[:30]:
        d5 = fetch_binance_klines(sym, "5m", 120)
        if d5:
            breadth_snap.append({"symbol": sym, "closes": [x["close"] for x in d5]})
        time.sleep(0.05)

    funding_avg = fetch_funding_rate_avg(symbols)
    regime_obj = compute_market_regime(btc5, btc15, btc1h, breadth_snap, funding_avg)
    print(f"Regime determined: {regime_obj.get('regime')} (score={regime_obj.get('score')})")

    all_results = []
    for coin in symbols:
        print(f" - Analyzing {coin}...")
        time.sleep(0.2)
        data_5m = fetch_binance_klines(coin, "5m", 200)
        if not data_5m: continue
        res = analyze_data(coin, data_5m, regime_obj)
        if res:
            all_results.append(res)

    if all_results:
        strong_signals = [s for s in all_results if "Strong" in s.get('signal',"")]
        print(f"\nAnalysis complete. Found {len(strong_signals)} strong signals.")
        print("Saving full analysis file...")

        # IST timestamped filenames
        utc_now = datetime.now(pytz.utc)
        ist_tz = pytz.timezone("Asia/Kolkata")
        ist_now = utc_now.astimezone(ist_tz)
        timestamp_str = ist_now.strftime("%Y-%m-%d_%H-%M-%S")

        file_suffix = "_STRONG" if strong_signals else ""
        archive_filename = f"signals_{timestamp_str}{file_suffix}.json"

        os.makedirs(ARCHIVE_FOLDER, exist_ok=True)
        archive_filepath = os.path.join(ARCHIVE_FOLDER, archive_filename)

        with open(archive_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Archive file saved to {archive_filepath}")

        with open(LIVE_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        print(f"SUCCESS: Live data file saved as {LIVE_FILENAME}")
    else:
        print("\nNo results generated. No file will be saved.")


