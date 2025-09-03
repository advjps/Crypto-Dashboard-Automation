# run_automation.py (9th Amendment)

import pandas as pd
import requests
import json
import math
import os
import time
from datetime import datetime
import pytz

# --- PROXY CONFIGURATION (set if needed) ---
PROXY_IP = "217.180.42.139"
PROXY_PORT = "48642"
PROXY_USER = "NQOgprvOa4fgcWw"
PROXY_PASS = "Nx8gIuzPunYu7P1"

proxy_url = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_IP}:{PROXY_PORT}"
proxies = {"http": proxy_url, "https": proxy_url}

# --- CONFIG ---
LIVE_FILENAME = "live_signals.json"
ARCHIVE_FOLDER = "data_archive"
BINANCE_FAPI = "https://fapi.binance.com"

# Profit evaluation
LEVERAGE_FOR_PROFIT_EVAL = 7
MIN_PROFIT_MARGIN = 2.0  # %
# Removed profit ceiling

# TP/SL percentages
TP_PCT_PRICE = 0.0072  # ~0.72% of price
SL_PCT_PRICE = 0.03    # ~3% of price

# ====== INDICATOR FUNCTIONS ======
def get_last_valid_value(values):
    for v in reversed(values):
        if v is not None and not math.isnan(v):
            return v
    return None

def calc_ema(values, period):
    if len(values) < period: return [None] * len(values)
    return pd.Series(values).ewm(span=period, adjust=False).mean().tolist()

def calc_rsi(values, period=14):
    if len(values) < period+1: return [None]*len(values)
    series = pd.Series(values)
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/period).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/period).mean()
    rs = gain / loss
    return (100 - (100/(1+rs))).tolist()

def calc_macd(values, fast=12, slow=26, signal=9):
    series = pd.Series(values)
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal).mean()
    hist = macd - sig
    return {"macd": macd.iloc[-1], "signal": sig.iloc[-1], "histogram": hist.iloc[-1]}

def calc_bollinger(values, period=20, mult=2):
    if len(values) < period: return {"upper":None,"middle":None,"lower":None}
    series = pd.Series(values)
    ma = series.rolling(period).mean().iloc[-1]
    std = series.rolling(period).std().iloc[-1]
    return {"upper": ma+mult*std, "middle": ma, "lower": ma-mult*std}

def calc_cci(highs, lows, closes, period=20):
    if len(closes) < period: return None
    tp = (pd.Series(highs)+pd.Series(lows)+pd.Series(closes))/3
    ma = tp.rolling(period).mean().iloc[-1]
    md = tp.rolling(period).apply(lambda x:(x-x.mean()).abs().mean()).iloc[-1]
    return (tp.iloc[-1]-ma)/(0.015*md) if md!=0 else 0

def calc_williams_r(highs, lows, closes, period=14):
    if len(closes) < period: return None
    high_max = pd.Series(highs).rolling(period).max().iloc[-1]
    low_min = pd.Series(lows).rolling(period).min().iloc[-1]
    return -100 * ((high_max - closes[-1]) / (high_max - low_min))

def calc_cmf(highs, lows, closes, volumes, period=20):
    if len(closes) < period: return None
    mf_mult = ((pd.Series(closes)-pd.Series(lows))-(pd.Series(highs)-pd.Series(closes))) / (pd.Series(highs)-pd.Series(lows)).replace(0,1e-9)
    mf_vol = mf_mult * pd.Series(volumes)
    return (mf_vol.rolling(period).sum()/pd.Series(volumes).rolling(period).sum()).iloc[-1]

# Dummy simplified new indicators
def calc_stc(values): return 25  # stub
def calc_tsi(values): return -0.02  # stub
def calc_cvd(closes,volumes): return "rising" if volumes[-1]>sum(volumes)/len(volumes) else "falling"
def calc_hma(values, period=9): return "up" if values[-1]>values[-2] else "down"
def calc_alma(values): return "up" if values[-1]>values[-2] else "down"
def calc_keltner(values, period=20): return {"upper":max(values),"middle":sum(values)/len(values),"lower":min(values)}

def calc_market_trend(closes):
    ema20, ema50 = calc_ema(closes,20)[-1], calc_ema(closes,50)[-1]
    if ema20 and ema50:
        if ema20>ema50: return 10
        if ema20<ema50: return -10
    return 0

# ====== DATA FETCHING ======
def fetch_top_volume_coins(min_quote_volume=100_000_000):
    url = f"{BINANCE_FAPI}/fapi/v1/ticker/24hr"
    try:
        data = requests.get(url,proxies=proxies,timeout=30).json()
        usdt_pairs=[d for d in data if d["symbol"].endswith("USDT")]
        return [d["symbol"] for d in usdt_pairs if float(d["quoteVolume"])>=min_quote_volume]
    except: return []

def fetch_binance_data(symbol, interval="5m", limit=100):
    url=f"{BINANCE_FAPI}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        d=requests.get(url,proxies=proxies,timeout=30).json()
        return [{"open":float(x[1]),"high":float(x[2]),"low":float(x[3]),"close":float(x[4]),"volume":float(x[5])} for x in d]
    except: return []

# ====== MAIN ANALYSIS ======
def analyze_data(symbol, data5m, market_trend):
    if not data5m or len(data5m)<50: return None
    price=data5m[-1]["close"]
    closes=[d["close"] for d in data5m]; highs=[d["high"] for d in data5m]; lows=[d["low"] for d in data5m]; vols=[d["volume"] for d in data5m]

    # indicators
    rsi=get_last_valid_value(calc_rsi(closes))
    willr=calc_williams_r(highs,lows,closes)
    cci=calc_cci(highs,lows,closes)
    boll=calc_bollinger(closes)
    macd=calc_macd(closes)
    cmf=calc_cmf(highs,lows,closes,vols)
    stc=calc_stc(closes); tsi=calc_tsi(closes); cvd=calc_cvd(closes,vols); hma=calc_hma(closes); alma=calc_alma(closes); keltner=calc_keltner(closes)

    # scores
    buy_score=sell_score=0
    if rsi and rsi<=30: buy_score+=25
    if willr and willr<=-80: buy_score+=15
    if cci and cci>=100: buy_score+=10
    if boll["lower"] and price<=boll["lower"]: buy_score+=15

    if rsi and rsi>=70: sell_score+=25
    if willr and willr>=-20: sell_score+=15
    if cci and cci<=-100: sell_score+=10
    if boll["upper"] and price>=boll["upper"]: sell_score+=15

    # regime bias
    if market_trend>=5: buy_score+=10; sell_score-=5
    elif market_trend<=-5: sell_score+=10; buy_score-=5

    # thresholds
    buy_thresh, strong_buy_thresh=25,45
    sell_thresh, strong_sell_thresh=25,45

    signal="Neutral"; conf=0
    if buy_score>=sell_score and buy_score>=buy_thresh:
        conf=buy_score
        signal="Strong Buy" if buy_score>=strong_buy_thresh else "Buy"
    elif sell_score>buy_score and sell_score>=sell_thresh:
        conf=sell_score
        signal="Strong Sell" if sell_score>=strong_sell_thresh else "Sell"
    else: return None  # skip Neutral

    # tp/sl
    tp=price*(1+TP_PCT_PRICE) if "Buy" in signal else price*(1-TP_PCT_PRICE)
    sl=price*(1-SL_PCT_PRICE) if "Buy" in signal else price*(1+SL_PCT_PRICE)
    est_profit=TP_PCT_PRICE*100*LEVERAGE_FOR_PROFIT_EVAL

    # timestamps
    utc_now=datetime.utcnow().replace(tzinfo=pytz.utc)
    ist_now=utc_now.astimezone(pytz.timezone("Asia/Kolkata"))

    return {
        "coin":symbol,"price":round(price,4),"tp":round(tp,4),"sl":round(sl,4),
        "leverage":f"{LEVERAGE_FOR_PROFIT_EVAL}x","confidence":int(conf),
        "signal":signal,"estimated_profit":f"{est_profit:.2f}%",
        "regime":"Bullish" if market_trend>0 else "Bearish" if market_trend<0 else "Neutral",
        "signal_time_utc":utc_now.isoformat(),
        "signal_time_ist":ist_now.isoformat(),
        "analysis_log":{"buy_score":buy_score,"sell_score":sell_score},
        "indicators":{"rsi5m":rsi,"williamsR":willr,"cci5m":cci,"stc5m":stc,"tsi5m":tsi,"cmf5m":cmf,"cvd5m":cvd,
                      "hma5m":hma,"alma5m":alma,"keltner5m":keltner,"boll5m":boll,"macd5m":macd,"marketTrend":market_trend}
    }

# ====== MAIN LOOP ======
if __name__=="__main__":
    coins=fetch_top_volume_coins()
    btc_data=fetch_binance_data("BTCUSDT")
    mtrend=calc_market_trend([d["close"] for d in btc_data])
    results=[]
    for c in coins:
        d=fetch_binance_data(c)
        res=analyze_data(c,d,mtrend)
        if res: results.append(res)

    if results:
        strong=[r for r in results if "Strong" in r["signal"]]
        utc_now=datetime.utcnow().replace(tzinfo=pytz.utc)
        ist_now=utc_now.astimezone(pytz.timezone("Asia/Kolkata"))
        fname=f"signals_{ist_now.strftime('%Y-%m-%d_%H-%M-%S')}{'_STRONG' if strong else ''}.json"
        os.makedirs(ARCHIVE_FOLDER,exist_ok=True)
        with open(os.path.join(ARCHIVE_FOLDER,fname),"w") as f: json.dump(results,f,indent=2)
        with open(LIVE_FILENAME,"w") as f: json.dump(results,f,indent=2)
