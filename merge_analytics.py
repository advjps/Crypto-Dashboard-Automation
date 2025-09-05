# merge_analytics.py (v2) -- merges ANY analytics/*.csv, explodes indicator_score columns,
# and produces automated checklist outputs: confluence table + indicator attribution.
import os
import glob
from datetime import datetime
import pytz
import pandas as pd
import numpy as np

ANALYTICS_DIR = "analytics"
ALL_SIGNALS_CSV = os.path.join(ANALYTICS_DIR, "all_signals.csv")
os.makedirs(ANALYTICS_DIR, exist_ok=True)

def ist_timestamp():
    return datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d_%H-%M-%S")

def load_all_csvs():
    files = sorted(glob.glob(os.path.join(ANALYTICS_DIR, "*.csv")))
    if not files:
        print(f"[INFO] No CSV files found in {ANALYTICS_DIR}/")
        return pd.DataFrame()
    frames = []
    for f in files:
        # skip the merged master if it exists already
        if os.path.basename(f) == os.path.basename(ALL_SIGNALS_CSV):
            continue
        try:
            df = pd.read_csv(f, dtype=str)  # read as str; coerce later
            df["__source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")
    if not frames:
        return pd.DataFrame()
    # concat with sort=False to preserve all columns (union)
    return pd.concat(frames, ignore_index=True, sort=False)

def safe_col(df, names):
    """Return first existing column among `names` (case-insensitive), or None."""
    if df is None or df.empty:
        return None
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n and n.lower() in cols:
            return cols[n.lower()]
    return None

def normalize_main_columns(df):
    """
    Create canonical columns if possible, do not drop extras.
    We'll add canonical columns if found under alternate names.
    """
    if df is None or df.empty:
        return df
    colmap = {}
    lower = {c.lower(): c for c in df.columns}

    def find(*opts):
        for o in opts:
            if o and o.lower() in lower:
                return lower[o.lower()]
        return None

    mapping = {
        "Coin": find("coin", "Coin"),
        "Signal": find("signal", "Signal"),
        "Outcome": find("outcome", "Outcome"),
        "Confidence": find("confidence", "Confidence", "POP", "pop"),
        "Regime": find("regime", "Regime"),
        "Estimated_Profit": find("estimated_profit", "Estimated_Profit"),
        "DeservedStrongBuy": find("deservedstrongbuy", "deserved_strong_buy", "DeservedStrongBuy"),
        "DeservedStrongSell": find("deservedstrongsell", "deserved_strong_sell", "DeservedStrongSell"),
        "Buy_Score": find("buy_score", "Buy_Score"),
        "Sell_Score": find("sell_score", "Sell_Score"),
        "Num_Confluence_Buy": find("num_confluence_buy", "Num_Confluence_Buy", "num_conf_buy"),
        "Num_Confluence_Sell": find("num_confluence_sell", "Num_Confluence_Sell", "num_conf_sell"),
        "Duration(min)": find("duration(min)", "Duration(min)"),
        "Did_TP_Hit": find("did_tp_hit", "Did_TP_Hit"),
        "SignalTimeUTC": find("signal_time_utc", "SignalTimeUTC"),
        "SignalTimeIST": find("signal_time_ist", "SignalTimeIST"),
        "SourceFile": find("sourcefile", "__source_file", "SourceFile")
    }
    # Create canonical columns (copy if existing)
    for canon, found in mapping.items():
        if found and found != canon:
            df[canon] = df[found]
        elif found is None and canon not in df.columns:
            df[canon] = pd.NA
    return df

def coerce_numeric_columns(df):
    """
    Attempt to coerce a wide set of likely numeric columns to numbers.
    Leave others unchanged.
    """
    if df is None or df.empty:
        return df
    # heuristics: any column with these substrings or ending with _score etc
    to_try = []
    for c in df.columns:
        lc = c.lower()
        if ("score" in lc) or any(s in lc for s in ["rsi", "cci", "cmf", "macd", "boll_", "keltner_", "ema50", "tsi", "stc", "williams", "confidence", "pop", "duration", "buy_score", "sell_score", "markettrend"]):
            to_try.append(c)
    for c in to_try:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass
    return df

def collect_indicator_score_columns(df):
    """
    Find all columns that look like per-indicator scores or raw indicator fields.
    Detection heuristics:
      - column names containing '_score' OR
      - known indicator names: rsi5m, williamsr, cci5m, cmf5m, macd5m_hist etc
    Return sorted list of column names.
    """
    if df is None or df.empty:
        return []
    cols = df.columns.tolist()
    matched = set()
    for c in cols:
        lc = c.lower()
        if "_score" in lc:
            matched.add(c)
        # raw indicators
        if any(k in lc for k in ["rsi", "williams", "williamsr", "cci", "cmf", "cvd", "cvd5m", "hma", "alma", "keltner", "boll", "macd", "tsi", "stc", "ema50"]):
            matched.add(c)
    return sorted(matched)

def ensure_uniform_columns(df, extra_cols):
    """
    Guarantee that all names in extra_cols exist in df, creating them as NaN if missing.
    Returns df with columns extended.
    """
    for c in extra_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def compute_confluence_table(df, out_prefix):
    """
    Build confluence table: for buy_overshoot (0..4) & sell_overshoot (0..4),
    compute Total, Success, Fail, WinRate, AvgConfidence.
    Save CSV and return DataFrame.
    """
    # use canonical numeric columns if present
    buy_col = None
    if "Num_Confluence_Buy" in df.columns:
        buy_col = "Num_Confluence_Buy"
    else:
        # try alternative
        for c in df.columns:
            if "num_conf" in c.lower() and "buy" in c.lower():
                buy_col = c; break
    sell_col = None
    if "Num_Confluence_Sell" in df.columns:
        sell_col = "Num_Confluence_Sell"
    else:
        for c in df.columns:
            if "num_conf" in c.lower() and "sell" in c.lower():
                sell_col = c; break

    conf_rows = []
    # We'll do separate tables for buy and sell overshoots
    for side, col in [("Buy", buy_col), ("Sell", sell_col)]:
        if col is None or col not in df.columns:
            continue
        # ensure numeric
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
        for n in range(0, 5):
            sub = df[df[col] == n]
            total = len(sub)
            if total == 0:
                success = fail = inconcl = 0
                avg_conf = 0.0
                wr = 0.0
            else:
                success = int((sub["Outcome"] == "Success").sum()) if "Outcome" in sub.columns else 0
                fail = int((sub["Outcome"] == "Fail").sum()) if "Outcome" in sub.columns else 0
                inconcl = int((sub["Outcome"] == "Inconclusive").sum()) if "Outcome" in sub.columns else 0
                wr = (success / (success + fail) * 100.0) if (success + fail) > 0 else 0.0
                avg_conf = float(sub["Confidence"].dropna().astype(float).mean()) if "Confidence" in sub.columns else 0.0
            conf_rows.append({
                "Side": side,
                "Num_Overshoots": n,
                "Total": total,
                "Success": success,
                "Fail": fail,
                "Inconclusive": inconcl,
                "WinRate": wr,
                "AvgConfidence": avg_conf
            })
    conf_df = pd.DataFrame(conf_rows)
    conf_csv = os.path.join(ANALYTICS_DIR, f"confluence_table_{out_prefix}.csv")
    conf_df.to_csv(conf_csv, index=False)
    return conf_df

def compute_indicator_attribution(df, out_prefix):
    """
    For each indicator/score column found (by collect_indicator_score_columns),
    compute:
      - Count when indicator is present and >0
      - SuccessCount / FailCount among rows where indicator>0
      - SuccessRateWhenPresent
      - AvgContributionOnSuccess (mean of value when Outcome==Success)
      - AvgContributionOnFail
      - Lift = SuccessRateWhenPresent - OverallSuccessRate
    Save CSV and return DataFrame.
    """
    indicators = collect_indicator_score_columns(df)
    if not indicators:
        print("[INFO] No indicator columns detected for attribution.")
        return pd.DataFrame()

    overall_success = (df["Outcome"] == "Success").sum() if "Outcome" in df.columns else 0
    overall_fail = (df["Outcome"] == "Fail").sum() if "Outcome" in df.columns else 0
    overall_wr = (overall_success / (overall_success + overall_fail) * 100.0) if (overall_success + overall_fail) > 0 else 0.0

    rows = []
    for col in indicators:
        # only numeric contributor columns make sense for avg contribution
        ser = pd.to_numeric(df.get(col, pd.Series([np.nan]*len(df))), errors="coerce")
        present_mask = ser.notna()
        pos_mask = ser > 0  # positive contribution
        count_present = int(present_mask.sum())
        count_pos = int(pos_mask.sum())
        if count_pos == 0:
            succ_pos = fail_pos = 0
            succ_rate_pos = 0.0
            avg_contrib_succ = avg_contrib_fail = np.nan
        else:
            sub_pos = df[pos_mask]
            succ_pos = int((sub_pos["Outcome"] == "Success").sum()) if "Outcome" in sub_pos.columns else 0
            fail_pos = int((sub_pos["Outcome"] == "Fail").sum()) if "Outcome" in sub_pos.columns else 0
            succ_rate_pos = (succ_pos / (succ_pos + fail_pos) * 100.0) if (succ_pos + fail_pos) > 0 else 0.0
            avg_contrib_succ = float(ser[(df["Outcome"] == "Success") & pos_mask].mean()) if "Outcome" in df.columns else np.nan
            avg_contrib_fail = float(ser[(df["Outcome"] == "Fail") & pos_mask].mean()) if "Outcome" in df.columns else np.nan

        lift = succ_rate_pos - overall_wr
        rows.append({
            "Indicator": col,
            "CountPresent": count_present,
            "CountPositive": count_pos,
            "SuccessPos": succ_pos,
            "FailPos": fail_pos,
            "SuccessRateWhenPos(%)": succ_rate_pos,
            "AvgContribOnSuccess": avg_contrib_succ,
            "AvgContribOnFail": avg_contrib_fail,
            "OverallWinRate(%)": overall_wr,
            "Lift(%)": lift
        })
    out_df = pd.DataFrame(rows).sort_values(by="Lift(%)", ascending=False)
    out_csv = os.path.join(ANALYTICS_DIR, f"indicator_attribution_{out_prefix}.csv")
    out_df.to_csv(out_csv, index=False)
    return out_df

def main():
    df = load_all_csvs()
    if df.empty:
        print("[INFO] Nothing to merge. Exiting.")
        return

    # Normalize canonical main columns
    df = normalize_main_columns(df)

    # Coerce many likely numeric columns
    df = coerce_numeric_columns(df)

    # Discover all indicator & score columns across files and ensure uniform columns
    all_indicator_cols = collect_indicator_score_columns(df)
    df = ensure_uniform_columns(df, all_indicator_cols)

    # Final coercion pass (for any newly added columns)
    df = coerce_numeric_columns(df)

    # Save canonical master (preserve all columns so indicator-level columns remain)
    try:
        df.to_csv(ALL_SIGNALS_CSV, index=False)
        print(f"[OK] Wrote {ALL_SIGNALS_CSV} with {len(df)} rows.")
    except Exception as e:
        print(f"[ERR] Could not write {ALL_SIGNALS_CSV}: {e}")

    # --- Summaries as before ---
    stamp = ist_timestamp()
    # By Signal bucket
    buckets = ["Strong Buy", "Buy", "Strong Sell", "Sell"]
    bucket_rows = []
    for b in buckets:
        if "Signal" not in df.columns:
            bucket_rows.append({"Section": b, "Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0})
            continue
        sub = df[df["Signal"] == b]
        if sub.empty:
            bucket_rows.append({"Section": b, "Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0})
            continue
        succ = int((sub["Outcome"] == "Success").sum()) if "Outcome" in sub.columns else 0
        fail = int((sub["Outcome"] == "Fail").sum()) if "Outcome" in sub.columns else 0
        inc = int((sub["Outcome"] == "Inconclusive").sum()) if "Outcome" in sub.columns else 0
        wr = (succ / (succ + fail) * 100.0) if (succ + fail) > 0 else 0.0
        avg_conf = float(sub["Confidence"].dropna().astype(float).mean()) if "Confidence" in sub.columns else 0.0
        bucket_rows.append({"Section": b, "Total": len(sub), "Success": succ, "Fail": fail, "Inconclusive": inc, "WinRate": wr, "AvgConfidence": avg_conf})
    bucket_df = pd.DataFrame(bucket_rows)
    bucket_csv = os.path.join(ANALYTICS_DIR, f"summary_by_signal_{stamp}.csv")
    bucket_df.to_csv(bucket_csv, index=False)

    # By Regime
    regime_df = pd.DataFrame()
    if "Regime" in df.columns:
        rows = []
        for regime, g in df.groupby("Regime"):
            succ = int((g["Outcome"] == "Success").sum()) if "Outcome" in g.columns else 0
            fail = int((g["Outcome"] == "Fail").sum()) if "Outcome" in g.columns else 0
            inc = int((g["Outcome"] == "Inconclusive").sum()) if "Outcome" in g.columns else 0
            wr = (succ / (succ + fail) * 100.0) if (succ + fail) > 0 else 0.0
            avg_conf = float(g["Confidence"].dropna().astype(float).mean()) if "Confidence" in g.columns else 0.0
            rows.append({"Regime": regime, "Total": len(g), "Success": succ, "Fail": fail, "Inconclusive": inc, "WinRate": wr, "AvgConfidence": avg_conf})
        regime_df = pd.DataFrame(rows)
    regime_csv = os.path.join(ANALYTICS_DIR, f"summary_by_regime_{stamp}.csv")
    regime_df.to_csv(regime_csv, index=False)

    # Deserved strong
    deserved_df = pd.DataFrame()
    ds_rows = []
    if "DeservedStrongBuy" in df.columns:
        sub = df[pd.to_numeric(df["DeservedStrongBuy"], errors="coerce").fillna(0).astype(int) == 1]
        succ = int((sub["Outcome"] == "Success").sum()) if "Outcome" in sub.columns else 0
        fail = int((sub["Outcome"] == "Fail").sum()) if "Outcome" in sub.columns else 0
        inc = int((sub["Outcome"] == "Inconclusive").sum()) if "Outcome" in sub.columns else 0
        wr = (succ / (succ + fail) * 100.0) if (succ + fail) > 0 else 0.0
        avg_conf = float(sub["Confidence"].dropna().astype(float).mean()) if "Confidence" in sub.columns else 0.0
        ds_rows.append({"Group": "DeservedStrongBuy", "Total": len(sub), "Success": succ, "Fail": fail, "Inconclusive": inc, "WinRate": wr, "AvgConfidence": avg_conf})
    if "DeservedStrongSell" in df.columns:
        sub = df[pd.to_numeric(df["DeservedStrongSell"], errors="coerce").fillna(0).astype(int) == 1]
        succ = int((sub["Outcome"] == "Success").sum()) if "Outcome" in sub.columns else 0
        fail = int((sub["Outcome"] == "Fail").sum()) if "Outcome" in sub.columns else 0
        inc = int((sub["Outcome"] == "Inconclusive").sum()) if "Outcome" in sub.columns else 0
        wr = (succ / (succ + fail) * 100.0) if (succ + fail) > 0 else 0.0
        avg_conf = float(sub["Confidence"].dropna().astype(float).mean()) if "Confidence" in sub.columns else 0.0
        ds_rows.append({"Group": "DeservedStrongSell", "Total": len(sub), "Success": succ, "Fail": fail, "Inconclusive": inc, "WinRate": wr, "AvgConfidence": avg_conf})
    if ds_rows:
        deserved_df = pd.DataFrame(ds_rows)
    deserved_csv = os.path.join(ANALYTICS_DIR, f"summary_deserved_{stamp}.csv")
    deserved_df.to_csv(deserved_csv, index=False)

    # Confluence table
    conf_df = compute_confluence_table(df, stamp)

    # Indicator attribution
    ind_attr_df = compute_indicator_attribution(df, stamp)

    # Human readable summary (extended)
    summary_txt = os.path.join(ANALYTICS_DIR, f"summary_{stamp}.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("============== GLOBAL SUMMARY ==============\n\n")
        f.write("By Signal Bucket\n")
        f.write("-------------------------------------------\n")
        if not bucket_df.empty:
            f.write(bucket_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

        f.write("By Regime\n")
        f.write("-------------------------------------------\n")
        if not regime_df.empty:
            f.write(regime_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

        f.write("Deserved Strong (Analysis-only)\n")
        f.write("-------------------------------------------\n")
        if not deserved_df.empty:
            f.write(deserved_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

        f.write("Confluence Table (overshoots -> performance)\n")
        f.write("-------------------------------------------\n")
        if not conf_df.empty:
            f.write(conf_df.to_string(index=False, formatters={
                "WinRate": lambda x: f"{x:6.2f}%",
                "AvgConfidence": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

        f.write("Top indicator attribution (by Lift%)\n")
        f.write("-------------------------------------------\n")
        if not ind_attr_df.empty:
            top = ind_attr_df.head(20)
            f.write(top.to_string(index=False, formatters={
                "SuccessRateWhenPos(%)": lambda x: f"{x:6.2f}%",
                "OverallWinRate(%)": lambda x: f"{x:6.2f}%",
                "Lift(%)": lambda x: f"{x:6.2f}"
            }) + "\n\n")
        else:
            f.write("(no data)\n\n")

    print(f"[OK] Wrote summaries:\n - {bucket_csv}\n - {regime_csv}\n - {deserved_csv}\n - {os.path.join(ANALYTICS_DIR, f'confluence_table_{stamp}.csv')}\n - {os.path.join(ANALYTICS_DIR, f'indicator_attribution_{stamp}.csv')}\n - {summary_txt}")

if __name__ == "__main__":
    main()
