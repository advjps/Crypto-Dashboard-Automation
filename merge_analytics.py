#!/usr/bin/env python3
# merge_analytics.py  (robust aggregator)
# Usage: python merge_analytics.py

import os
import glob
import json
from datetime import datetime
import pytz
import pandas as pd

ANALYTICS_DIR = "analytics"
os.makedirs(ANALYTICS_DIR, exist_ok=True)
ALL_SIGNALS_CSV = os.path.join(ANALYTICS_DIR, "all_signals.csv")

def ist_timestamp():
    ist = pytz.timezone("Asia/Kolkata")
    return datetime.now(ist).strftime("%Y-%m-%d_%H-%M-%S")

def load_all_csvs():
    files = sorted(glob.glob(os.path.join(ANALYTICS_DIR, "*.csv")))
    if not files:
        print(f"[INFO] No per-file analytics CSVs found in {ANALYTICS_DIR}/")
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def normalize_col(df, colnames):
    """
    Given dataframe 'df' and a list of possible column names in priority,
    returns series using first existing column (case-insensitive).
    """
    if df is None or df.empty:
        return pd.Series(dtype=object)
    cols_lower = {c.lower(): c for c in df.columns}
    for c in colnames:
        if c.lower() in cols_lower:
            return df[cols_lower[c.lower()]]
    # not found -> return series of NAs
    return pd.Series([None]*len(df))

def winrate(success, fail):
    denom = (success or 0) + (fail or 0)
    return (success / denom * 100.0) if denom > 0 else 0.0

def summarize_bucket(df, bucket_name):
    # case-insensitive matching for Signal column
    signal_series = normalize_col(df, ["Signal", "signal"])
    outcome_series = normalize_col(df, ["Outcome", "outcome"])
    conf_series = normalize_col(df, ["Confidence", "confidence"])
    if signal_series is None or signal_series.empty:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    mask = signal_series.astype(str).str.strip().str.lower() == bucket_name.lower()
    sub = df[mask].copy()
    if sub.empty:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    # outcome mapping
    succ = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "success").sum() if outcome_series.name in sub.columns else 0
    fail = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "fail").sum() if outcome_series.name in sub.columns else 0
    inconc = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "inconclusive").sum() if outcome_series.name in sub.columns else 0
    wr = winrate(int(succ), int(fail))
    avg_conf = pd.to_numeric(sub[conf_series.name], errors="coerce").dropna().mean() if conf_series.name in sub.columns else 0.0
    return {"Total": int(len(sub)), "Success": int(succ), "Fail": int(fail), "Inconclusive": int(inconc), "WinRate": wr, "AvgConfidence": float(avg_conf or 0.0)}

def regime_summary(df):
    outcome_series = normalize_col(df, ["Outcome", "outcome"])
    conf_series = normalize_col(df, ["Confidence", "confidence"])
    # regime column might be named 'Regime' or 'regime'
    regime_series = normalize_col(df, ["Regime", "regime"])
    if df.empty or regime_series.empty:
        return pd.DataFrame()
    rows = []
    for regime, g in df.groupby(regime_series):
        if regime in (None, "", float("nan")):
            regime = "Unknown"
        succ = (g[outcome_series.name].astype(str).str.strip().str.lower() == "success").sum() if outcome_series.name in g.columns else 0
        fail = (g[outcome_series.name].astype(str).str.strip().str.lower() == "fail").sum() if outcome_series.name in g.columns else 0
        inc = (g[outcome_series.name].astype(str).str.strip().str.lower() == "inconclusive").sum() if outcome_series.name in g.columns else 0
        wr = winrate(int(succ), int(fail))
        avg_conf = pd.to_numeric(g[conf_series.name], errors="coerce").dropna().mean() if conf_series.name in g.columns else 0.0
        rows.append({
            "Regime": str(regime), "Total": int(len(g)), "Success": int(succ), "Fail": int(fail),
            "Inconclusive": int(inc), "WinRate": wr, "AvgConfidence": avg_conf
        })
    return pd.DataFrame(rows).sort_values(["Regime"])

def deserved_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = []
    # try columns that may flag deserved strong; be tolerant of names
    for flag_col in ["DeservedStrongBuy", "deservedstrongbuy", "DeservedStrongSell", "deservedstrongsell", "DeservedStrong_Buy", "DeservedStrong_Sell"]:
        if flag_col not in df.columns:
            # try case-insensitive match
            matches = [c for c in df.columns if c.lower() == flag_col.lower()]
            if matches:
                flag_col = matches[0]
            else:
                continue
        sub = df[pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        succ = (sub[normalize_col(sub, ["Outcome", "outcome"]).name].astype(str).str.strip().str.lower() == "success").sum() if not sub.empty else 0
        fail = (sub[normalize_col(sub, ["Outcome", "outcome"]).name].astype(str).str.strip().str.lower() == "fail").sum() if not sub.empty else 0
        inc = (sub[normalize_col(sub, ["Outcome", "outcome"]).name].astype(str).str.strip().str.lower() == "inconclusive").sum() if not sub.empty else 0
        wr = winrate(int(succ), int(fail))
        avg_conf = pd.to_numeric(sub[normalize_col(sub, ["Confidence","confidence"]).name], errors="coerce").dropna().mean() if not sub.empty else 0.0
        label = flag_col
        out.append({"Group": label, "Total": int(len(sub)), "Success": int(succ), "Fail": int(fail), "Inconclusive": int(inc), "WinRate": wr, "AvgConfidence": avg_conf})
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)

def main():
    df = load_all_csvs()
    if df.empty:
        print("[INFO] Nothing to merge. Exiting.")
        return

    # Normalize columns by ensuring consistent names exist
    # Save combined file
    try:
        df.to_csv(ALL_SIGNALS_CSV, index=False)
        print(f"[OK] Wrote {ALL_SIGNALS_CSV} with {len(df)} rows.")
    except Exception as e:
        print(f"[ERROR] Could not write all_signals.csv: {e}")
        return

    # Build bucket summary
    buckets = ["Strong Buy", "Buy", "Strong Sell", "Sell"]
    rows = []
    for b in buckets:
        rows.append({"Section": b, **summarize_bucket(df, b)})
    bucket_df = pd.DataFrame(rows)

    # Regime summary
    regime_df = regime_summary(df)

    # Deserved strong (if any)
    deserved_df = deserved_summary(df)

    # Save CSV summaries
    stamp = ist_timestamp()
    bucket_csv = os.path.join(ANALYTICS_DIR, f"summary_by_signal_{stamp}.csv")
    regime_csv = os.path.join(ANALYTICS_DIR, f"summary_by_regime_{stamp}.csv")
    deserved_csv = os.path.join(ANALYTICS_DIR, f"summary_deserved_{stamp}.csv")
    bucket_df.to_csv(bucket_csv, index=False)
    regime_df.to_csv(regime_csv, index=False)
    deserved_df.to_csv(deserved_csv, index=False)

    # Human readable summary text
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
            }) + "\n")
        else:
            f.write("(no data)\n")

    print(f"[OK] Wrote summaries:\n - {bucket_csv}\n - {regime_csv}\n - {deserved_csv}\n - {summary_txt}")

if __name__ == "__main__":
    main()
