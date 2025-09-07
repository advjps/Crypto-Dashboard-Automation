#!/usr/bin/env python3
# merge_analytics.py  (robust aggregator)
# Usage: python merge_analytics.py

import os
import glob
import json
import re
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
    # Load all CSVs in analytics/, EXCLUDING the merged all_signals.csv itself
    files = sorted(glob.glob(os.path.join(ANALYTICS_DIR, "*.csv")))
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(ALL_SIGNALS_CSV)]
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
    returns series using first existing column (case-insensitive) and sets .name appropriately.
    If none present, returns an empty Series with name=None.
    """
    if df is None or df.empty:
        return pd.Series([], name=None)
    cols_lower = {c.lower(): c for c in df.columns}
    for c in colnames:
        if c is None:
            continue
        if c.lower() in cols_lower:
            real = cols_lower[c.lower()]
            s = df[real]
            s.name = real
            return s
    # not found -> return series of NAs
    s = pd.Series([None] * len(df))
    s.name = None
    return s

def winrate(success, fail):
    denom = (success or 0) + (fail or 0)
    return (success / denom * 100.0) if denom > 0 else 0.0

def summarize_bucket(df, bucket_name):
    # case-insensitive matching for Signal column
    signal_series = normalize_col(df, ["Signal", "signal"])
    if signal_series.name is None:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    outcome_series = normalize_col(df, ["Outcome", "outcome"])
    conf_series = normalize_col(df, ["Confidence", "confidence"])
    mask = signal_series.astype(str).str.strip().str.lower() == bucket_name.lower()
    sub = df[mask].copy()
    if sub.empty:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    # outcome mapping (be defensive)
    if outcome_series.name and outcome_series.name in sub.columns:
        succ = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "success").sum()
        fail = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "fail").sum()
        inconc = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "inconclusive").sum()
    else:
        succ = fail = inconc = 0
    wr = winrate(int(succ), int(fail))
    avg_conf = 0.0
    if conf_series.name and conf_series.name in sub.columns:
        avg_conf = pd.to_numeric(sub[conf_series.name], errors="coerce").dropna().mean() or 0.0
    return {"Total": int(len(sub)), "Success": int(succ), "Fail": int(fail), "Inconclusive": int(inconc), "WinRate": wr, "AvgConfidence": float(avg_conf)}

def regime_summary(df):
    outcome_series = normalize_col(df, ["Outcome", "outcome"])
    conf_series = normalize_col(df, ["Confidence", "confidence"])
    regime_series = normalize_col(df, ["Regime", "regime"])
    if df.empty or regime_series.name is None:
        return pd.DataFrame()
    rows = []
    for regime, g in df.groupby(regime_series.name):
        reg_label = regime if regime not in (None, "", float("nan")) else "Unknown"
        if outcome_series.name and outcome_series.name in g.columns:
            succ = (g[outcome_series.name].astype(str).str.strip().str.lower() == "success").sum()
            fail = (g[outcome_series.name].astype(str).str.strip().str.lower() == "fail").sum()
            inc = (g[outcome_series.name].astype(str).str.strip().str.lower() == "inconclusive").sum()
        else:
            succ = fail = inc = 0
        wr = winrate(int(succ), int(fail))
        avg_conf = pd.to_numeric(g[conf_series.name], errors="coerce").dropna().mean() if conf_series.name and conf_series.name in g.columns else 0.0
        rows.append({
            "Regime": str(reg_label), "Total": int(len(g)), "Success": int(succ), "Fail": int(fail),
            "Inconclusive": int(inc), "WinRate": wr, "AvgConfidence": float(avg_conf or 0.0)
        })
    return pd.DataFrame(rows).sort_values(["Regime"])

def deserved_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = []
    # tolerant list of possible deserved-strong columns (case/underscore variations)
    possible_flags = ["DeservedStrongBuy", "deservedstrongbuy", "DeservedStrongSell", "deservedstrongsell",
                      "DeservedStrong_Buy", "DeservedStrong_Sell", "deserved_strong_buy", "deserved_strong_sell"]
    # find matches in df columns (case-insensitive)
    found = []
    for pc in possible_flags:
        matches = [c for c in df.columns if c.lower() == pc.lower()]
        for m in matches:
            if m not in found:
                found.append(m)
    if not found:
        return pd.DataFrame()
    outcome_series = normalize_col(df, ["Outcome", "outcome"])
    conf_series = normalize_col(df, ["Confidence", "confidence"])
    for flag_col in found:
        sub = df[pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        if sub.empty:
            succ = fail = inc = 0
            avg_conf = 0.0
        else:
            if outcome_series.name and outcome_series.name in sub.columns:
                succ = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "success").sum()
                fail = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "fail").sum()
                inc = (sub[outcome_series.name].astype(str).str.strip().str.lower() == "inconclusive").sum()
            else:
                succ = fail = inc = 0
            avg_conf = pd.to_numeric(sub[conf_series.name], errors="coerce").dropna().mean() if conf_series.name and conf_series.name in sub.columns else 0.0

        out.append({
            "Group": flag_col, "Total": int(len(sub)), "Success": int(succ), "Fail": int(fail),
            "Inconclusive": int(inc), "WinRate": winrate(int(succ), int(fail)), "AvgConfidence": float(avg_conf or 0.0)
        })
    return pd.DataFrame(out)

def main():
    df = load_all_csvs()
    if df.empty:
        print("[INFO] Nothing to merge. Exiting.")
        return

    # Normalize common names to consistent columns for downstream code readability (do not delete originals)
    if "Signal" not in df.columns and "signal" in df.columns:
        df["Signal"] = df["signal"]
    if "Outcome" not in df.columns and "outcome" in df.columns:
        df["Outcome"] = df["outcome"]
    if "Confidence" not in df.columns and "confidence" in df.columns:
        df["Confidence"] = df["confidence"]
    if "Regime" not in df.columns and "regime" in df.columns:
        df["Regime"] = df["regime"]

    # Create binary FLAG__ columns globally if CONFLUENCE_FLAGS present
    if "CONFLUENCE_FLAGS" in df.columns:
        df["CONFLUENCE_FLAGS"] = df["CONFLUENCE_FLAGS"].fillna("")
        all_flags = set()
        for s in df["CONFLUENCE_FLAGS"].unique():
            if not s:
                continue
            parts = [p.strip() for p in str(s).split(";") if p.strip()]
            for p in parts:
                all_flags.add(p)
        for fl in sorted(all_flags):
            col = "FLAG__" + re.sub(r'[^0-9A-Za-z]+', '_', fl).strip('_')
            if col not in df.columns:
                df[col] = df["CONFLUENCE_FLAGS"].apply(lambda x: 1 if fl in (x or "") else 0)

    # Save merged all_signals.csv
    try:
        df.to_csv(ALL_SIGNALS_CSV, index=False)
        print(f"[OK] Wrote {ALL_SIGNALS_CSV} with {len(df)} rows.")
    except Exception as e:
        print(f"[ERROR] Could not write {ALL_SIGNALS_CSV}: {e}")
        return

    # --- Global bucket summary (by Signal) ---
    buckets = ["Strong Buy", "Buy", "Strong Sell", "Sell"]
    rows = []
    for b in buckets:
        rows.append({"Section": b, **summarize_bucket(df, b)})
    bucket_df = pd.DataFrame(rows)

    # --- Regime summary ---
    regime_df = regime_summary(df)

    # --- DeservedStrong summaries (analysis-only) ---
    deserved_df = deserved_summary(df)

    # Save CSV summaries
    stamp = ist_timestamp()
    bucket_csv = os.path.join(ANALYTICS_DIR, f"summary_by_signal_{stamp}.csv")
    regime_csv = os.path.join(ANALYTICS_DIR, f"summary_by_regime_{stamp}.csv")
    deserved_csv = os.path.join(ANALYTICS_DIR, f"summary_deserved_{stamp}.csv")
    bucket_df.to_csv(bucket_csv, index=False)
    regime_df.to_csv(regime_csv, index=False)
    deserved_df.to_csv(deserved_csv, index=False)

    # Human-readable summary.txt
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
