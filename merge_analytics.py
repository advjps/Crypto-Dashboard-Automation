#!/usr/bin/env python3
# merge_analytics.py  (updated for 10A flattened analytics)
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
    # accept any csv in analytics/ (previously some were signals_*.csv, some backtest_*.csv)
    files = sorted(glob.glob(os.path.join(ANALYTICS_DIR, "*.csv")))
    if not files:
        print(f"[INFO] No per-file analytics CSVs found in {ANALYTICS_DIR}/")
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=object)  # read as object to avoid coercion surprises
            df["__source_file"] = os.path.basename(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Skipped {f}: {e}")
    if not frames:
        return pd.DataFrame()
    # concat tolerant to different columns
    return pd.concat(frames, ignore_index=True, sort=False)

def normalize_col(df, colnames):
    """
    Given dataframe 'df' and a list of possible column names in priority,
    returns series and its actual column name using first existing column (case-insensitive).
    If not found returns (pd.Series with None, None)
    """
    if df is None or df.empty:
        return (pd.Series([None]*0), None)
    cols_lower = {c.lower(): c for c in df.columns}
    for c in colnames:
        if c.lower() in cols_lower:
            real = cols_lower[c.lower()]
            return (df[real], real)
    return (pd.Series([None]*len(df)), None)

def winrate(success, fail):
    denom = (success or 0) + (fail or 0)
    return (success / denom * 100.0) if denom > 0 else 0.0

def safe_int(x):
    try:
        return int(float(x))
    except Exception:
        return None

def parse_would_be_strong_col(val):
    if not val or (isinstance(val, float) and pd.isna(val)):
        return {}
    try:
        if isinstance(val, str):
            return json.loads(val)
        else:
            return dict(val)
    except Exception:
        # try eval-like fallback
        try:
            return json.loads(str(val).replace("'", '"'))
        except Exception:
            return {}

def summarize_bucket(df, bucket_name):
    signal_ser, signal_col = normalize_col(df, ["Signal", "signal"])
    outcome_ser, outcome_col = normalize_col(df, ["Outcome", "outcome"])
    conf_ser, conf_col = normalize_col(df, ["Confidence", "confidence"])
    if signal_col is None:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    mask = signal_ser.astype(str).str.strip().str.lower() == bucket_name.lower()
    sub = df[mask].copy()
    if sub.empty:
        return {"Total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "WinRate": 0.0, "AvgConfidence": 0.0}
    succ = 0; fail = 0; inconc = 0
    if outcome_col:
        out = sub[outcome_col].astype(str).str.strip().str.lower()
        succ = (out == "success").sum()
        fail = (out == "fail").sum()
        inconc = (out == "inconclusive").sum()
    wr = winrate(int(succ), int(fail))
    avg_conf = 0.0
    if conf_col:
        avg_conf = pd.to_numeric(sub[conf_col], errors="coerce").dropna().astype(float).mean()
        if pd.isna(avg_conf):
            avg_conf = 0.0
    return {"Total": int(len(sub)), "Success": int(succ), "Fail": int(fail), "Inconclusive": int(inconc), "WinRate": wr, "AvgConfidence": float(avg_conf or 0.0)}

def regime_summary(df):
    if df.empty:
        return pd.DataFrame()
    outcome_ser, outcome_col = normalize_col(df, ["Outcome", "outcome"])
    conf_ser, conf_col = normalize_col(df, ["Confidence", "confidence"])
    regime_ser, regime_col = normalize_col(df, ["Regime", "regime"])
    if regime_col is None:
        return pd.DataFrame()
    rows = []
    for regime, g in df.groupby(regime_ser.fillna("Unknown")):
        if regime in (None, "", float("nan")):
            regime = "Unknown"
        succ = fail = inc = 0
        if outcome_col:
            out = g[outcome_col].astype(str).str.strip().str.lower()
            succ = (out == "success").sum()
            fail = (out == "fail").sum()
            inc = (out == "inconclusive").sum()
        wr = winrate(int(succ), int(fail))
        avg_conf = 0.0
        if conf_col:
            avg_conf = pd.to_numeric(g[conf_col], errors="coerce").dropna().astype(float).mean()
            if pd.isna(avg_conf):
                avg_conf = 0.0
        rows.append({
            "Regime": str(regime),
            "Total": int(len(g)),
            "Success": int(succ),
            "Fail": int(fail),
            "Inconclusive": int(inc),
            "WinRate": wr,
            "AvgConfidence": float(avg_conf or 0.0)
        })
    return pd.DataFrame(rows).sort_values(["Regime"])

def deserved_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = []
    # find columns that could indicate deserved strong flags
    possible_flags = [c for c in df.columns if c.lower().startswith("deserved") or c.lower().startswith("deservedstrong")]
    if not possible_flags:
        return pd.DataFrame()
    outcome_ser, outcome_col = normalize_col(df, ["Outcome", "outcome"])
    conf_ser, conf_col = normalize_col(df, ["Confidence", "confidence"])
    for flag_col in possible_flags:
        try:
            sub = df[pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int) == 1].copy()
        except Exception:
            # fallback: check string "1" or "true"
            sub = df[sub := (df[flag_col].astype(str).str.strip().str.lower().isin(["1","true"]))].copy()
        succ = fail = inc = 0
        if not sub.empty and outcome_col:
            out_s = sub[outcome_col].astype(str).str.strip().str.lower()
            succ = (out_s == "success").sum()
            fail = (out_s == "fail").sum()
            inc = (out_s == "inconclusive").sum()
        wr = winrate(int(succ), int(fail))
        avg_conf = 0.0
        if not sub.empty and conf_col:
            avg_conf = pd.to_numeric(sub[conf_col], errors="coerce").dropna().astype(float).mean()
            if pd.isna(avg_conf):
                avg_conf = 0.0
        out.append({
            "Group": flag_col,
            "Total": int(len(sub)),
            "Success": int(succ),
            "Fail": int(fail),
            "Inconclusive": int(inc),
            "WinRate": wr,
            "AvgConfidence": float(avg_conf or 0.0)
        })
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out)

def expand_would_be_strong(df):
    """
    If a WOULD_BE_STRONG column exists (stringified JSON), expand into WOULD__missing_points and top components.
    """
    col_candidates = [c for c in df.columns if c.upper() == "WOULD_BE_STRONG" or c.upper().endswith("WOULD_BE_STRONG")]
    if not col_candidates:
        return df
    col = col_candidates[0]
    missing_list = []
    top1 = []
    top2 = []
    top3 = []
    parsed_vals = []
    for v in df[col].fillna("").tolist():
        parsed = parse_would_be_strong_col(v)
        parsed_vals.append(parsed)
        missing_list.append(parsed.get("missing_points") if isinstance(parsed, dict) else None)
        tops = parsed.get("top_missing_components") if isinstance(parsed, dict) else []
        if isinstance(tops, list) and tops:
            top1.append(tops[0].get("component") + ":" + str(tops[0].get("gap")) if isinstance(tops[0], dict) else str(tops[0]))
            top2.append(tops[1].get("component") + ":" + str(tops[1].get("gap")) if len(tops) > 1 and isinstance(tops[1], dict) else (tops[1] if len(tops) > 1 else None))
            top3.append(tops[2].get("component") + ":" + str(tops[2].get("gap")) if len(tops) > 2 and isinstance(tops[2], dict) else (tops[2] if len(tops) > 2 else None))
        else:
            top1.append(None); top2.append(None); top3.append(None)

    df["WOULD__missing_points"] = missing_list
    df["WOULD__top1"] = top1
    df["WOULD__top2"] = top2
    df["WOULD__top3"] = top3
    return df

def main():
    df = load_all_csvs()
    if df.empty:
        print("[INFO] Nothing to merge. Exiting.")
        return

    # some CSVs may include WOULD_BE_STRONG as JSON string; expand it for analytics convenience
    df = expand_would_be_strong(df)

    # normalize common column names to consistent casing (not strictly required but convenient)
    # i.e., ensure there's 'Signal', 'Outcome', 'Confidence', 'Regime' present by case-insensitive rename
    col_map = {}
    for desired in ["Signal","Outcome","Confidence","Regime","Coin","Estimated_profit","signal_time_utc","signal_time_ist"]:
        # find existing
        matches = [c for c in df.columns if c.lower() == desired.lower()]
        if matches:
            col_map[matches[0]] = desired
    if col_map:
        df = df.rename(columns=col_map)

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
