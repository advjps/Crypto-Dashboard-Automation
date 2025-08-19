# merge_reports.py (V4 – GitHub Actions Ready + Global Summary)
import os
import glob
from datetime import datetime
import pytz
import re
from collections import defaultdict

REPORTS_FOLDER = 'backtest_reports'
MERGED_REPORTS_FOLDER = 'merged_reports'

SECTION_TITLES = ["STRONG BUY SIGNALS", "BUY SIGNALS", "STRONG SELL SIGNALS", "SELL SIGNALS"]

# Split on 2+ spaces to read the fixed-width table safely
SPLIT_RE = re.compile(r"\s{2,}")

def merge_reports():
    """
    Finds all .txt files in 'backtest_reports', merges them into a single IST-timestamped file
    in 'merged_reports', and appends a GLOBAL SUMMARY (counts, win rates, avg confidence).
    """
    os.makedirs(MERGED_REPORTS_FOLDER, exist_ok=True)

    ist_tz = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist_tz).strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'All_Reports_Combined_{timestamp}.txt'
    output_filepath = os.path.join(MERGED_REPORTS_FOLDER, output_filename)

    if not os.path.isdir(REPORTS_FOLDER):
        print(f"Error: The '{REPORTS_FOLDER}' directory was not found.")
        return

    report_files = glob.glob(os.path.join(REPORTS_FOLDER, '*.txt'))
    if not report_files:
        print(f"No .txt report files found in the '{REPORTS_FOLDER}' folder.")
        return

    # Natural-ish sort by datetime in filename if present
    report_files.sort()

    # Aggregates across all files
    agg = {
        # section name -> counters
        "STRONG BUY": {"total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "conf_sum": 0},
        "BUY":        {"total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "conf_sum": 0},
        "STRONG SELL":{"total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "conf_sum": 0},
        "SELL":       {"total": 0, "Success": 0, "Fail": 0, "Inconclusive": 0, "conf_sum": 0},
    }

    def update_agg(section_key: str, row_map: dict):
        # Outcome
        outcome = (row_map.get("Outcome") or "").strip()
        if outcome not in ("Success", "Fail", "Inconclusive"):
            return
        agg[section_key]["total"] += 1
        agg[section_key][outcome] += 1
        # Confidence
        conf = row_map.get("Confidence")
        try:
            if conf not in ("", None):
                agg[section_key]["conf_sum"] += int(round(float(conf)))
        except Exception:
            pass

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for filename in report_files:
            outfile.write("============================================================\n")
            outfile.write(f"====== CONTENTS OF: {os.path.basename(filename)} ======\n")
            outfile.write("============================================================\n\n")

            try:
                with open(filename, 'r', encoding='utf-8') as infile:
                    contents = infile.read()
            except Exception as e:
                outfile.write(f"[ERROR] Could not read file: {e}\n\n")
                continue

            outfile.write(contents)
            outfile.write("\n\n")

            # --------- Parse this file for summary aggregation ----------
            # We’ll scan line by line, remember the current section, find the header row,
            # build a column name -> index mapping using 2+ space splits, then parse rows.
            lines = contents.splitlines()
            cur_section = None
            header_cols = None  # list of column names in this section
            for line in lines:
                s = line.strip()

                # Section detection
                if s.startswith("--- ") and s.endswith(" ---"):
                    title = s[4:-4].strip().upper()
                    cur_section = None
                    header_cols = None
                    if title in SECTION_TITLES:
                        # Map to our agg keys
                        if "STRONG BUY" in title:   cur_section = "STRONG BUY"
                        elif "STRONG SELL" in title: cur_section = "STRONG SELL"
                        elif title == "BUY SIGNALS": cur_section = "BUY"
                        elif title == "SELL SIGNALS": cur_section = "SELL"
                    continue

                # Skip empties and non-sections
                if not s or cur_section is None:
                    continue

                # (None) means no rows for this section
                if s == "(None)":
                    cur_section = None
                    header_cols = None
                    continue

                # Capture header row (starts with "Coin  Signal  Confidence ...")
                if s.startswith("Coin") and "Outcome" in s:
                    header_cols = SPLIT_RE.split(s)
                    continue

                # Data rows: must have a header reference and not be "Coin..." again
                if header_cols:
                    # We expect data rows to have values; split by 2+ spaces
                    parts = SPLIT_RE.split(s)
                    # Ignore if parts don’t match header length (best-effort parsing)
                    if len(parts) < len(header_cols):
                        continue
                    row_map = {header_cols[i]: parts[i] for i in range(len(header_cols))}
                    update_agg(cur_section, row_map)

    # ------------- Append GLOBAL SUMMARY -------------
    with open(output_filepath, 'a', encoding='utf-8') as outfile:
        outfile.write("\n\n==================== GLOBAL SUMMARY (All Reports) ====================\n")
        outfile.write("Section        Total   Success  Fail   Inconcl.  WinRate   AvgConf\n")
        outfile.write("---------------------------------------------------------------------\n")
        for key in ["STRONG BUY", "BUY", "STRONG SELL", "SELL"]:
            d = agg[key]
            total = d["total"]
            succ = d["Success"]
            fail = d["Fail"]
            inc  = d["Inconclusive"]
            win_rate = (succ / max(1, (succ + fail))) * 100.0  # exclude Inconclusive from WR
            avg_conf = (d["conf_sum"] / total) if total > 0 else 0.0
            outfile.write(
                f"{key.ljust(13)}  {str(total).rjust(5)}   {str(succ).rjust(7)}  {str(fail).rjust(5)}  {str(inc).rjust(8)}"
                f"   {win_rate:6.2f}%   {avg_conf:7.2f}\n"
            )
        outfile.write("---------------------------------------------------------------------\n")
        outfile.write("Notes:\n")
        outfile.write("• WinRate excludes Inconclusive rows (first-touch TP/SL not reached).\n")
        outfile.write("• AvgConf = average Confidence across rows in section (0–100).\n")
        outfile.write("• Counts/averages are approximations if a row was malformed.\n")

    print(f"SUCCESS! All reports have been combined into '{output_filepath}'.")
    print("A GLOBAL SUMMARY section has been appended at the end.")

if __name__ == "__main__":
    merge_reports()
