# merge_reports.py (V3 - GitHub Actions Ready)
import os
import glob
from datetime import datetime
import pytz

def merge_reports():
    """
    Finds all .txt files in the 'backtest_reports' folder and merges them
    into a single, timestamped text file in the 'merged_reports' folder.
    """
    # --- UPDATED: Define source and output folders ---
    REPORTS_FOLDER = 'backtest_reports'
    MERGED_REPORTS_FOLDER = 'merged_reports'

    # --- NEW: Create the output folder if it doesn't exist ---
    os.makedirs(MERGED_REPORTS_FOLDER, exist_ok=True)
    
    # Use IST for the timestamp in the filename
    ist_tz = pytz.timezone("Asia/Kolkata")
    timestamp = datetime.now(ist_tz).strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'All_Reports_Combined_{timestamp}.txt'
    
    # --- UPDATED: Construct the full output path ---
    output_filepath = os.path.join(MERGED_REPORTS_FOLDER, output_filename)

    if not os.path.isdir(REPORTS_FOLDER):
        print(f"Error: The '{REPORTS_FOLDER}' directory was not found.")
        return

    report_files = glob.glob(os.path.join(REPORTS_FOLDER, '*.txt'))

    if not report_files:
        print(f"No .txt report files found in the '{REPORTS_FOLDER}' folder.")
        return

    print(f"Found {len(report_files)} reports. Merging into '{output_filepath}'...")

    report_files.sort()

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for filename in report_files:
            outfile.write(f"============================================================\n")
            outfile.write(f"====== CONTENTS OF: {os.path.basename(filename)} ======\n")
            outfile.write(f"============================================================\n\n")
            
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    print(f"SUCCESS! All reports have been combined into '{output_filepath}'.")

if __name__ == "__main__":
    merge_reports()
