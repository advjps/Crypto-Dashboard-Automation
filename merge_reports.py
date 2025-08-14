# merge_reports.py (V2 - with Timestamp)
import os
import glob
from datetime import datetime

def merge_reports():
    """
    Finds all .txt files in the 'reports' folder and merges them
    into a single, timestamped text file for analysis.
    """
    reports_folder = 'reports'
    
    # Create a timestamp for the output filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f'All_Reports_Combined_{timestamp}.txt'

    if not os.path.isdir(reports_folder):
        print(f"Error: The '{reports_folder}' directory was not found.")
        return

    report_files = glob.glob(os.path.join(reports_folder, '*.txt'))

    if not report_files:
        print(f"No .txt report files found in the '{reports_folder}' folder.")
        return

    print(f"Found {len(report_files)} reports. Merging into '{output_filename}'...")

    # Sort files to ensure a consistent order
    report_files.sort()

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for filename in report_files:
            outfile.write(f"============================================================\n")
            outfile.write(f"====== CONTENTS OF: {os.path.basename(filename)} ======\n")
            outfile.write(f"============================================================\n\n")
            
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    print(f"SUCCESS! All reports have been combined into '{output_filename}'.")

if __name__ == "__main__":
    merge_reports()
