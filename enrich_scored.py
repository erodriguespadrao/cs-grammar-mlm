"""
Joins speaker, file, and folder columns from miami_preprocessed_cs_only.csv
into a scored pairs CSV using the sentence_id column as the key.

Usage:
    python enrich_scored.py --input pairs_scored_salazar.csv --output pairs_scored_salazar_enriched.csv
    python enrich_scored.py --input pairs_scored_kauf.csv    --output pairs_scored_kauf_enriched.csv
    python enrich_scored.py
"""

import csv
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input',   type=Path, default=Path('pairs_scored.csv'),
                    help='Scored pairs CSV to enrich')
parser.add_argument('--output',  type=Path, default=Path('pairs_scored_enriched.csv'),
                    help='Output path')
parser.add_argument('--cs_only', type=Path, default=Path('miami_preprocessed_cs_only.csv'),
                    help='CS-only CSV from preprocess_miami.py')
args = parser.parse_args()

# Load CS-only file
print(f"Loading {args.cs_only}...")
with open(args.cs_only, encoding="utf-8") as f:
    cs_rows = list(csv.DictReader(f))
print(f"  {len(cs_rows)} CS utterances")

# Load scored file
print(f"Loading {args.input}...")
with open(args.input, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    orig_fields = list(reader.fieldnames)
    scored = list(reader)
print(f"  {len(scored)} scored rows")

# Verify range
max_id = max(int(r["sentence_id"]) for r in scored)
if max_id >= len(cs_rows):
    raise ValueError(
        f"sentence_id {max_id} exceeds CS-only file length {len(cs_rows)}. "
        "Make sure you are using the correct cs_only file."
    )

# Add columns (avoid duplicates if already enriched)
new_fields = orig_fields.copy()
for col in ["speaker", "conversation", "folder"]:
    if col not in new_fields:
        new_fields.append(col)

print(f"Writing {args.output}...")
with open(args.output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=new_fields)
    writer.writeheader()
    for r in scored:
        sid = int(r["sentence_id"])
        cs  = cs_rows[sid]
        row = dict(r)
        row["speaker"] = cs["speaker"]
        row["conversation"] = cs["file"]
        row["folder"]  = cs["folder"]
        writer.writerow(row)

print(f"Completed. {len(scored)} rows written to {args.output}")
print("Unique speakers:", len({cs_rows[int(r['sentence_id'])]['speaker'] for r in scored}))
print("Unique conversations: ", len({cs_rows[int(r['sentence_id'])]['file'] for r in scored}))
print("\n Run analyse_mixed.R in RStudio with this file as INPUT_FILE")
