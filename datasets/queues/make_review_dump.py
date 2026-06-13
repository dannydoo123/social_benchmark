"""Write a compact review dump (index, machine labels, span text) for a queue CSV."""
import csv
import sys

queue_path, out_path = sys.argv[1], sys.argv[2]

with open(queue_path, encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

with open(out_path, "w", encoding="utf-8") as handle:
    for index, row in enumerate(rows):
        handle.write(f"=== {index} {row['review_id']}\n")
        handle.write(
            "M: {model} | {task} {aspect} {evidence} {polarity} fh={fh}\n".format(
                model=row.get("model_id", ""),
                task=row.get("task_category", ""),
                aspect=row.get("aspect_category", ""),
                evidence=row.get("evidence_type", ""),
                polarity=row.get("polarity_score", ""),
                fh=row.get("firsthand_flag", ""),
            )
        )
        handle.write(f"T: {row.get('evidence_text', '')[:450]}\n")
print(f"wrote {len(rows)} rows -> {out_path}")
