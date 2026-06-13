import json
import sys

path = sys.argv[1]
d = json.load(open(path))
for field, info in d["fields"].items():
    print("==", field)
    for thr, row in info["thresholds"].items():
        print(f"   t={thr} p={row['precision']:.3f} cov={row['coverage']:.3f} n={row['covered']}")
