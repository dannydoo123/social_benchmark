"""Audit evidence coverage without inventing missing metadata or public scores."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def audit(rows):
    fields = ["published_at", "community_id", "parent_id", "author_id_hash", "context_text"]
    models = sorted({str(r.get("model_id", "unknown")) for r in rows} | {"gpt-6-astra", "gpt-6.1-sol"})
    result = []
    for model in models:
        group = [r for r in rows if r.get("model_id") == model]
        sources = Counter(r.get("source_platform") or "unknown" for r in group)
        communities = Counter(f"{r.get('source_platform', 'unknown')}:{r['community_id']}" for r in group if r.get("community_id"))
        result.append({"model": model, "observations": len(group), "threads": len({r["thread_id"] for r in group if r.get("thread_id")}), "sources": dict(sources), "communities": dict(communities), "missing": {f: sum(not r.get(f) for r in group) for f in fields}, "status": "no_evidence" if not group else "coverage_unvalidated", "public_score": None})
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "rows": len(rows), "scope": "Coverage audit of the supplied corpus; not a representative public-opinion estimate.", "models": result}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="web/public/perspective-audit.json")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    Path(args.out).write_text(json.dumps(audit(rows), indent=2), encoding="utf-8")
