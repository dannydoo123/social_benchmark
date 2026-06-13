"""Generate compact SQL to seed Supabase from a snapshot JSON.

Used for the one-off initial seed applied via the Supabase MCP (privileged,
RLS-bypassing). For routine reloads use `python -m social_benchmark.pipeline.cli
load-supabase` with a service-role key instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SNAP = Path("web/public/snapshot.json")
OUT = Path("scripts/_seed")


def q(value) -> str:
    if value is None or value == "":
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def jq(value) -> str:
    return "'" + json.dumps(value).replace("'", "''") + "'::jsonb"


def chunked(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def main() -> None:
    snap = json.loads(SNAP.read_text(encoding="utf-8"))
    sid = snap["snapshot_id"]
    OUT.mkdir(parents=True, exist_ok=True)

    # Snapshot meta + reset (single live snapshot; children cascade-delete).
    meta = []
    meta.append("delete from public.score_snapshots;")
    meta.append(
        "insert into public.score_snapshots"
        "(snapshot_id,generated_at,is_current,gates,tier_thresholds,corpus,overall,providers,coverage,methodology) values("
        f"{q(sid)},{q(snap['generated_at'])},true,{jq(snap['gates'])},"
        f"{jq(snap.get('tier_thresholds', {}))},{jq(snap['corpus'])},{jq(snap['overall'])},"
        f"{jq(snap.get('providers', []))},{jq(snap.get('coverage', {}))},{jq(snap['methodology'])});"
    )
    (OUT / "00_meta.sql").write_text("\n".join(meta), encoding="utf-8")

    # Leaderboard.
    lb_vals = []
    for r in snap["leaderboard"]:
        ci = r.get("ci") or [None, None]
        lb_vals.append(
            f"({q(sid)},{q(r['model_id'])},{q(r.get('provider_id',''))},{q(r['aspect'])},"
            f"{q(r['score'])},{q(ci[0])},{q(ci[1])},{q(r.get('ess'))},{q(r.get('weighted_n'))},"
            f"{q(r.get('n_observations'))},{q(r.get('n_threads'))},{q(r.get('n_authors'))},"
            f"{q(r.get('firsthand_ratio'))},{q(r.get('human_share'))},{jq(r.get('warnings',[]))},"
            f"{q(bool(r.get('publishable')))},{q(r.get('tier'))})"
        )
    cols = ("snapshot_id,model_id,provider_id,aspect,score,ci_low,ci_high,ess,weighted_n,"
            "n_observations,n_threads,n_authors,firsthand_ratio,human_share,warnings,publishable,tier")
    parts = []
    for i, batch in enumerate(chunked(lb_vals, 200)):
        parts.append(f"insert into public.leaderboard_rows({cols}) values\n" + ",\n".join(batch) + ";")
    (OUT / "01_leaderboard.sql").write_text("\n".join(parts), encoding="utf-8")

    # Tasks.
    task_vals = []
    for task, rows in snap["tasks"].items():
        for r in rows:
            ci = r.get("ci") or [None, None]
            task_vals.append(
                f"({q(sid)},{q(task)},{q(r['model_id'])},{q(r['aspect'])},{q(r['score'])},"
                f"{q(ci[0])},{q(ci[1])},{q(r.get('ess'))},{q(r.get('tier'))})"
            )
    tcols = "snapshot_id,task,model_id,aspect,score,ci_low,ci_high,ess,tier"
    parts = []
    for batch in chunked(task_vals, 300):
        parts.append(f"insert into public.task_rows({tcols}) values\n" + ",\n".join(batch) + ";")
    (OUT / "02_tasks.sql").write_text("\n".join(parts), encoding="utf-8")

    # Evidence.
    ev_vals = []
    for key, samples in snap["evidence"].items():
        model_id, _, aspect = key.partition("|")
        for ordi, s in enumerate(samples):
            ev_vals.append(
                f"({q(sid)},{q(model_id)},{q(aspect)},{ordi},{q(s.get('span',''))},"
                f"{q(s.get('url',''))},{q(s.get('polarity'))},{q(s.get('evidence_type'))},"
                f"{q(bool(s.get('firsthand')))},{q(bool(s.get('human_labeled')))})"
            )
    ecols = "snapshot_id,model_id,aspect,ord,span,url,polarity,evidence_type,firsthand,human_labeled"
    parts = []
    for batch in chunked(ev_vals, 220):
        parts.append(f"insert into public.evidence_samples({ecols}) values\n" + ",\n".join(batch) + ";")
    (OUT / "03_evidence.sql").write_text("\n".join(parts), encoding="utf-8")

    sizes = {p.name: p.stat().st_size for p in sorted(OUT.glob("*.sql"))}
    counts = {
        "leaderboard": len(lb_vals),
        "tasks": len(task_vals),
        "evidence": len(ev_vals),
    }
    print(json.dumps({"snapshot_id": sid, "sizes": sizes, "counts": counts}))


if __name__ == "__main__":
    sys.exit(main())
