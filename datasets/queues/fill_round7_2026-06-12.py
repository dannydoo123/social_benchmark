"""Apply LLM-reviewer decisions to the round-7 labeling queue."""
import csv

QUEUE = "datasets/queues/targeted_training_review_round7_2026-06-12.csv"
OUT = "datasets/queues/targeted_training_review_round7_2026-06-12_filled.csv"

EXCLUDED = {
    0: "contest subscription banter, no claim",
    6: "book project process intro, no model-specific claim",
    7: "llama.cpp Android build infrastructure, no model claim",
    8: "voice app feature listing, no claim",
    10: "span describes Claude/Codex workflow; gemini incidental",
    12: "latency question, no claim",
    17: "duplicate span; labeled under index 64",
    18: "router product description, no model claim",
    19: "incident-PR vendor pitch, no model quality claim",
    21: "memory curation workflow tip, no model quality claim",
    22: "skill tooling description, no model quality claim",
    23: "tooling question, no claim",
    26: "duplicate book project intro",
    28: "router product description, no model claim",
    30: "A/B testing framework description, no model claim",
    31: "routing-efficiency generic claim, vendor rationale",
    32: "router Show HN pitch, no model quality claim",
    33: "venue agent vendor description, no model claim",
    34: "venue agent vendor description, no model claim",
    35: "duplicate span; labeled under index 9",
    36: "voice app feature listing, no claim",
    37: "voice app feature listing, no claim",
    38: "tool-calling format speculation, no quality claim",
    39: "voice plugin description, not model quality",
    40: "duplicate latency question",
    43: "duplicate span; labeled under index 42",
    46: "vendor reassurance about underlying models, no claim",
    47: "memory add-on overhead commentary, not model quality",
    49: "duplicate vendor reassurance",
    50: "duplicate span; labeled under index 13",
    51: "CLI token optimization description, no model claim",
    52: "agent traffic anecdote, no model quality claim",
    53: "automation driver vendor usage note, no claim",
    54: "automation driver vendor usage note, no claim",
    55: "usage-with-tools note, no claim",
    56: "venue agent vendor description, no model claim",
    58: "venue agent vendor description, no model claim",
    59: "two's company quip, no quality claim",
    60: "Screeps musing, no claim",
    61: "benchmark methodology shortcut, no claim",
    63: "session analysis setup, no claim",
    65: "future testing plans, no claim",
    66: "duplicate Screeps musing",
    67: "benchmark methodology note, no claim",
    70: "workflow adoption commentary, no model claim",
    71: "automation driver vendor usage note, no claim",
    75: "Google product integration critique, not model quality",
    78: "web app connection setup note, no claim",
    79: "supported-backend listing, no claim",
}

# index: (task, aspect, evidence, polarity, firsthand, model_override, note)
LABELS = {
    1: ("coding", "regression_stability", "bug_regression_report", -1, True, None, "feels slower over time, though does more per prompt"),
    2: ("general", "value", "pricing_value_comment", -1, True, None, "models burn through credits quickly"),
    3: ("coding", "task_fit", "hearsay", 1, False, "claude-opus-4.7", "point it at repo, it knows how to set up"),
    4: ("agents", "task_fit", "firsthand_usage", 1, True, "claude-opus", "main agent often benefits from Opus; router vendor"),
    5: ("general", "satisfaction", "firsthand_usage", 1, True, None, "qlora fine-tune learned my voice perfectly"),
    9: ("agents", "satisfaction", "benchmark_anecdote", 1, True, "claude-opus-4.5", "most dominant in LLM Skirmish, weak round 1"),
    11: ("coding", "task_fit", "firsthand_usage", 1, True, None, "more success with Claude planning + Codex validation"),
    13: ("general", "task_fit", "firsthand_usage", -1, True, None, "hoped Gemini would nail it; had to build it themselves"),
    14: ("roleplay", "satisfaction", "firsthand_usage", 2, True, "mistral-small-3.2", "Spanish conversation partner works incredibly well"),
    15: ("coding", "satisfaction", "hearsay", -1, False, "kimi-2.5", "dismissed as not frontier"),
    16: ("coding", "satisfaction", "benchmark_anecdote", -1, True, "claude-opus-4.8", "performed quite poorly vs specialized tool; vendor 8x claim"),
    20: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "impressed with CLI video editing via Remotion"),
    24: ("general", "satisfaction", "release_update_reaction", 1, True, "qwen-3.5", "running on $300 Android within hours, 8 tok/s"),
    25: ("coding", "trust_reliability", "firsthand_usage", -1, True, None, "forgets working directory mid-task"),
    27: ("general", "value", "pricing_value_comment", 1, True, None, "almost never runs out of weekly usage on $100 plan"),
    29: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "working reliable MCP server in an hour of tokens"),
    41: ("agents", "task_fit", "firsthand_usage", 1, True, None, "tiny unix-pipeline one-task agents work well"),
    42: ("api_developer_workflow", "value", "pricing_value_comment", -1, True, None, "annoyingly expensive when cache breaks on screenshots"),
    44: ("coding", "trust_reliability", "firsthand_usage", -1, True, None, "isolates fallout of runaway sessions in worktrees"),
    45: ("coding", "task_fit", "firsthand_usage", -1, True, None, "auto-fix fails on subtle invariant bugs"),
    48: ("coding", "task_fit", "firsthand_usage", -1, True, None, "auto-fix fails on subtle invariant bugs"),
    57: ("writing", "task_fit", "firsthand_usage", 1, True, None, "useful at calling out ChatGPT's tells"),
    62: ("agents", "value", "pricing_value_comment", -1, True, "claude-sonnet", "less capability-per-dollar than open models; router vendor bias"),
    64: ("multimodal", "task_fit", "firsthand_usage", -1, True, None, "subpar multi-speaker transcription vs Whisper"),
    68: ("coding", "trust_reliability", "firsthand_usage", -1, True, None, "only trusted inside a git directory with easy revert"),
    69: ("agents", "task_fit", "firsthand_usage", 1, True, None, "Flash sweet spot for agentic actions; Flash-Lite hit or miss"),
    72: ("coding", "satisfaction", "benchmark_anecdote", -1, True, "gemini-3.5-pro", "performed quite poorly vs specialized tool; vendor 8x claim"),
    73: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "switched to Opencode, experience was rough"),
    74: ("coding", "satisfaction", "hearsay", 1, False, None, "40-year engineer loves seeing what it can do"),
    76: ("agents", "satisfaction", "hearsay", 1, False, None, "recursive self-writing agent demo"),
    77: ("agents", "satisfaction", "benchmark_anecdote", 1, True, None, "won the RTS match vs Opus"),
}


def main() -> None:
    with open(QUEUE, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert len(rows) == 80, len(rows)
    missing = [i for i in range(len(rows)) if i not in EXCLUDED and i not in LABELS]
    assert not missing, missing
    overlap = set(EXCLUDED) & set(LABELS)
    assert not overlap, overlap

    for index, row in enumerate(rows):
        row["reviewed_flag"] = "True"
        if index in EXCLUDED:
            row["human_excluded_from_scoring"] = "True"
            row["human_exclusion_reason"] = EXCLUDED[index]
            continue
        task, aspect, evidence, polarity, firsthand, model, note = LABELS[index]
        row["human_excluded_from_scoring"] = "False"
        row["human_task_category"] = task
        row["human_aspect_category"] = aspect
        row["human_evidence_type"] = evidence
        row["human_polarity_score"] = str(polarity)
        row["human_firsthand_flag"] = "True" if firsthand else "False"
        row["human_model_id"] = model or row.get("model_id", "")
        row["human_provider_id"] = row.get("provider_id", "")
        row["human_notes"] = note

    with open(OUT, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    print(f"excluded={len(EXCLUDED)} labeled={len(LABELS)}")


if __name__ == "__main__":
    main()
