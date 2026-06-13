"""Apply LLM-reviewer decisions to the round-6 labeling queue."""
import csv

QUEUE = "datasets/queues/targeted_training_review_round6_2026-06-12.csv"
OUT = "datasets/queues/targeted_training_review_round6_2026-06-12_filled.csv"

EXCLUDED = {
    0: "telemetry/credits vendor reply, no model quality claim",
    4: "app prompt tuning note, no quality claim",
    6: "multi-model workflow description, no quality claim",
    11: "effort-param doc speculation, no quality claim",
    13: "duplicate span; labeled under index 55",
    14: "Show HN security fixes, no model quality claim",
    16: "ChatGPT product/account behavior complaint, not model quality",
    17: "Show HN infrastructure description, no claim",
    19: "vendor app description, no quality claim",
    22: "security tooling vendor pitch, no model quality claim",
    25: "duplicate span; labeled under index 26",
    27: "supported-model listing, no claim",
    28: "pricing plan pitch, no model quality claim",
    30: "AGI speculation, no specific model quality claim",
    40: "span describes Grok DJ; claude misattributed",
    41: "duplicate span; labeled under index 2",
    45: "vendor email product pitch, no model quality claim",
    46: "duplicate span; labeled under index 2",
    47: "cloud dev product pitch, no model quality claim",
    50: "comparison caveat, no resolvable claim",
    54: "duplicate span; labeled under index 49",
    56: "multi-model app motivation, no quality claim",
    57: "hook configuration advice, no claim",
    58: "ChatGPT product/account behavior complaint, not model quality",
    60: "vendor product pitch, no model quality claim",
    61: "vendor product pitch, no model quality claim",
    62: "vendor product pitch, no model quality claim",
    63: "retrieval infrastructure description, no model claim",
    64: "duplicate span; labeled under index 113",
    66: "product branding complaint, no model quality claim",
    67: "tool/model usage statement, no claim",
    71: "self-hosting setup question, no claim",
    77: "AI watermark observation, no quality claim",
    78: "duplicate span; labeled under index 12",
    80: "duplicate span; labeled under index 79",
    83: "app prompt tuning note, no quality claim",
    84: "duplicate comparison caveat, no claim",
    86: "duplicate span; labeled under index 38",
    87: "release anticipation commentary, no quality claim",
    89: "duplicate multi-model workflow description",
    90: "API caching cost note, not model quality",
    91: "duplicate API caching cost note",
    92: "cache warning product question, no claim",
    94: "ChatGPT product/account behavior complaint, not model quality",
    95: "duplicate span; labeled under index 82",
    96: "cloud dev product pitch, no model quality claim",
    97: "vendor product pitch, no model quality claim",
    98: "computer-use product listing, no claim",
    99: "music taste banter, no quality claim",
    101: "duplicate span; labeled under index 69",
    102: "channel name banter, no claim",
    103: "quantization hosting note, no model quality claim",
    105: "compression speed question, no claim",
    107: "team plan pricing advice, no model quality claim",
    108: "supported-model listing, no claim",
    109: "vibe-coding culture commentary, no model quality claim",
    110: "duplicate span; labeled under index 79",
    111: "duplicate span; labeled under index 52",
    114: "skill-growth commentary, no model quality claim",
    116: "span evaluates o1; gpt-4o from title listing",
    117: "tool integration pitch, no model quality claim",
    118: "abuse report commentary, no quality claim",
    119: "cultural snark about generated languages, no claim",
    120: "local execution usage note, no claim",
    121: "gui/tui note, no claim",
    122: "multi-chat app description, no claim",
    124: "subscription recommendation, no claim",
    125: "multi-chat app description, no claim",
    126: "hosting hardware note, no claim",
    127: "vendor self-described biased pitch, no model claim",
    129: "duplicate span; labeled under index 51",
    130: "internal model mix description, no claim",
    131: "Show HN infrastructure description, no claim",
    134: "agent swarm tooling description, no claim",
    137: "AI-language speculation, no quality claim",
    138: "rhetorical snark question, no claim",
    141: "duplicate rhetorical snark question",
}

# index: (task, aspect, evidence, polarity, firsthand, model_override, note)
LABELS = {
    1: ("general", "task_fit", "firsthand_usage", -1, True, None, "memories fail to capture intent"),
    2: ("coding", "satisfaction", "firsthand_usage", -1, True, "deepseek-v4-pro", "incredibly slow via opencode+openrouter; possibly provider"),
    3: ("coding", "trust_reliability", "firsthand_usage", 2, True, None, "rarely fails me, used everywhere"),
    5: ("coding", "task_fit", "firsthand_usage", -1, True, "qwen-3.5", "limited success on long json-output prompts"),
    7: ("coding", "satisfaction", "hearsay", 1, False, None, "heard models haven't improved since 4.5"),
    8: ("coding", "satisfaction", "comparative_evaluation", 1, True, "gemini-2.5-pro", "did much better in one shot"),
    9: ("general", "satisfaction", "firsthand_usage", 1, True, None, "daily driver in routing mix"),
    10: ("general", "satisfaction", "firsthand_usage", -2, True, None, "DJ stream degenerated into gibberish, UFO obsession"),
    12: ("coding", "satisfaction", "firsthand_usage", 0, True, None, "primary model with occasional failures, falls back to opus"),
    15: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "used for spec'ing projects, dropped GPT"),
    18: ("general", "satisfaction", "comparative_evaluation", -1, True, "kimi-2.6", "very slow vs others"),
    20: ("coding", "trust_reliability", "bug_regression_report", -2, True, None, "errors out 15 times in a row on nonexistent paths"),
    21: ("agents", "value", "pricing_value_comment", 1, True, "deepseek-v4-pro", "long agentic sessions, higher quality for cheap"),
    23: ("coding", "task_fit", "firsthand_usage", 1, True, None, "used for frontend in routing mix"),
    24: ("coding", "value", "pricing_value_comment", 1, True, "kimi-2.6", "used when task should be free"),
    26: ("coding", "value", "pricing_value_comment", 1, True, "gpt-5.3-codex", "more than capable, quite a bit cheaper"),
    29: ("coding", "task_fit", "firsthand_usage", 1, True, None, "very good at md outline/checklist org mode"),
    31: ("long_context", "satisfaction", "benchmark_anecdote", -1, False, None, "65.1% needle retrieval at 1M vs 90.6% at 256K, from announcement"),
    32: ("long_context", "satisfaction", "benchmark_anecdote", -1, False, "gpt-5.4", "97.2% at 32k to 36.6% at 1M; SLM vendor framing"),
    33: ("agents", "satisfaction", "release_update_reaction", 1, False, "claude-opus-4.8", "launch made startup features irrelevant, memory way better"),
    34: ("general", "satisfaction", "benchmark_anecdote", -1, True, "claude-opus-4", "was poor even at first Baba is Eval level"),
    35: ("coding", "value", "pricing_value_comment", 1, True, "claude-opus-4.5", "$100 API spend produced 60+ apps; thinking profile"),
    36: ("agents", "task_fit", "firsthand_usage", 1, True, None, "works far better when it can self-heal in loop"),
    37: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "frontend 4x faster and prettier than doing it himself"),
    38: ("general", "regression_stability", "bug_regression_report", -2, True, None, "useful to near useless in a week, suspects backend fiddling"),
    39: ("coding", "satisfaction", "comparative_evaluation", 1, True, None, "no comparison velocity-wise; mid-stream corrections muddle review"),
    42: ("coding", "trust_reliability", "firsthand_usage", -2, True, None, "results bad, mostly wrong, wasted time and money"),
    43: ("multimodal", "task_fit", "firsthand_usage", 1, True, "gemini-3.1-flash-lite", "used for image understanding in routing mix"),
    44: ("coding", "task_fit", "firsthand_usage", 1, True, None, "claude code for everything but pure coding tasks"),
    48: ("agents", "satisfaction", "firsthand_usage", -1, True, None, "no agent smart enough to figure out layout unguided"),
    49: ("general", "regression_stability", "bug_regression_report", -1, True, None, "projects stopped working after two months fine"),
    51: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "vague-prompt iterate loop, planning not enforced"),
    52: ("general", "value", "pricing_value_comment", -2, True, None, "org removing access, bill 3x cloud spend"),
    53: ("coding", "satisfaction", "hearsay", 1, False, None, "recommended for analysing messy codebases"),
    55: ("general", "regression_stability", "bug_regression_report", -2, True, None, "clear decline, cannot complete tasks in one turn"),
    59: ("coding", "value", "pricing_value_comment", 1, True, None, "Max plan way more usage, code audits useful"),
    65: ("coding", "satisfaction", "firsthand_usage", 2, True, "claude-opus", "unbelievably proficient at spatial design since 4.6"),
    68: ("multimodal", "satisfaction", "firsthand_usage", 2, True, None, "DJ Gemini quite good, amazing voice"),
    69: ("long_context", "satisfaction", "benchmark_anecdote", 0, False, "gemini-3.1", "80% at 128K but still highly useful"),
    70: ("coding", "satisfaction", "hearsay", 2, False, None, "average-effort CC project beats median programmer"),
    72: ("general", "trust_reliability", "hearsay", -1, False, None, "CoT summarizer prompt-injectable, demonstrated"),
    73: ("coding", "task_fit", "firsthand_usage", -1, True, None, "amazing bootstrapping, breaks down on mature codebases"),
    74: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "used for coding in personal mix"),
    75: ("long_context", "satisfaction", "firsthand_usage", 1, True, None, "impressed, large context useful for long projects"),
    76: ("general", "task_fit", "firsthand_usage", -1, True, None, "cannot explain own plugin install"),
    79: ("coding", "satisfaction", "hearsay", 1, False, None, "lots of people seem to prefer Claude Code"),
    81: ("general", "satisfaction", "firsthand_usage", 1, True, "claude-opus-4.7", "used for basically everything personal"),
    82: ("general", "satisfaction", "comparative_evaluation", 1, True, "claude-opus-4.7", "parity with GPT 5.5, no clear preference"),
    85: ("general", "satisfaction", "hearsay", 1, False, "gpt-5.5", "considering it given all the positive hearsay"),
    88: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "slow outdated vms, cannot standardize environments"),
    93: ("coding", "satisfaction", "comparative_evaluation", -1, True, None, "team switched to Codex for speed and steerability"),
    100: ("general", "task_fit", "hearsay", -1, False, None, "markdown memories fail, still net useful"),
    104: ("general", "regression_stability", "bug_regression_report", -2, True, None, "basically nerfed, downgraded and cancelling"),
    106: ("general", "value", "pricing_value_comment", -2, True, None, "Pro subscription completely unusable with limits"),
    112: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "/compact painfully slow"),
    113: ("coding", "regression_stability", "bug_regression_report", -1, True, None, "seems to be getting worse, still beats free options"),
    115: ("coding", "satisfaction", "comparative_evaluation", -1, True, None, "worse than degraded Claude per author"),
    123: ("coding", "developer_ergonomics", "firsthand_usage", 1, True, None, "VPS + mobile /rc workflow really handy"),
    128: ("general", "satisfaction", "firsthand_usage", 2, True, None, "100x'd my work since team access"),
    132: ("agents", "satisfaction", "firsthand_usage", 0, True, None, "workflow dependence concern amid degradation"),
    133: ("agents", "satisfaction", "firsthand_usage", -2, True, None, "local models like Kimi barely functional"),
    135: ("general", "task_fit", "firsthand_usage", 0, True, None, "extremely good at some things, actively terrible at others"),
    136: ("general", "value", "pricing_value_comment", -1, True, None, "hates session restrictions on x5 plan"),
    139: ("general", "value", "pricing_value_comment", -1, False, None, "competitor pitch re Claude UI rate limits"),
    140: ("general", "satisfaction", "firsthand_usage", -2, True, None, "DJ stream stuck repeating ad infinitum"),
    142: ("general", "hallucination_safety", "firsthand_usage", -1, True, None, "admitted it may have hallucinated a detail"),
    143: ("general", "satisfaction", "firsthand_usage", 1, True, "claude-opus-4.7", "fallback when Sonnet fails"),
    144: ("general", "satisfaction", "firsthand_usage", 1, True, None, "default model, occasional failures"),
}


def main() -> None:
    with open(QUEUE, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert len(rows) == 145, len(rows)
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
