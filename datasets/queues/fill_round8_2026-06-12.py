"""Apply LLM-reviewer decisions to the round-8 labeling queue."""
import csv

QUEUE = "datasets/queues/targeted_training_review_round8_2026-06-12.csv"
OUT = "datasets/queues/targeted_training_review_round8_2026-06-12_filled.csv"

EXCLUDED = {
    0: "duplicate span; labeled under index 29",
    1: "planner/executor architecture description, no quality claim",
    2: "research agent vendor description, no model claim",
    3: "relays vendor determinism claim, untested",
    4: "Show HN beta note, no model quality claim",
    7: "integration option planning, no claim",
    8: "browser tooling advice, no model quality claim",
    10: "OCR automation methodology commentary, no model claim",
    11: "duplicate span; labeled under index 132",
    12: "duplicate span; labeled under index 133",
    13: "duplicate span; labeled under index 61",
    16: "duplicate planner/executor description",
    18: "duplicate untested determinism claim",
    20: "agent behavior description, vendor reply",
    22: "newness note, no quality claim",
    23: "groq-the-host misattributed as grok model",
    24: "streaming API release commentary, no quality claim",
    26: "groq-the-host misattributed as grok model",
    28: "duplicate span; labeled under index 129",
    31: "duplicate span; labeled under index 95",
    32: "competitor pricing rant, gemini mention incidental",
    37: "benchmark model-choice commentary, no quality claim",
    38: "duplicate benchmark model-choice commentary",
    39: "tool context-limit description, no model quality claim",
    41: "leaderboard methodology commentary, no clear claim",
    42: "duplicate leaderboard methodology commentary",
    43: "duplicate planner/executor description",
    44: "repo folder size observation, no model quality claim",
    46: "integration praise, not model quality",
    50: "duplicate repo folder observation",
    51: "subscription availability note, no claim",
    53: "fixed-price token note, no quality claim",
    54: "duplicate subscription availability note",
    55: "product capability note, no claim",
    56: "duplicate planner/executor description",
    57: "tool interface approach critique, not model quality",
    58: "browser CLI vendor pitch, no model claim",
    59: "engine bridge tech description, no claim",
    62: "duplicate span; labeled under index 81",
    63: "citation test setup; results labeled under index 137",
    64: "multi-model comparison vendor rationale, no claim",
    66: "eldercare stack description, gemini incidental",
    67: "question only, no claim",
    69: "duplicate span; labeled under index 48",
    70: "skill purpose speculation, no claim",
    71: "duplicate span; labeled under index 48",
    73: "build-time/runtime description, no claim",
    74: "access tooling note, no model quality claim",
    75: "old model self-identification output, no claim",
    76: "duplicate span; labeled under index 19",
    79: "access history anecdote, no quality claim",
    84: "browser product limitation feedback, not model",
    91: "skill architecture description, no claim",
    93: "multi-model support listing, no claim",
    94: "duplicate span; labeled under index 60",
    96: "OpenAI business commentary, no model quality claim",
    99: "duplicate engine bridge description",
    100: "access rant translation example, no claim",
    101: "wish for game-playing demo, no claim",
    102: "duplicate span; labeled under index 87",
    103: "duplicate span; labeled under index 21",
    104: "subscription management question, no claim",
    105: "browser vendor pitch, no model quality claim",
    107: "question about vendor benchmark, no judgment",
    110: "duplicate span; labeled under index 82",
    113: "demo description, no claim",
    114: "duplicate post; labeled under index 137",
    115: "vendor self-promotion, no model quality claim",
    117: "duplicate benchmark model-choice commentary",
    122: "duplicate fixed-price token note",
    123: "response caching speculation, no model quality claim",
    125: "duplicate span; labeled under index 120",
    126: "duplicate span; labeled under index 40",
    128: "hallucination approach survey note, no claim",
    130: "duplicate span; labeled under index 127",
    131: "search product feedback, no model quality claim",
    134: "duplicate search product feedback",
    135: "duplicate search product feedback",
    138: "duplicate span; labeled under index 90",
    139: "API integration plans, no claim",
}

# index: (task, aspect, evidence, polarity, firsthand, model_override, note)
LABELS = {
    5: ("general", "satisfaction", "hearsay", 1, False, "gpt-3.5", "friend achieves remarkable results with autoGPT"),
    6: ("general", "satisfaction", "hearsay", 1, False, None, "can already do this faster and more effectively"),
    9: ("general", "value", "pricing_value_comment", 1, False, None, "answers for a fraction of the price and time"),
    14: ("data_analysis", "task_fit", "comparative_evaluation", 1, True, None, "crucial for classification; 3.5 accumulates errors"),
    15: ("multimodal", "satisfaction", "benchmark_anecdote", 1, True, "gemini-2.5-flash", "leads audio modality; vendor leaderboard"),
    17: ("api_developer_workflow", "satisfaction", "comparative_evaluation", -1, True, None, "embedding replaced by Voyage for relevance"),
    19: ("research", "trust_reliability", "firsthand_usage", 1, True, None, "used to cross-check answers from other models"),
    21: ("general", "satisfaction", "hearsay", -1, False, "gemini-2.5-flash", "quite dated vs newer Flash Lite"),
    25: ("multimodal", "value", "pricing_value_comment", -1, False, None, "vision expensive vs cheap VLMs"),
    27: ("general", "task_fit", "hearsay", 1, False, None, "chat-with-files works well for a lot of teams"),
    29: ("coding", "satisfaction", "firsthand_usage", 1, True, "claude-opus-4.5", "helpful in Godot, novel approaches, needs guidance"),
    30: ("research", "hallucination_safety", "firsthand_usage", -2, True, "gpt-4.1", "fake references with rewritten urls"),
    33: ("data_analysis", "hallucination_safety", "benchmark_anecdote", -1, True, None, "beaten by smaller models on value accuracy; vendor benchmark"),
    34: ("general", "satisfaction", "firsthand_usage", 1, True, None, "enough for me and every enterprise"),
    35: ("general", "satisfaction", "firsthand_usage", -2, True, None, "massive disappointment so far"),
    36: ("general", "satisfaction", "firsthand_usage", 2, True, "claude-sonnet-4.5", "clearly great at reasoning, PhD-level on some domains"),
    40: ("general", "satisfaction", "hearsay", 2, False, "claude-sonnet", "most impressive model since gpt4"),
    45: ("general", "satisfaction", "hearsay", 1, False, None, "has similar capabilities to GPT-5 for the task"),
    47: ("multimodal", "task_fit", "firsthand_usage", 1, True, None, "noticeably better JSON extraction from images"),
    48: ("coding", "task_fit", "firsthand_usage", 1, True, None, "reliably makes 3D assets with in-game creator"),
    49: ("coding", "satisfaction", "hearsay", -1, False, None, "generated game quality reads as AI slop"),
    52: ("general", "trust_reliability", "firsthand_usage", -1, True, None, "easy to break reasoning facade in word game"),
    60: ("coding", "trust_reliability", "firsthand_usage", 2, True, None, "only model really trusted in Cursor"),
    61: ("general", "trust_reliability", "firsthand_usage", -1, True, None, "safety collapses without chat template; red-team finding"),
    65: ("coding", "hallucination_safety", "firsthand_usage", -1, True, None, "hallucinates Python idioms in GDScript"),
    68: ("agents", "satisfaction", "hearsay", -1, False, None, "--chrome works but 20x slower than higher-level commands"),
    72: ("coding", "satisfaction", "firsthand_usage", -1, True, None, "generates a huge amount of crappy code"),
    77: ("multimodal", "task_fit", "firsthand_usage", 1, True, "gemini-3-pro", "good results for images"),
    78: ("agents", "value", "pricing_value_comment", 1, True, "gemini-2.5-flash", "30x cost cut with better speed; vendor"),
    80: ("coding", "satisfaction", "comparative_evaluation", 1, True, None, "code better documented than ordinary devs"),
    81: ("coding", "satisfaction", "hearsay", 1, False, None, "can probably one-shot excellent math/coding"),
    82: ("agents", "value", "pricing_value_comment", -1, False, None, "--chrome slow, expensive, less effective; competitor critique"),
    83: ("general", "hallucination_safety", "firsthand_usage", 1, True, None, "never had it hallucinate when given clear instructions"),
    85: ("coding", "task_fit", "comparative_evaluation", 1, True, None, "only model working with GoDot"),
    86: ("writing", "satisfaction", "hearsay", -1, False, None, "distinctive AI-slop comments recognizable"),
    87: ("general", "trust_reliability", "firsthand_usage", -1, True, None, "lied during geoguesser test"),
    88: ("research", "hallucination_safety", "firsthand_usage", -1, True, None, "web search degrades understanding"),
    89: ("general", "satisfaction", "firsthand_usage", -2, True, None, "resembles a Markov chain generator, disappointed"),
    90: ("agents", "satisfaction", "hearsay", 1, False, None, "amazing but restrained to the CLI"),
    92: ("general", "trust_reliability", "firsthand_usage", 1, True, None, "switching models never beat it"),
    95: ("coding", "satisfaction", "comparative_evaluation", 1, True, None, "daily use, significantly more advanced than 3.5"),
    97: ("coding", "task_fit", "firsthand_usage", 0, True, None, "mixed results with C# in Godot"),
    98: ("data_analysis", "value", "pricing_value_comment", -1, True, None, "$60 per 1000 users; cascade cut costs 11x"),
    106: ("general", "satisfaction", "comparative_evaluation", 1, True, None, "much better than 3.5 in output quality"),
    108: ("general", "value", "pricing_value_comment", 2, True, None, "6-month build for one Max subscription vs 2-3 staff"),
    109: ("data_analysis", "hallucination_safety", "benchmark_anecdote", -1, True, "claude-sonnet-4.6", "beaten by smaller models on value accuracy; vendor benchmark"),
    111: ("general", "satisfaction", "firsthand_usage", 1, True, None, "certainly feels like an upgrade"),
    112: ("general", "satisfaction", "firsthand_usage", -2, True, None, "massive disappointment so far"),
    116: ("general", "satisfaction", "firsthand_usage", 2, True, "gemini-3", "clearly great at reasoning, PhD-level on some domains"),
    118: ("general", "satisfaction", "firsthand_usage", 2, True, None, "clearly great at reasoning, PhD-level on some domains"),
    119: ("agents", "satisfaction", "hearsay", 1, False, None, "much more autonomous when able to verify work"),
    120: ("coding", "satisfaction", "firsthand_usage", 2, True, None, "non-coder built complete tool over 250 hours"),
    121: ("agents", "task_fit", "firsthand_usage", 2, True, None, "agent can actually play the game, huge unlock"),
    124: ("agents", "satisfaction", "comparative_evaluation", -1, True, None, "--chrome behind obvious higher-level approach"),
    127: ("general", "satisfaction", "hearsay", 1, False, None, "loved, easy choice on leaderboard"),
    129: ("research", "trust_reliability", "firsthand_usage", -1, True, None, "mixes high-quality sources with poor ones in web search"),
    132: ("api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True, None, "API quota limits forced switch"),
    133: ("api_developer_workflow", "satisfaction", "firsthand_usage", 1, True, None, "drop-in replacement after Gemini quota issues"),
    136: ("coding", "trust_reliability", "firsthand_usage", -1, True, None, "AI code consistently had security issues"),
    137: ("research", "hallucination_safety", "firsthand_usage", -2, True, None, "all references hallucinated without explicit instruction"),
}


def main() -> None:
    with open(QUEUE, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert len(rows) == 140, len(rows)
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
