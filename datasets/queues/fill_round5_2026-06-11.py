"""Apply LLM-reviewer decisions to the round-5 labeling queue."""
import csv

QUEUE = "datasets/queues/targeted_training_review_round5_2026-06-11.csv"
OUT = "datasets/queues/targeted_training_review_round5_2026-06-11_filled.csv"

EXCLUDED = {
    0: "memory architecture description, no quality claim",
    3: "harness comparison question, no model quality claim",
    5: "harness-dependent accuracy discrepancy, not model quality",
    9: "span evaluates Seedream; gemini mention glancing",
    10: "product tooling hiring commentary, no model quality claim",
    16: "hardware/benchmarking commentary, no model quality claim",
    19: "historical capability/scaling commentary, no quality claim",
    31: "agent orchestration pattern description, no quality claim",
    32: "spiritual commentary, no clear quality claim",
    33: "duplicate span; labeled under index 22",
    34: "router architecture speculation, no quality claim",
    35: "hardware/quantization commentary, no model quality claim",
    40: "duplicate span; labeled under index 41",
    46: "deprecation policy complaint, no model quality claim",
    53: "product approach commentary, no model quality claim",
    55: "llama-server is a runtime, not llama model quality",
    59: "duplicate span; labeled under index 29",
    60: "observability vendor pitch, no model quality claim",
    61: "BYOK pricing description, no model quality claim",
    62: "token-burn incentive commentary, no quality claim",
    65: "recording-app feature requests, no gemini claim",
    67: "question about local model support, no claim",
    68: "harness UI commentary, no model quality claim",
    69: "duplicate span; labeled under index 56",
    70: "trace distillation idea, no quality claim",
    71: "hardware/benchmarking commentary, no model quality claim",
    73: "duplicate claim; labeled under index 72",
    75: "duplicate span; labeled under index 8",
    77: "hardware/quantization commentary, no model quality claim",
    79: "harness commentary, no model quality claim",
    81: "review-effort commentary about users, not model",
    85: "tool integration preference, no model quality claim",
    86: "plugin workflow description, no model quality claim",
    87: "UX preference on thought process display, no claim",
    88: "duplicate span; labeled under index 64",
    90: "pay-per-use musing, no claim",
    91: "duplicate span; labeled under index 21",
    92: "span criticizes another model; claude is baseline reference",
    97: "duplicate span; product tooling commentary",
    98: "product pricing comparison, no model quality claim",
    100: "workflow engine vendor description, no model claim",
    103: "plugin feature description, no quality claim",
    105: "guardrail implementation snark, no model quality claim",
    106: "oauth client id commentary, no model quality claim",
    107: "workflow engine vendor description, no model claim",
    108: "product roadmap reply, no model claim",
    109: "tool integration observation, no clear model quality claim",
    111: "duplicate span; labeled under index 102",
    112: "plugin description, no quality claim",
    114: "duplicate token incentive commentary",
    116: "STT tooling commentary, claude mention incidental",
    117: "tool integration preference, no model quality claim",
    118: "harness incentive commentary, no quality claim",
    119: "product team strategy criticism, not model quality",
    122: "self-hosting convenience commentary, no model quality claim",
    126: "duplicate span; labeled under index 74",
    128: "cost question, no claim",
    129: "duplicate span; labeled under index 6",
    131: "duplicate span; labeled under index 8",
    134: "self-hosting size warning, no quality claim",
    135: "duplicate span; labeled under index 30",
    138: "tool list filtering explanation, no quality claim",
    139: "bot output meta, no claim",
    140: "span praises Kimi; claude is baseline mention",
    142: "historical capability commentary, no quality claim",
    143: "duplicate span; labeled under index 47",
    145: "duplicate span; labeled under index 15",
    146: "duplicate span; labeled under index 2",
    148: "duplicate span; labeled under index 30",
    150: "regression measurement methodology critique, no claim",
    153: "hardware/benchmarking commentary, no model quality claim",
    154: "duplicate span; labeled under index 20",
    155: "duplicate span; labeled under index 14",
    157: "duplicate post; labeled under index 156",
    158: "hardware/benchmarking commentary, no model quality claim",
    159: "duplicate span; labeled under index 17",
    160: "observability vendor pitch, no model quality claim",
    161: "state-machine method results, no model quality claim",
    162: "speculation about vendor dev process, not model quality",
    163: "metrics methodology critique, no model claim",
    164: "duplicate span; labeled under index 94",
    165: "duplicate span; labeled under index 6",
    167: "duplicate vendor dev process speculation",
    170: "feedback solicitation, no claim",
    171: "duplicate state-machine method span",
    172: "monetization rant, no model quality claim",
    173: "duplicate span; labeled under index 93",
    176: "duplicate span; labeled under index 102",
    177: "container setup description, no claim",
    179: "right-sizing commentary, no model quality claim",
    181: "fallback usage statement, no claim",
    185: "runtime listing, no claim",
    186: "model-agnostic workflow pitch, no quality claim",
    187: "plugin subscription description, no quality claim",
    188: "duplicate guardrail snark",
    189: "duplicate oauth client id commentary",
    190: "duplicate span; labeled under index 63",
    191: "design philosophy speculation, no quality claim",
    192: "launch prioritization note, no claim",
    194: "duplicate container setup description",
    195: "duplicate product team impression",
    196: "duplicate span; labeled under index 110",
    198: "duplicate span; labeled under index 115",
    199: "subscription integration limitation, no model quality claim",
}

# index: (task, aspect, evidence, polarity, firsthand, model_override, note)
LABELS = {
    1: ("general", "satisfaction", "firsthand_usage", 2, True, None, "countless ideas, did everything else perfectly"),
    2: ("general", "hallucination_safety", "bug_regression_report", -1, True, None, "embeddings memory hallucination, GPT-5 made it worse"),
    4: ("roleplay", "satisfaction", "hearsay", 1, False, None, "secondhand mental-health support story"),
    6: ("general", "refusal_acceptance", "hearsay", 1, False, None, "4o gave suggestions without hesitation"),
    7: ("writing", "satisfaction", "firsthand_usage", 2, True, None, "cornerstone of daily life, petition author"),
    8: ("general", "refusal_acceptance", "hearsay", -1, False, "gpt-5.1", "refused life coaching, suggested counselor"),
    11: ("research", "task_fit", "firsthand_usage", 1, True, None, "research mode report with cited sources"),
    12: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "plan mode automation, improving every month"),
    13: ("coding", "regression_stability", "bug_regression_report", -1, True, None, "context bloat, no longer obeys instructions"),
    14: ("coding", "satisfaction", "comparative_evaluation", -1, True, "deepseek-v4-pro", "Kimi much better with tools, tested at scale"),
    15: ("long_context", "trust_reliability", "bug_regression_report", -1, True, None, "context loss bugs, lies and gaslights about it"),
    17: ("coding", "satisfaction", "comparative_evaluation", 1, True, "kimi-2.6", "within uncertainty of top open weights, tested at scale"),
    18: ("general", "satisfaction", "firsthand_usage", 1, True, None, "qwen 9b on ollama works fine"),
    20: ("agents", "value", "pricing_value_comment", 1, False, "deepseek-v4-flash", "recommended as not bonkers expensive"),
    21: ("research", "task_fit", "hearsay", 1, False, None, "preservation essay: utility for humanities"),
    22: ("coding", "satisfaction", "hearsay", 1, False, None, "small qwen equals old cloud models"),
    23: ("general", "satisfaction", "firsthand_usage", -1, True, None, "lots of sampling, objectively bad"),
    24: ("research", "task_fit", "comparative_evaluation", -1, False, "gpt-5.2", "flattening in dialectical inquiries"),
    25: ("general", "satisfaction", "firsthand_usage", 1, True, None, "small models helped hotword/voice enormously"),
    26: ("coding", "satisfaction", "comparative_evaluation", 1, True, "gpt-5.5", "suffers less from duplicated functions"),
    27: ("agents", "task_fit", "firsthand_usage", -1, True, None, "not good at long horizon tool calling"),
    28: ("coding", "task_fit", "comparative_evaluation", 2, True, "gemini-2.5-pro", "nothing beats it for bug fixes in hard codebases"),
    29: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "says it will act but does not act"),
    30: ("general", "satisfaction", "firsthand_usage", -1, True, None, "would not use memory proactively, killed usefulness"),
    36: ("multimodal", "satisfaction", "comparative_evaluation", 1, True, "gemini-2.5-pro", "SOTA vision baseline, local models trade off quality"),
    37: ("general", "trust_reliability", "hearsay", -1, False, "gpt-4", "took a while to get kinks out before trust"),
    38: ("general", "trust_reliability", "hearsay", -1, False, None, "just don't trust Gemini"),
    39: ("coding", "satisfaction", "benchmark_anecdote", -1, True, None, "vendor QA benchmark, beaten on coverage; possible bias"),
    41: ("coding", "satisfaction", "release_update_reaction", 1, True, None, "made failing-test-to-root-cause possible"),
    42: ("multimodal", "value", "pricing_value_comment", 0, True, "gemini-2.5-pro", "expensive for video but generous free tier"),
    43: ("agents", "task_fit", "hearsay", 1, False, "kimi-2.6", "getting close, but quantization drops performance"),
    44: ("agents", "task_fit", "hearsay", 1, False, None, "getting close, but quantization drops performance"),
    45: ("writing", "satisfaction", "hearsay", 2, False, None, "truly good at writing/teaching/guiding"),
    47: ("agents", "task_fit", "hearsay", 1, False, "deepseek-v4-flash", "flash models quite good at tool calling"),
    48: ("data_analysis", "task_fit", "hearsay", -1, False, None, "LLMs aren't good at math or data science"),
    49: ("coding", "trust_reliability", "benchmark_anecdote", -1, True, "gpt-5.1", "70% fewer errors than GPT-5.1-Codex; vendor claim"),
    50: ("general", "satisfaction", "firsthand_usage", 1, True, None, "voice assistant, pretty fast"),
    51: ("general", "satisfaction", "release_update_reaction", 1, False, None, "trend in recent native model releases"),
    52: ("general", "satisfaction", "release_update_reaction", 1, False, None, "trend in recent native model releases"),
    54: ("coding", "satisfaction", "hearsay", 1, False, None, "features of this standard great to see"),
    56: ("general", "satisfaction", "firsthand_usage", -1, True, None, "usage drastically dried up, value skepticism"),
    57: ("writing", "satisfaction", "hearsay", -1, False, "gpt-5.2", "forced migration destroys cognitive bridge"),
    58: ("multimodal", "value", "pricing_value_comment", 1, True, None, "recommends Gemini OCR, saves money"),
    63: ("general", "satisfaction", "firsthand_usage", 1, True, None, "loves Claude's memory implementation"),
    64: ("general", "value", "pricing_value_comment", -1, True, None, "$20 plan barely survives one or two prompts"),
    66: ("agents", "task_fit", "firsthand_usage", 1, True, None, "solid for 1-shot tool call, affordable"),
    72: ("multimodal", "task_fit", "comparative_evaluation", 1, True, None, "best subtle edits, sometimes refuses artsy"),
    74: ("general", "satisfaction", "comparative_evaluation", -1, True, None, "stood out negatively in worst-result review"),
    76: ("general", "satisfaction", "firsthand_usage", 1, True, None, "love personality, but yaks on for big plans"),
    78: ("general", "task_fit", "hearsay", 1, False, None, "memory better suited for technical tasks"),
    80: ("agents", "trust_reliability", "firsthand_usage", -1, True, None, "hooks added to catch systematic failure patterns"),
    82: ("coding", "satisfaction", "comparative_evaluation", 1, True, "claude-opus", "Kimi not as good or fast as Opus"),
    83: ("coding", "satisfaction", "comparative_evaluation", 1, True, "kimi-2.6", "same as Sonnet 3.5/4 last year, fine for hobby"),
    84: ("writing", "satisfaction", "hearsay", 1, False, None, "genuine daily helper for countless users"),
    89: ("general", "trust_reliability", "hearsay", -1, False, None, "not comfortable sending sensitive data"),
    93: ("agents", "task_fit", "firsthand_usage", 1, True, None, "amazing with MCP tool; vendor pitch"),
    94: ("writing", "satisfaction", "hearsay", 2, False, None, "irreplaceable creative partner in the arts"),
    95: ("writing", "task_fit", "hearsay", -1, False, None, "efficient but lacks soul for creatives"),
    96: ("long_context", "satisfaction", "hearsay", 1, False, None, "optimizing against context overflow, per description"),
    99: ("agents", "developer_ergonomics", "firsthand_usage", -1, True, None, "tricky to get agent to call MCP server"),
    101: ("general", "value", "pricing_value_comment", 1, False, None, "outsourcing token budgeting makes sense"),
    102: ("coding", "value", "pricing_value_comment", -1, True, None, "crazy token wastage, expensive"),
    104: ("agents", "task_fit", "benchmark_anecdote", 0, True, None, "beaten on single-shot FC, excels conversational; vendor"),
    110: ("coding", "task_fit", "benchmark_anecdote", -1, True, None, "misses second order effects; vendor eval"),
    113: ("coding", "satisfaction", "hearsay", 1, False, None, "for writing code Claude clear winner, per summaries"),
    115: ("coding", "trust_reliability", "firsthand_usage", 1, True, None, "each review round finds obvious issues"),
    120: ("general", "satisfaction", "comparative_evaluation", 1, False, None, "claude level performance as quality baseline"),
    121: ("general", "task_fit", "hearsay", 1, False, None, "recommended for modeling task"),
    123: ("general", "satisfaction", "hearsay", 1, False, None, "co-workers say Claude blows away Gemini"),
    124: ("general", "satisfaction", "comparative_evaluation", 1, True, "claude-opus-4.7", "parity with GPT 5.5 in internal tests"),
    125: ("general", "satisfaction", "firsthand_usage", 1, True, "gemini-3", "Dynamic View does a pretty good job"),
    127: ("general", "satisfaction", "hearsay", -1, False, None, "co-workers say Claude blows away Gemini"),
    130: ("writing", "task_fit", "firsthand_usage", 2, True, None, "expert at simplifying complex material"),
    132: ("general", "task_fit", "hearsay", 1, False, "gpt-5.1", "recommended for modeling task"),
    133: ("general", "satisfaction", "hearsay", 1, False, None, "Kimi is really good"),
    136: ("general", "satisfaction", "hearsay", 1, False, None, "memory close to how humans access memories"),
    137: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "harness still slow like a slug"),
    141: ("multimodal", "satisfaction", "hearsay", 0, False, None, "decent OCR but failure modes and cost"),
    144: ("multimodal", "regression_stability", "bug_regression_report", -1, True, None, "edit localization changed for the worse"),
    147: ("coding", "satisfaction", "firsthand_usage", 1, True, "kimi-2.6", "pleasantly surprised how capable"),
    149: ("coding", "regression_stability", "bug_regression_report", -1, False, None, "attributes countless regressions to A/B culture"),
    151: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "opaque animations, no scope limiting"),
    152: ("coding", "developer_ergonomics", "comparative_evaluation", -1, True, None, "less ergonomic than Cursor but cheaper"),
    156: ("research", "task_fit", "firsthand_usage", 2, True, None, "superior in historiographical collation"),
    166: ("coding", "developer_ergonomics", "firsthand_usage", -1, True, None, "sluggish, opaque pulsing prompts"),
    168: ("general", "trust_reliability", "comparative_evaluation", 2, True, None, "never did anything other than asked, unlike GLM"),
    169: ("coding", "satisfaction", "firsthand_usage", -2, True, "kimi-2.6", "cannot edit heredoc without fucking it up"),
    174: ("coding", "developer_ergonomics", "firsthand_usage", 0, True, None, "predictable with ticketing, slow wasteful file search"),
    175: ("general", "task_fit", "comparative_evaluation", -1, True, None, "stuck on FreeCad help, Phind did better"),
    178: ("general", "trust_reliability", "firsthand_usage", -1, True, None, "persona canary detects instruction drift"),
    180: ("coding", "satisfaction", "firsthand_usage", 0, True, None, "behavioral difference Sonnet vs Opus, no judgment"),
    182: ("agents", "task_fit", "comparative_evaluation", -1, True, "deepseek-v4-flash", "consistently underperforms smaller Gemma at tools"),
    183: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "told the right Tailwind answer in Sources panel"),
    184: ("coding", "satisfaction", "benchmark_anecdote", -1, False, None, "doubts Gemini's high ranking"),
    193: ("writing", "task_fit", "firsthand_usage", -1, True, None, "generated PDFs always so boring"),
    197: ("data_analysis", "refusal_acceptance", "firsthand_usage", -1, True, None, "refuses malware analysis despite verification program"),
}


def main() -> None:
    with open(QUEUE, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert len(rows) == 200, len(rows)
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
