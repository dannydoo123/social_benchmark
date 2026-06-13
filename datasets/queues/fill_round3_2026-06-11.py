"""Apply LLM-reviewer decisions to the round-3 labeling queue.

Decisions are keyed by row index in the exported CSV (same order as the
review dump). Excluded rows carry a reason; labeled rows carry
(task, aspect, evidence, polarity, firsthand, model_override, note).
"""
import csv

QUEUE = "datasets/queues/targeted_training_review_round3_2026-06-11.csv"
OUT = "datasets/queues/targeted_training_review_round3_2026-06-11_filled.csv"

EXCLUDED = {
    7: "joke/fictional paper title, no model quality claim",
    10: "span praises a different model than the extracted mention",
    19: "documentation link issue, not model quality",
    22: "misattributed baseline mention; criticism targets Gemini 3.0 A/B",
    30: "hardware/serving speculation, no model quality claim",
    39: "span praises a different model; Opus 4.5 is only the review target",
    49: "duplicate of same-span observation labeled under gpt-5.4",
    52: "tooling history dispute, no model quality claim",
    53: "product prompt discussion, not model quality",
    60: "wishful speculation, no quality claim",
    62: "llama.cpp runtime mention, model not identified",
    71: "technique comparison (RAG vs fine-tuning), not model quality",
    83: "study about human learning effects, not model quality",
    85: "speculation about GPT-series, misattributed to deepseek",
    90: "speculation about unreleased model",
    99: "pricing/architecture speculation about gpt-5, misattributed to gpt-4",
    101: "tooling ecosystem critique, no model quality claim",
    104: "question only, no claim",
    105: "paper methodology mention, no model quality claim",
    107: "privacy policy discussion, no quality claim",
    110: "technique proposal, no model quality claim",
    119: "duplicate span; labeled under claude-sonnet row",
    128: "praise for app, not model quality",
    133: "usage load commentary, no quality claim",
    141: "duplicate of richer span in same thread (index 153)",
    142: "runtime tooling comparison, not model quality",
    147: "joke about contractors, not model quality",
    155: "duplicate span; runtime mention without identified model",
    159: "Gemini only mentioned as API target; praised model is different",
    163: "runtime router support, no model quality claim",
    165: "product policy discussion, not model quality",
    173: "setup instructions advice, no quality claim",
    174: "jest, no claim",
    175: "duplicate product policy discussion",
    177: "curiosity only, no claim",
    181: "tool announcement, no model quality claim",
    182: "tool announcement, no model quality claim",
    193: "duplicate of same-span observation labeled under gpt-5.4",
    195: "deployment options list, descriptive only",
    199: "platform pitch, no model quality claim",
    200: "no claim yet, will check out",
    201: "infrastructure idea, no model quality claim",
    202: "platform pitch, no model quality claim",
    211: "leaked tool list, no quality claim",
    212: "source credibility meta-discussion, no model claim",
    218: "personal anecdote, no model quality claim",
    219: "duplicate personal anecdote, no model quality claim",
}

# index: (task, aspect, evidence, polarity, firsthand, model_override, note)
LABELS = {
    0: ("general", "trust_reliability", "hearsay", 0, False, None, "ambiguous attribution of error"),
    1: ("general", "regression_stability", "hearsay", -1, False, None, "diagnosis speculation"),
    2: ("coding", "value", "firsthand_usage", -1, True, None, "plan limits hit during trivial work"),
    3: ("general", "satisfaction", "firsthand_usage", -1, True, None, ""),
    4: ("research", "satisfaction", "firsthand_usage", -2, True, "gemini-2.0-flash", "failed miserably on thesis task"),
    5: ("general", "satisfaction", "benchmark_anecdote", 1, False, "gpt-5.1", "ARC-AGI-2 gains reaction"),
    6: ("multimodal", "hallucination_safety", "firsthand_usage", -2, True, None, "image task confabulation"),
    8: ("coding", "value", "pricing_value_comment", -1, True, None, "first impression: expensive"),
    9: ("coding", "task_fit", "firsthand_usage", 1, True, "claude-opus-4", "patch review catches errors"),
    11: ("multimodal", "value", "pricing_value_comment", 1, True, None, "transcription cost vs time saved"),
    12: ("multimodal", "satisfaction", "firsthand_usage", -1, True, None, "voice mode poor on iOS"),
    13: ("long_context", "satisfaction", "comparative_evaluation", -1, True, None, "forgets rules more than newer model"),
    14: ("general", "satisfaction", "hearsay", 1, False, None, "meta progress commentary"),
    15: ("general", "task_fit", "comparative_evaluation", 1, True, None, ""),
    16: ("general", "satisfaction", "hearsay", 1, False, None, "meta progress commentary"),
    17: ("agents", "developer_ergonomics", "comparative_evaluation", 1, True, None, "agent SDK superior"),
    18: ("coding", "task_fit", "firsthand_usage", 1, True, None, "mixed: praises code archeology, warns co-hallucination"),
    20: ("general", "hallucination_safety", "benchmark_anecdote", 0, False, None, "mixed benchmark commentary"),
    21: ("general", "trust_reliability", "comparative_evaluation", 0, True, None, "mixed: amazing memory but amnesia; trusts ChatGPT more"),
    23: ("general", "trust_reliability", "benchmark_anecdote", -2, False, None, "iterations veer wildly off-track"),
    24: ("general", "satisfaction", "benchmark_anecdote", 1, False, None, ""),
    25: ("general", "trust_reliability", "hearsay", -1, False, None, "confidently wrong impression"),
    26: ("coding", "satisfaction", "firsthand_usage", 1, True, "qwen-2.5-coder", "fast local completions"),
    27: ("general", "satisfaction", "comparative_evaluation", 1, True, None, "prefers it over Gemini despite benchmarks"),
    28: ("general", "trust_reliability", "benchmark_anecdote", 1, True, None, "reliable across 144 runs"),
    29: ("coding", "satisfaction", "benchmark_anecdote", 1, False, None, "pelican SVG buzz"),
    31: ("coding", "task_fit", "comparative_evaluation", -1, True, None, "failed to find bug newer model found"),
    32: ("coding", "developer_ergonomics", "firsthand_usage", 1, True, "gpt-oss-120b", "good locally but needed patches/prompting"),
    33: ("general", "hallucination_safety", "firsthand_usage", -1, True, None, "fantastical plausible explanations"),
    34: ("coding", "satisfaction", "firsthand_usage", 0, True, None, "mixed: fast and smart but lower-quality code"),
    35: ("agents", "trust_reliability", "firsthand_usage", -2, True, None, "agent unfairly shifted blame"),
    36: ("multimodal", "task_fit", "firsthand_usage", 1, True, "nano-banana-pro", "unusual anatomy editing"),
    37: ("general", "satisfaction", "firsthand_usage", 0, True, None, "adoption statement"),
    38: ("coding", "refusal_acceptance", "firsthand_usage", -2, True, None, "repeated malware suspicion refusals"),
    40: ("long_context", "satisfaction", "release_update_reaction", 0, False, "gpt-5.4", "skeptical of 1M context gains"),
    41: ("writing", "regression_stability", "bug_regression_report", -2, True, "gemini-3", "smaller output limit broke use case"),
    42: ("coding", "trust_reliability", "firsthand_usage", -2, True, None, "sycophantic echo, built thing broken"),
    43: ("agents", "satisfaction", "release_update_reaction", 1, False, None, "anticipation, no access yet"),
    44: ("writing", "refusal_acceptance", "firsthand_usage", -2, True, None, "refuses translations"),
    45: ("general", "regression_stability", "bug_regression_report", -1, True, None, "expletive-rate analysis of own logs"),
    46: ("general", "satisfaction", "benchmark_anecdote", 1, False, None, "LMArena standings"),
    47: ("coding", "task_fit", "comparative_evaluation", 1, True, None, "more creative designs"),
    48: ("coding", "task_fit", "comparative_evaluation", -1, True, None, "less creative than gpt-4.1 for design"),
    50: ("coding", "regression_stability", "benchmark_anecdote", -1, False, None, "Terminal Bench regression"),
    51: ("general", "value", "pricing_value_comment", 0, True, None, "pricing listing"),
    54: ("general", "trust_reliability", "comparative_evaluation", 1, True, None, "self-correcting vs stubborn rivals"),
    55: ("api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True, None, "flakey preview API"),
    56: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "really smart one-shot coding"),
    57: ("general", "hallucination_safety", "release_update_reaction", -1, False, None, "implied hallucination baseline"),
    58: ("general", "hallucination_safety", "release_update_reaction", -1, False, None, "implied hallucination baseline"),
    59: ("general", "satisfaction", "firsthand_usage", 1, True, None, "personal favourite via API"),
    61: ("general", "satisfaction", "firsthand_usage", -2, True, None, "calls release a failure"),
    63: ("general", "trust_reliability", "firsthand_usage", -1, True, "claude-sonnet-3.5", "excessively follows original instructions"),
    64: ("general", "satisfaction", "comparative_evaluation", 1, True, "gemini-2.0-pro", "most capable in their trial"),
    65: ("general", "task_fit", "firsthand_usage", 1, True, None, "preferred for pre-coding design"),
    66: ("coding", "refusal_acceptance", "firsthand_usage", -2, True, None, "near-duplicate sentiment of malware-refusal complaint"),
    67: ("coding", "refusal_acceptance", "firsthand_usage", -2, True, None, "refused trivial timing question"),
    68: ("general", "satisfaction", "firsthand_usage", 2, True, "gemini-3-flash", "hits so many sweet-spots"),
    69: ("general", "satisfaction", "comparative_evaluation", 1, True, "gemini-3.1-pro", ""),
    70: ("multimodal", "task_fit", "firsthand_usage", 1, True, "gemini-3.1-pro", "SVG artwork generation"),
    72: ("general", "satisfaction", "firsthand_usage", 0, True, "llama-3.2", "setup listing, neutral"),
    73: ("general", "regression_stability", "firsthand_usage", 0, True, None, "author sees no change; coworkers complain"),
    74: ("general", "satisfaction", "benchmark_anecdote", 1, True, "deepseek-r1", "NYT Connections standing"),
    75: ("coding", "value", "pricing_value_comment", 1, True, None, "cheapest viable model"),
    76: ("general", "value", "pricing_value_comment", 1, False, "gpt-5.4", "much cheaper than Opus"),
    77: ("coding", "value", "pricing_value_comment", -1, True, None, "subscription limits halt work"),
    78: ("coding", "task_fit", "comparative_evaluation", 1, True, None, "more creative designs"),
    79: ("coding", "trust_reliability", "comparative_evaluation", -1, False, None, "5.2 Codex more reliable"),
    80: ("general", "value", "comparative_evaluation", -1, True, None, "flash outperforms at fraction of price"),
    81: ("multimodal", "satisfaction", "comparative_evaluation", 1, True, None, "transcribed decently vs failing modes"),
    82: ("general", "satisfaction", "comparative_evaluation", -1, False, None, "poorer than frontier models"),
    84: ("general", "satisfaction", "release_update_reaction", 0, False, None, "incremental vs expectations"),
    86: ("coding", "satisfaction", "firsthand_usage", 1, True, "gemini-3.1-pro", "pricey but nailed tasks"),
    87: ("general", "task_fit", "hearsay", 1, False, None, "good at correcting pointed-out mistakes"),
    88: ("coding", "value", "comparative_evaluation", 1, True, None, "pretty good for the price"),
    89: ("general", "value", "benchmark_anecdote", 1, False, None, "open-weight near flagship at lower cost"),
    91: ("general", "satisfaction", "hearsay", 0, False, None, "customer adoption mention"),
    92: ("general", "satisfaction", "hearsay", 0, False, None, "customer adoption mention"),
    93: ("coding", "task_fit", "comparative_evaluation", 1, True, None, "best at coding-heavy tasks"),
    94: ("general", "value", "pricing_value_comment", 0, False, "gemini-3.1-flash-lite", "pricing listing"),
    95: ("general", "satisfaction", "firsthand_usage", 1, True, None, "go-to problem solving partner"),
    96: ("general", "value", "firsthand_usage", 2, True, "deepseek-v4", "incredible value in every aspect"),
    97: ("multimodal", "hallucination_safety", "firsthand_usage", -2, True, None, "confabulated illusion explanation"),
    98: ("general", "satisfaction", "firsthand_usage", 1, True, "gemini-3.0-pro", "excited despite rate limits"),
    100: ("general", "satisfaction", "comparative_evaluation", 1, True, None, "Claude Opus is great"),
    102: ("general", "value", "pricing_value_comment", -2, True, "gpt-5", "terrible value vs prior limits"),
    103: ("general", "satisfaction", "comparative_evaluation", 1, False, "deepseek-v4-flash", "good for size, below flagships"),
    106: ("general", "satisfaction", "comparative_evaluation", -1, False, None, "new model beats it in every discipline"),
    108: ("general", "value", "benchmark_anecdote", 2, False, "gemini-3-flash", "insane value if benchmarks hold"),
    109: ("coding", "trust_reliability", "comparative_evaluation", 0, True, "gemini-2.5-pro", "mixed: fast, large context, but loops"),
    111: ("general", "value", "pricing_value_comment", 0, False, None, "pricing baseline mention"),
    112: ("coding", "satisfaction", "firsthand_usage", 1, True, "qwen3-coder", "local Claude-Code-like setup"),
    113: ("general", "value", "pricing_value_comment", 0, False, None, "pricing history listing"),
    114: ("agents", "task_fit", "firsthand_usage", 1, True, None, "CLIs and skills workflow"),
    115: ("multimodal", "satisfaction", "firsthand_usage", -1, True, None, "visual recognition disappointing"),
    116: ("coding", "trust_reliability", "firsthand_usage", -1, True, None, "expects occasional bad failures"),
    117: ("general", "satisfaction", "comparative_evaluation", 1, True, None, "still feels stronger"),
    118: ("general", "trust_reliability", "integration_failure", -2, True, None, "outages and churn, switching away"),
    120: ("writing", "satisfaction", "hearsay", 1, False, None, "witty generated title"),
    121: ("general", "trust_reliability", "firsthand_usage", -2, True, None, "falls apart off-script"),
    122: ("multimodal", "satisfaction", "firsthand_usage", -1, True, "gemini-3", "leg counting failure"),
    123: ("multimodal", "satisfaction", "firsthand_usage", -1, True, None, "finger counting failure"),
    124: ("multimodal", "task_fit", "firsthand_usage", 1, True, None, "five-legged dog first try"),
    125: ("general", "satisfaction", "hearsay", 2, False, "gemini-3", "net-new creative designs"),
    126: ("general", "hallucination_safety", "firsthand_usage", -1, True, None, "one right, one nonsense answer"),
    127: ("multimodal", "satisfaction", "firsthand_usage", 1, True, None, "impressive finger counting"),
    129: ("general", "developer_ergonomics", "firsthand_usage", -1, True, None, "compaction slow"),
    130: ("general", "regression_stability", "bug_regression_report", -2, True, None, "authored degradation report"),
    131: ("agents", "task_fit", "firsthand_usage", 1, True, None, "local computer tasks"),
    132: ("agents", "trust_reliability", "firsthand_usage", -1, True, None, "fears agent damaging documents"),
    134: ("coding", "task_fit", "firsthand_usage", 1, True, "claude-opus-4.5", "fully working game"),
    135: ("multimodal", "task_fit", "comparative_evaluation", 2, True, None, "phenomenal OCR vs ChatGPT"),
    136: ("agents", "task_fit", "hearsay", -2, False, None, "comically behind agentically"),
    137: ("general", "satisfaction", "firsthand_usage", -2, True, None, "console-embedded version terrible"),
    138: ("coding", "task_fit", "comparative_evaluation", 0, True, "gemini-3-pro", "succeeded but worse than Opus"),
    139: ("general", "regression_stability", "bug_regression_report", -1, True, None, "thought summaries made it less useful"),
    140: ("research", "satisfaction", "firsthand_usage", 2, True, None, "educated tremendously, saved time"),
    143: ("general", "regression_stability", "bug_regression_report", -1, True, None, "shallow-thinking indicators follow-up"),
    144: ("api_developer_workflow", "developer_ergonomics", "integration_failure", -2, True, None, "retry storm exhausted ports"),
    145: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "project 100% written with it"),
    146: ("general", "refusal_acceptance", "firsthand_usage", -2, True, None, "account banned for a question"),
    148: ("writing", "satisfaction", "hearsay", -1, False, None, "generated work judged mediocre"),
    149: ("multimodal", "developer_ergonomics", "integration_failure", -1, True, None, "app bugs around good OCR"),
    150: ("research", "task_fit", "comparative_evaluation", 2, True, "gemini-3", "handles every use case better"),
    151: ("general", "hallucination_safety", "firsthand_usage", -1, True, None, "fabricated usernames"),
    152: ("research", "trust_reliability", "comparative_evaluation", -2, True, None, "never cites sources, unusable"),
    153: ("general", "task_fit", "comparative_evaluation", -1, True, None, "worse console instructions than ChatGPT"),
    154: ("general", "task_fit", "comparative_evaluation", 1, True, None, "deeper troubleshooting"),
    156: ("general", "satisfaction", "hearsay", 1, False, None, "meta praise of capability"),
    157: ("data_analysis", "satisfaction", "firsthand_usage", 2, True, None, "charts without a single mistake"),
    158: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "likes coding agent functionality"),
    160: ("general", "satisfaction", "firsthand_usage", -1, True, None, "evals come up lacking"),
    161: ("general", "trust_reliability", "firsthand_usage", -1, True, "gemini-3.1", "only obeys rules when reminded"),
    162: ("writing", "task_fit", "comparative_evaluation", 1, True, "gemini-2.5-pro", "summary way better at the time"),
    164: ("coding", "task_fit", "comparative_evaluation", 0, True, None, "found bug but not one-shot"),
    166: ("general", "value", "hearsay", -1, False, None, "users running out of usage"),
    167: ("general", "satisfaction", "hearsay", 0, False, None, "leaderboard musing"),
    168: ("long_context", "trust_reliability", "comparative_evaluation", 1, True, None, "does not flush context"),
    169: ("coding", "task_fit", "firsthand_usage", 2, True, "gemini-3", "first LLM to build parser from scratch"),
    170: ("general", "satisfaction", "hearsay", 0, False, None, "leaderboard musing"),
    171: ("research", "task_fit", "firsthand_usage", 1, True, None, "mixed: strong research, agentic nonstarter"),
    172: ("coding", "task_fit", "comparative_evaluation", 1, True, None, "picks up Codex workflow naturally"),
    176: ("general", "satisfaction", "firsthand_usage", 0, True, None, "tested, planning rollout"),
    178: ("general", "satisfaction", "firsthand_usage", -2, True, None, "always the worst by a big margin"),
    179: ("coding", "satisfaction", "firsthand_usage", 1, True, None, "built CRM in stack"),
    180: ("writing", "satisfaction", "firsthand_usage", 1, True, "gemini-2.5-pro", "easy smooth text"),
    183: ("research", "task_fit", "comparative_evaluation", -2, True, None, "wildly inferior at search grounding"),
    184: ("general", "regression_stability", "hearsay", 0, False, None, "skeptical of degradation claims"),
    185: ("coding", "satisfaction", "firsthand_usage", 0, True, "claude-opus-4.5", "outcome not stated in span"),
    186: ("coding", "task_fit", "comparative_evaluation", 1, True, "claude-opus-4.5", "close second in bakeoffs"),
    187: ("api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True, None, "confusing model tier docs"),
    188: ("coding", "satisfaction", "firsthand_usage", 0, True, "gemini-3", "outcome not stated in span"),
    189: ("agents", "task_fit", "comparative_evaluation", -1, True, "gemini-3", "troubles with agentic coding"),
    190: ("general", "satisfaction", "firsthand_usage", -1, True, None, "mixed: best of best but fails complex prompts"),
    191: ("research", "hallucination_safety", "comparative_evaluation", -2, True, None, "few searches then makes things up"),
    192: ("writing", "task_fit", "comparative_evaluation", 1, True, "gemini-2.5-pro", "smoother than GPT5 thinking"),
    194: ("coding", "satisfaction", "benchmark_anecdote", 0, False, "gpt-5.4", "modest SWE-Bench gain analysis"),
    196: ("agents", "value", "hearsay", 0, False, None, "big models not needed for focused agents"),
    197: ("general", "developer_ergonomics", "firsthand_usage", -1, True, None, "hopes current issues get fixed"),
    198: ("api_developer_workflow", "developer_ergonomics", "integration_failure", -1, True, "gpt-5.3", "model unavailable across access paths"),
    203: ("general", "satisfaction", "comparative_evaluation", 1, False, None, "better in practice than Gemini"),
    204: ("general", "satisfaction", "comparative_evaluation", -2, False, "gemini-3", "much worse in practice, benchmaxing"),
    205: ("general", "satisfaction", "firsthand_usage", 0, True, None, "stack adoption listing"),
    206: ("general", "satisfaction", "firsthand_usage", 0, True, None, "stack adoption listing"),
    207: ("general", "satisfaction", "hearsay", 1, False, None, "more interesting taste professionally"),
    208: ("general", "satisfaction", "firsthand_usage", 1, True, "claude-sonnet-3.7", "impressed from first interaction"),
    209: ("general", "satisfaction", "firsthand_usage", -2, True, None, "more or less useless"),
    210: ("general", "satisfaction", "benchmark_anecdote", -1, True, None, "all failed math puzzle"),
    213: ("agents", "task_fit", "release_update_reaction", 1, False, None, "anticipated desktop cleanup use"),
    214: ("general", "satisfaction", "firsthand_usage", 2, True, None, "wants to share with spouse"),
    215: ("general", "value", "pricing_value_comment", -2, True, None, "cancelled over insane limits"),
    216: ("general", "satisfaction", "firsthand_usage", 1, True, None, "great with this, gets it"),
    217: ("general", "regression_stability", "bug_regression_report", -2, True, None, "error-prone today vs previously"),
}


def main() -> None:
    with open(QUEUE, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert len(rows) == 220, len(rows)
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
