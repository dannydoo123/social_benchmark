# HN Labeling Iteration Report

Date: 2026-05-31

Scope:
- Source scope was Hacker News only.
- The loop was capped at 5 rounds of 300 rows.
- I stopped after 2 completed rounds because the repeated extractor bugs largely flattened by round 2.

## What Changed

Code changes:
- `src/social_benchmark/pipeline/extractors/rules.py`
- `src/social_benchmark/pipeline/labeling.py`
- `src/social_benchmark/pipeline/cli.py`
- `web/src/App.tsx`
- `web/src/types.ts`
- `tests/test_extractor.py`
- `tests/test_labeling.py`

New behavior:
- Skip title-only Hacker News root stories with no body unless they are `Ask HN`, `Show HN`, or `Tell HN`.
- Strip leading quoted sentences from candidate spans.
- Strip leading and trailing question-only sentences from candidate windows.
- Drop instructional install/prompt spans such as `Install manually:` and `Run npx ...`.
- Stop treating bare `works` as praise and ignore the `bad actors` phrase for polarity.
- Map sandbox/trust/safety language into `trust_reliability`.
- Export a stable `review_id` in the review CSV.
- Export a JSONL sidecar with full raw source context keyed by `review_id`.

## Artifacts

Round 1 artifacts:
- `data/processed/hn_all_combined_observations_v4.jsonl`
- `data/processed/observation_report_hn_all_combined_v4.json`
- `data/processed/label_queue_hn_all_combined_assisted_v3.csv`
- `data/processed/label_queue_hn_all_combined_assisted_v3_context.jsonl`

Round 2 artifacts:
- `data/processed/hn_all_combined_observations_v5.jsonl`
- `data/processed/observation_report_hn_all_combined_v5.json`
- `data/processed/label_queue_hn_all_combined_assisted_v4.csv`
- `data/processed/label_queue_hn_all_combined_assisted_v4_context.jsonl`

Baseline artifacts used for comparison:
- `data/processed/label_queue_hn_all_combined_assisted_v2.csv`
- `data/processed/observation_report_hn_all_combined_v3.json`

## Queue-Level Improvement

Top-300 queue counts:

| Metric | Baseline `assisted_v2` | Round 1 `assisted_v3` | Round 2 `assisted_v4` |
|---|---:|---:|---:|
| Quote-start rows | 17 | 0 | 0 |
| Title-only HN root rows | 1 | 0 | 0 |
| Question-ending rows | 27 | 7 | 0 |
| Instructional install/prompt rows | 2 | 2 | 0 |
| `bad actors` polarity false positives | 1 | 0 | 0 |

Observation report summary:

| Metric | Baseline report `v3` | Round 1 report `v4` | Round 2 report `v5` |
|---|---:|---:|---:|
| Total observations | 538 | 485 | 490 |
| Avg extractor confidence | 0.682 | 0.686 | 0.688 |
| Observations below 0.60 confidence | 160 | 146 | 142 |
| `trust_reliability` observations | 29 | 27 | 45 |

Interpretation:
- Round 1 removed the obvious junk spans from the review queue.
- Round 2 removed most remaining question/instruction residue and improved routing of safety/sandbox comments into `trust_reliability`.
- After round 2, the remaining errors looked more like ambiguous labeling or classifier weakness than broad parsing failures.

## Real CSV Examples

### Example 1: Title-only HN root story removed

Baseline `label_queue_hn_all_combined_assisted_v2.csv`:
- `source_item_id=48321638`
- `evidence_text=Claude Cowork Private Version: What It Is, Cost Breakdown and ROI Guide (2026)`

Round 1 and round 2:
- `source_item_id=48321638` no longer appears in the top-300 review queue.

Reason:
- This was a title-only root story with no body. It should not have been exported as a model-quality observation.

### Example 2: Quoted support Q/A row removed

Baseline `label_queue_hn_all_combined_assisted_v2.csv`:
- `source_item_id=48316127`
- `evidence_text=> is Zed capable of using a Claude Code Subscription? Yes. Zed connects to Claude Code via ACP.`

Round 1 and round 2:
- `source_item_id=48316127` no longer appears in the top-300 review queue.

Reason:
- This was quoted support text, not a usable quality claim.

### Example 3: Install/prompt row removed while substantive row stayed

Round 1 `label_queue_hn_all_combined_assisted_v3.csv` still contained:
- `source_item_id=48309986`
- `evidence_text=Install manually: npm install -g @kaelio/ktx ...`

Round 2 `label_queue_hn_all_combined_assisted_v4.csv`:
- The install/prompt row is gone.
- The substantive failure-example row remains:
- `source_item_id=48309986`
- `evidence_text=Agents are great at generating valid SQL, but it’s not always correct SQL. ...`

Reason:
- The extraction now prefers the actual evidence-bearing paragraph over install instructions and feedback prompts.

### Example 4: Question-heavy integration row cleaned up

Round 1 `label_queue_hn_all_combined_assisted_v3.csv`:
- `source_item_id=48313944`
- `evidence_text=VSCode has an official client? Given IDE usage is being restricted from Claude Code via the CC SDK tokens going to the Claude API rather than your CC Subscription, i'm unclear which IDEs can actually use claude code now. Eg is Zed capable of using a Claude Code Subscription?`

Round 2 `label_queue_hn_all_combined_assisted_v4.csv`:
- `source_item_id=48313944`
- `evidence_text=Given IDE usage is being restricted from Claude Code via the CC SDK tokens going to the Claude API rather than your CC Subscription, i'm unclear which IDEs can actually use claude code now.`

Reason:
- The leading and trailing question-only sentences were trimmed, leaving the actual integration-constraint claim.

### Example 5: Safety/sandbox row routed to `trust_reliability`

Round 1 `label_queue_hn_all_combined_assisted_v3.csv` included:
- `source_item_id=48319718`
- `aspect_category=satisfaction`
- `evidence_text=Claude works in his own separate vm with root access ...`

Round 2 `label_queue_hn_all_combined_assisted_v4.csv` includes:
- `source_item_id=48319718`
- `aspect_category=trust_reliability`
- `evidence_text=Claude works in his own separate vm with root access ...`

Reason:
- The text is about trust, safety, sandboxing, and risk, not generic satisfaction.

### Example 6: Security/threat-model row moved out of generic satisfaction

Round 1 `label_queue_hn_all_combined_assisted_v3.csv` included a weak row:
- `source_item_id=48312763`
- `aspect_category=satisfaction`
- `evidence_text=This is just one security model, there are many others! If a person is running claude in a stronger sandbox, that changes the model considerably. What threat model do you use to evaluate whether an agent's actions are safe?`

Round 2 `label_queue_hn_all_combined_assisted_v4.csv` improved it to:
- `source_item_id=48312763`
- `aspect_category=trust_reliability`
- `evidence_text=Here's the threat model I (a luddite) use to evaluate these. The claude code harness can be mostly trusted, the model cannot be trusted because it is exposed to untrusted data from the internet, and there is no separation of data/code in an llm [0][1]. ...`

Reason:
- The extractor now recognizes safety/sandbox/trust language and preserves the more useful trust-focused span.

## Validation

Validation status:
- Python tests: `33` passed.
- Web build: passed.

## Recommendation

Stop the rule-edit loop here and switch back to review.

Why:
- The obvious span-selection bugs were removed in 2 rounds.
- The queue is materially cleaner.
- The next bottleneck is not broad parsing noise anymore; it is limited reviewed training data and ambiguous labels.

Next step:
- Review `data/processed/label_queue_hn_all_combined_assisted_v4.csv`.
- Use `data/processed/label_queue_hn_all_combined_assisted_v4_context.jsonl` when the span alone is unclear.
- Add another 150-300 reviewed HN rows.
- Retrain and regenerate before doing more extractor surgery.
