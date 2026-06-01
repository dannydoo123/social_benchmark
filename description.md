# Human-Perceived LLM Benchmark Construction Report

## Executive Summary

What you are trying to build is not simply “another benchmark replacing existing benchmarks,” but rather something closer to a **decision-making layer** that structures and evaluates the actual quality, trustworthiness, task suitability, and perceived regressions that real users experience in practice.

Existing ability-centered benchmarks have a limitation: users still struggle to determine “which model is best for my actual workflow.” Recent research also suggests that separate evaluation systems reflecting real user scenarios and human preference are necessary.

URS (User-Reported Scenarios) organized six user intents using 1,846 real-world scenarios collected from 712 people across 23 countries and showed strong correlation with human preference. WildBench proposed “real prompt-based evaluation” by extracting 1,024 tasks from over one million real conversation logs. Chatbot Arena built a public human preference leaderboard using over 240,000 human votes. At the same time, recent interdisciplinary reviews point out that benchmark scores themselves eventually become optimization targets, causing divergence from real-world usability.

In other words, your product hypothesis is academically well justified.

The three most important product decisions are:

1. The MVP data source should begin with Hacker News only.
   This keeps the initial pipeline narrow, high-signal, and easier to validate before any future source expansion.

2. The scoring system should not produce only a single “universal score.”
   Instead, it should separate:

   * satisfaction
   * trustworthiness
   * task suitability
   * regression stability
   * hallucination complaint rate
   * refusal/censorship complaint rate
   * value-for-cost

   and allow weights to change depending on user type.

3. The core competitive advantage of this product is not simple crawling, but:

   * sampling design
   * manipulation resistance
   * evidence traceability
   * confidence interval presentation

Without these four components, the system becomes merely a “public opinion ranking.”
With them, it becomes a legitimate “decision-making product.”

Monetization is absolutely possible. However, revenue should come from:

* interpretation layers
* procurement insights
* decision support

—not from score manipulation.

The most realistic initial strategy is:

* free public leaderboard
* paid personalized alerts/comparisons
* enterprise procurement reports

Public scores should remain openly accessible, while:

* regression alerts
* team profiles
* internal PDF reports
* API access
* industry/language/workflow-specific slices

can become paid features.

The most important recommendation can be summarized in one sentence:

**Start only with officially accessible high-signal communities, make scores multidimensional and personalizable, design anti-manipulation around sampling/capping/cluster detection rather than direct score adjustment, and monetize the decision-support layer rather than the grades themselves.**

---

# Monitoring Platforms and Community Priorities

The scores and weekly sample counts below are operational estimates.

The criteria include:

* API accessibility
* legal clarity
* text structure analyzability
* technical expertise concentration
* long-term trend value
* manipulation risk

“Weekly usable samples” refers to posts/comments remaining after:

* duplicate removal
* spam filtering
* author/thread caps

Reddit AI communities contain a very high volume of direct model comparisons and regression discussions.
Communities like:

* r/ChatGPT
* r/ClaudeAI
* r/LocalLLaMA

contain long-term perception trends through megathreads and repeated discussions.

Hacker News, Stack Exchange, and Hugging Face contain more technical discussions:

* regressions
* agent workflows
* SDK failures
* deployment issues
* integration problems

These provide highly reproducible real-world failure evidence.

| Priority | Platform       | Example Communities                                        | Weekly Usable Samples | Signal Quality | API/Legal Accessibility | Manipulation Risk | Technical Expertise | Long-Term Trend Value | Recommendation   |
| -------- | -------------- | ---------------------------------------------------------- | --------------------- | -------------- | ----------------------- | ----------------- | ------------------- | --------------------- | ---------------- |
| A        | Reddit         | r/ChatGPT, r/OpenAI, r/ClaudeAI, r/Anthropic, r/LocalLLaMA | 8k–25k                | Very High      | High                    | Medium            | Medium-High         | Very High             | MVP Core         |
| D        | GitHub         | SDK repos, agent/tool discussions                          | 1k–4k                 | Very High      | Very High               | Low-Medium        | Very High           | Very High             | Deferred         |
| A        | Hacker News    | Ask HN, launch discussions                                 | 300–1.2k              | High           | Very High               | Low               | Very High           | High                  | MVP Core         |
| A        | Hugging Face   | Model/data discussions                                     | 500–2k                | High           | High                    | Medium            | High                | High                  | MVP Core         |
| A        | Stack Exchange | Stack Overflow LLM/API tags                                | 400–1.5k              | High           | High                    | Low               | Very High           | High                  | MVP Core         |
| B        | YouTube        | Review/comparison comments                                 | 2k–10k                | Medium         | High                    | High              | Medium              | Medium                | Phase 2          |
| B        | Naver          | Blogs/cafes/news                                           | 1k–5k                 | Medium         | High                    | Medium            | Medium              | High                  | Korean Expansion |
| B        | Discord        | Partner/public servers                                     | 500–3k                | High           | Medium-Low              | Medium            | High                | Medium                | Partner Only     |
| C        | X/Twitter      | AI power-user discussions                                  | 5k–30k                | Medium         | Medium-Low              | High              | Medium              | High                  | Optional         |

---

# Composite Scoring System

The scoring system should focus on refining **user-perceived quality**, not merely averaging sentiment analysis scores.

The recommended architecture is:

Observation → Aspect Score → Task Score → Personalized Overall Score

The primary dimensions should be:

* Satisfaction
* Trust
* Task Fit
* Regression Stability
* Hallucination Safety
* Refusal Tolerance
* Value

The observation unit is not simply a post/comment, but:

“A specific evidence item i about model m regarding task t and aspect a.”

Each item receives an aspect label:

s(i,a) ∈ {-2, -1, 0, 1, 2}

Where:

* -2 = strong dissatisfaction
* -1 = mild dissatisfaction
* 0 = neutral/mixed
* 1 = satisfied
* 2 = strongly satisfied

Examples:

* “Claude Code became noticeably worse after March”
  → negative coding/regression observation

* “Gemini is best for long document comparison”
  → positive research/satisfaction observation

---

# Weighting Function

The weight of each observation depends primarily on evidence quality.

Recommended formula:

w_i = q_src × q_firsthand × q_author × q_corroboration × q_recency

Where:

* q_src = platform/community quality
* q_firsthand = direct usage evidence
* q_author = author credibility
* q_corroboration = cross-platform agreement
* q_recency = recency decay

Structured technical communities such as:

* Reddit
* GitHub (deferred)
* Hacker News
* Stack Exchange
* Hugging Face

should receive higher source quality scores than general social media.

---

# Aspect Score Formula

Scores are normalized between 0–100:

Score(m,a) = 100 × [ Σ(w_i × ((s_i,a + 2)/4)) ] / Σ(w_i)

Interpretation:

* strongly negative → near 0
* neutral → near 50
* strongly positive → near 100

---

# Task Fit Portfolio

Task categories:

* coding
* writing
* research
* agents
* roleplay

Default weights:

* coding = 0.25
* writing = 0.20
* research = 0.20
* agents = 0.20
* roleplay = 0.15

These should become customizable per user profile.

---

# Regression Detection

Users are extremely sensitive to:
“the model used to be good, but recently became worse.”

Regression should therefore be treated separately from general satisfaction.

Definitions:

p_reg(k) = [ Σ(w_i × 1(reg_i = 1)) ] / Σ(w_i)

RegRisk(m) = 100 × sigmoid(
(p_reg(30) - p_reg(180)) / sqrt(v_hat)
)

Important rule:
Regression scores should only become strong if corroborated across at least two platform families.

Otherwise:

* brigading
* community-specific bias
* emotional overreaction

may distort results.

---

# Hallucination and Refusal Scores

Hallucination complaint rate:

HallRate(m) = 100 × p_hall(90d)

Refusal complaint rate:

RefRate(m) = 100 × p_ref(90d)

Converted into positive scores:

HallSafe(m) = 100 - HallRate(m)

RefAccept(m) = 100 - RefRate(m)

Important distinction:
Unsafe requests should not count negatively.

Only:

* excessive refusals
* over-censorship
* inappropriate policy responses

for legitimate tasks should affect refusal scores.

---

# Value Score

Value combines:

* perceived worth
* actual pricing

Value(m) =
0.7 × PerceivedValue(m)
+
0.3 × PriceNorm(m)

If price collection is difficult initially,
PerceivedValue alone is acceptable for MVP.

---

# Overall Score Formula

Overall(m) =
0.22 × Satisfaction
+
0.18 × Trust
+
0.25 × TaskFitPortfolio
+
0.10 × Value
+
0.10 × (100 - RegRisk)
+
0.10 × HallSafe
+
0.05 × RefAccept

This is only the default global score.
Real recommendations should become personalized per user type.

---

# Confidence Intervals

Confidence intervals must always be displayed.

Recommended:

* thread-week block bootstrap

Display:

* ±95% CI
* effective sample size
* source mix
* firsthand ratio

Effective sample size:

n_eff = (Σw_i)^2 / Σ(w_i^2)

Low confidence warning conditions:

* n_eff < 30
* single platform > 60%
* single community > 35%

---

# Anti-Manipulation Pipeline

Core philosophy:

* official APIs over scraping
* structured sampling over mass collection
* provenance over raw text resale

Recommended architecture:

* collectors
* normalization
* deduplication
* information extraction
* spam/trust scoring
* moderation queues
* aggregation engine
* dashboard/API layer

---

# Deduplication

Three stages:

1. Exact URL/ID deduplication
2. SimHash/MinHash near-duplicate detection
3. Embedding similarity clustering

This reduces:

* reposts
* propaganda campaigns
* spam duplication
* coordinated manipulation

---

# Manipulation Resistance

Trust score example:

TrustItem_i =
0.20A
+
0.15H
+
0.20F
+
0.15C
+
0.10N
+
0.20X

Where:

* A = account age/history
* H = healthy activity ratio
* F = firsthand evidence
* C = cross-platform corroboration
* N = linguistic originality
* X = inverse abnormal burst/cluster score

Operational caps:

* max 1 high-weight observation per author/day/model
* max 3 high-weight observations per thread/model
* community weight caps
* brand-community caps

---

# Personalized Recommendations

The “best model” differs depending on the user.

User profiles include:

* agent builders
* researchers
* writers
* enterprise users
* creative users
* privacy/local users
* budget-sensitive users

Recommendation formula:

U(u,m) =
Σ(alpha_u,t × Fit_m,t)
+
beta_u × Satisfaction
+
gamma_u × Trust
+
delta_u × Value
---------------

## rho_u × RegRisk

## eta_u × HallRate

kappa_u × RefRate

The advantage:
recommendations become explainable.

---

# Dashboard Features

Planned features:

* live leaderboards
* regression alerts
* confidence intervals
* provenance tracking
* evidence visualization
* personalized recommendations
* API access

The dashboard should behave like:
a decision-support console,
not merely a ranking board.

---

# Technology Stack

Frontend:

* React
* Next.js

Backend:

* FastAPI
* Python

Database:

* PostgreSQL
* Supabase

Additional Systems:

* vector search
* embedding pipeline
* spam detection
* statistical scoring engine

---

# Monetization

Recommended strategy:

* free public leaderboard
* paid alerts/comparisons
* enterprise procurement reports
* API subscriptions
* team dashboards

Avoid:

* pay-for-grade
* hidden sponsorship manipulation

Revenue should come from:
decision support,
not score corruption.

---

# Long-Term Vision

The long-term goal is to build a reliable public layer for evaluating LLMs based on:

* human perception
* real-world usage
* long-term behavioral trends

The system should help answer:

* Which model is best for my workflow?
* Which models are becoming unstable?
* Which updates are negatively perceived?
* Which models are trusted by experienced users?
* Which models provide the best value?

Instead of relying solely on benchmark scores,
this system attempts to measure how models are actually experienced by real users over time.
