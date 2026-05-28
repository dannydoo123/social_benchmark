# Social Benchmark

Social Benchmark is a human-perceived LLM benchmark and decision-support product. It evaluates large language models from real user evidence in public technical communities, then turns that evidence into multidimensional scores with provenance, confidence intervals, and personalized recommendations.

## Purpose

Traditional benchmarks often fail to answer the practical question: which model is best for a real workflow right now? This project focuses on user-reported model quality, regressions, trust, task suitability, hallucination complaints, refusal complaints, and value-for-cost.

## MVP Data Sources

- Reddit: AI and model-specific communities
- GitHub: SDK repos, agent repos, issue discussions, release regressions
- Hacker News: Ask HN, launch posts, model comparison discussions
- Hugging Face: model and dataset discussions
- Stack Exchange: LLM/API/developer workflow questions

## Planned Tech Stack

- Frontend: React, Next.js
- Backend: Python, FastAPI
- Database: PostgreSQL via Supabase
- Data systems: official API collectors, embedding pipeline, vector search, spam/manipulation detection, statistical scoring engine

## Core Scoring Dimensions

- Satisfaction
- Trustworthiness
- Task suitability
- Regression stability
- Hallucination complaint rate
- Refusal/censorship complaint rate
- Value-for-cost

## Product Principles

- Use official APIs and legally clear data access.
- Preserve evidence provenance and source mix.
- Show confidence intervals and effective sample size.
- Resist manipulation through deduplication, sampling caps, and cluster detection.
- Monetize decision support, not score manipulation.

## Current Status

This repository currently contains the product description and planning documents. The next implementation step is to scaffold the application stack and define the initial database schema for observations, sources, models, evidence provenance, score snapshots, and confidence intervals.

## Repository Docs

- `description.md`: original product and benchmark construction report
- `plan.md`: actionable build plan
- `claude.md`: compact project context and working instructions
