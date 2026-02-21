# Module 02 • Prompt Craft & Evaluation

**Goal:** Learn how to design, test, and iterate on prompts for different tasks.

## 1. Concepts
- Zero-shot vs few-shot vs chain-of-thought prompting
- System / user / assistant roles in chat APIs
- Guardrails: temperature, top-p, frequency & presence penalties
- Why evaluation matters (hallucinations, bias, toxicity)

## 2. Exercises
1. Recreate a simple sentiment classifier using only prompts (no fine-tuning). Measure accuracy on 20 labeled samples.
2. Build a "prompt catalog" file with reusable patterns (summaries, extraction, rewriting).
3. Experiment with structured outputs (JSON) and parse them in Python.

## 3. Metrics to record
- Latency per call
- Token usage (input/output)
- Subjective quality score (1–5) for each prompt variant

## 4. Stretch goal
Implement an automated evaluator using `guardrails-ai` or `lm-eval` to score the outputs for correctness.

> ✅ **Deliverable:** `notebooks/02-prompt-lab.ipynb` capturing experiments, plus a `prompts/catalog.md` with at least five reusable templates.
