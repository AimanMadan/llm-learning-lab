# Module 03 • Data & Fine-Tuning Basics

**Goal:** Understand when/why to fine-tune and run a small LoRA experiment.

## 1. When to fine-tune vs prompt
| Scenario | Prompting | Fine-tune |
|----------|-----------|-----------|
| Style or tone change | ✅ | ⚠️ |
| Domain-specific jargon | ⚠️ | ✅ |
| Structured extraction | ✅ | ⚠️ |
| Closed QA on proprietary data | ⚠️ | ✅ |

## 2. Dataset prep
- Collect 100–300 high-quality examples (JSONL)
- Normalize to `{"instruction": ..., "input": ..., "output": ...}` format
- Split into train/validation (80/20)

## 3. Hands-on: LoRA via `peft`
```bash
uv pip install datasets transformers peft accelerate bitsandbytes
python scripts/train_lora.py \
  --base_model meta-llama/Llama-3-8b-instruct \
  --data data/custom.jsonl \
  --output checkpoints/lora-v1
```

## 4. Evaluation
- Use the prompts from Module 02
- Compare base vs fine-tuned outputs side-by-side
- Record BLEU / ROUGE (if applicable) + qualitative notes

> ✅ **Deliverables:** `data/README.md` describing your dataset + `scripts/train_lora.py` with documented hyperparameters.
