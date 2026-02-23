# Data Directory

This directory contains datasets for fine-tuning and evaluation.

## Files

### devops_qa.jsonl
DevOps Q&A pairs for instruction tuning.

**Format:**
```json
{"instruction": "What is CI/CD?", "input": "", "output": "CI/CD stands for..."}
```

**Stats:**
- Entries: 8 (expand to 100+ for real training)
- Split: 80% train, 20% validation

### train.jsonl / val.jsonl
Train/validation splits generated from the main dataset.

## Adding Your Own Data

1. Create a JSONL file with the format:
```json
{"instruction": "...", "input": "...", "output": "..."}
```

2. Use `../notebooks/03-finetuning-prep.ipynb` to preprocess and split

3. Update this README with your dataset details

## Dataset Guidelines

- **Quality over quantity**: 100 high-quality examples > 1000 noisy ones
- **Diverse examples**: Cover different question types and difficulties
- **Consistent format**: Follow the instruction/input/output schema
- **No PII**: Remove personal information from training data
- **Version your data**: Keep track of dataset versions

## Sources

Curated from:
- Official documentation (Kubernetes, Docker, etc.)
- Common DevOps interview questions
- Best practices guides
