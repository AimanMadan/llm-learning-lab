# Module 01 • AI & LLM Foundations

**Goal:** Understand how modern LLMs are trained, what they can/can't do, and set up your tooling.

## 1. Concepts to internalize
- Deep learning ≠ magic: it's linear algebra + gradient descent at scale
- Difference between *base* vs *instruction* vs *chat* models
- Tokenization, context windows, and why prompt length matters
- Inference stacks: CPU vs GPU vs specialized accelerators

## 2. Reading & watching
- [3Blue1Brown • Neural networks](https://youtu.be/aircAruvnKk)
- [Lil'Log • A Friendly Introduction to Transformers](https://lilianweng.github.io/posts/2023-09-01-map)
- Hugging Face course chapters 1–3

## 3. Tooling checklist
| Tool | Why | Status |
|------|-----|--------|
| Python 3.10+ | Base language | ☐ |
| `uv` or `pipenv` | Reproducible envs | ☐ |
| VS Code + Python extension | Dev environment | ☐ |
| `curl` + `jq` | Quick API tests | ☐ |

## 4. Hands-on tasks
1. Create a new virtual env and install `torch`, `transformers`, and `datasets`.
2. Run the `notebooks/01-tokenization.ipynb` notebook to inspect token counts for different sentences.
3. Use the OpenAI (or compatible) REST API via `curl` and save the JSON response.

## 5. Reflection prompts
- In your own words, explain why bigger models aren't always better.
- What are three risks of deploying an LLM into production?
- How would you explain "prompt engineering" to a non-technical friend?

> ✅ **Deliverable:** notes in `notebooks/01-tokenization.ipynb` + answers to reflection prompts in your personal journal or repo fork.
