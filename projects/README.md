# Projects

Hands-on builds that reinforce the module content. Recommended order:

1. **Prompt-powered Research Assistant** (`prompt-research/`)
   - Summarize PDFs, extract key quotes, and answer free-form questions.
   - Focus: prompt iteration + evaluation metrics.

2. **Mini Helpdesk Chatbot** (`mini-helpdesk/`)
   - Few-shot + retrieval over FAQs, with structured logging.
   - Focus: grounding + guardrails.

3. **LoRA Fine-tuned Stylist** (`lora-stylist/`)
   - Take user notes and rewrite them in a consistent voice.
   - Focus: data prep, training scripts, checkpoints.

4. **RAG Chat Service** (`rag-chat/`)
   - Production-style FastAPI service with health checks, tracing, and containerization.
   - Focus: deployment + monitoring.

Each project directory includes:
- `README.md` with problem statement, architecture diagram, and stretch goals
- `tasks.md` with incremental milestones
- `checklist.md` for QA before sharing

> ✏️ Tip: Treat each project as a mini case study. Document assumptions, evaluation results, and future improvements in the `README`.
