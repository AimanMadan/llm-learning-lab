# LLM Learning Lab

A self-paced, hands-on introduction to modern AI and large language models for aspiring engineers.

## Who is this for?
- Learners who already know basic Python (functions, classes, virtual environments)
- Anyone curious about the practical side of LLMs: data, prompting, evaluation, and deployment
- Mentors who want a plug-and-play curriculum for onboarding new teammates

## How to use this repo
1. **Follow the modules** in `/modules` sequentially. Each module mixes short readings, coding tasks, and reflection prompts.
2. **Build the projects** in `/projects`—they synthesize the module content into something tangible.
3. **Track progress** in `/roadmap.md` by checking off milestones and adding notes.
4. **Explore further** with `/resources.md` when you're ready to dive deeper.

> ⚡️ Recommendation: create a dedicated virtual environment (e.g., `uv`, `pipenv`, or `conda`) and install the packages listed in each module before starting the coding tasks.

## Quick start
```bash
# Clone the repo
git clone https://github.com/AimanMadan/llm-learning-lab.git
cd llm-learning-lab

# Set up a clean environment
uv venv
source .venv/bin/activate
uv pip install -r modules/requirements.txt  # created per-module

# Begin Module 1
code modules/module-01-foundations.md
```

## Repo structure
```
llm-learning-lab/
├── README.md                # Overview + usage guide
├── modules/                 # Step-by-step lessons
├── projects/                # Capstone-style builds
├── notebooks/               # Optional Jupyter explorations
├── resources.md             # Curated links, podcasts, books
└── roadmap.md               # Suggested 6-week plan + progress tracker
```

Happy building, and feel free to fork + iterate once you're comfortable.
