# Prompt Catalog

Reusable prompt templates for common tasks.

## Classification

### Sentiment Analysis
```
Classify the sentiment of this text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: "{input}"

Respond with only the label.
```

### Topic Classification
```
Classify this text into one of these categories: {categories}

Text: "{input}"

Category:
```

## Extraction

### Named Entity Recognition
```
Extract named entities from this text. Return JSON with keys: people, organizations, locations, dates.

Text: "{input}"

JSON:
```

### Key Points Extraction
```
Extract the 3-5 most important points from this text as a bulleted list.

Text: "{input}"

Key points:
```

## Transformation

### Summarization
```
Summarize the following text in 2-3 sentences, capturing the key points.

Text: "{input}"

Summary:
```

### Rewrite Formal
```
Rewrite this text to be more professional and formal while keeping the meaning.

Original: "{input}"

Formal version:
```

### Simplify (ELI5)
```
Explain this concept in simple terms that a 5-year-old could understand. Use analogies.

Topic: "{input}"

Simple explanation:
```

## Code

### Code Review
```
Review this code for issues. Check for: bugs, performance, readability, security.

```{language}
{code}
```

Issues found:
```

### Debug Helper
```
This code has an error. Analyze it and suggest fixes.

Error: {error}
Code:
```{language}
{code}
```

Analysis:
```

## Creative

### Brainstorm
```
Brainstorm 5 creative ideas for: {topic}

Consider: feasibility, novelty, impact.

Ideas:
```

### Story Generator
```
Write a short story (200 words) with this premise: {premise}

Style: {style}
```

## Usage Tips

1. **Be specific**: More context = better results
2. **Show examples**: Few-shot prompting improves accuracy
3. **Set constraints**: Word limits, formats, tones
4. **Iterate**: Refine prompts based on outputs
5. **Test systematically**: Use the evaluation notebook to benchmark

## Template Variables

Use `{variable}` syntax for dynamic content:
- `{input}` - Main input text
- `{context}` - Additional context
- `{format}` - Output format instructions
- `{style}` - Writing style
- `{language}` - Programming language

---

Add your own templates as you discover what works!
