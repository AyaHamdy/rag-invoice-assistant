# RAG Invoice Assistant

A hybrid RAG (Retrieval-Augmented Generation) system for answering questions over invoice
documents, combining structured data lookup with semantic search, orchestrated through a
LangGraph agent with a self-verification step. Runs entirely locally using Ollama.

## What it does

Ask natural-language questions about a set of invoices and get grounded, correct answers —
whether the question requires comparing numbers across documents or understanding the
content of a specific invoice.

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Which invoice has the highest total?"}'
```

```json
{
  "query": "Which invoice has the highest total?",
  "route": "structured",
  "answer": "Adrian Hane — invoice_Adrian Hane_33399.pdf — $860.38",
  "confident": true
}
```

## Why hybrid retrieval

Semantic (vector) search finds chunks whose *meaning* is similar to a question — it cannot
compare or aggregate values across documents. Early in this project, asking "which invoice
has the highest total?" returned invoices that merely *mentioned* totals, not the one with
the actual highest value, because similarity search has no concept of numeric comparison.

The fix: extract structured fields (invoice number, customer, date, total) from every
document up front, and route each incoming question to whichever retrieval method actually
answers it:

- **Structured questions** ("highest total", "invoices from customer X") → answered with
  a direct lookup over the extracted structured data.
- **Semantic questions** ("what did customer X order?") → answered with embedding-based
  retrieval over the raw invoice text.

## Architecture

```
                 ┌─────────┐
   query ──────▶ │  route  │
                 └────┬────┘
           structured │ semantic
              ┌────────┴────────┐
              ▼                 ▼
      ┌──────────────┐  ┌──────────────┐
      │ structured_   │  │  semantic_   │
      │ node          │  │  node        │
      │ (data lookup) │  │ (LLM + RAG)  │
      └──────┬───────┘  └──────┬───────┘
             └─────────┬────────┘
                        ▼
                  ┌───────────┐
                  │   check   │  confident? ──▶ END
                  └─────┬─────┘
                         │ not confident (max 1 retry)
                         ▼
                   back to semantic_node
```

Built with [LangGraph](https://github.com/langchain-ai/langgraph): a `StateGraph` with
conditional routing and a confidence-check retry loop, so answers that fail a basic
sanity check get one retry attempt before returning to the user.

## A real bug this project surfaced

`route_node` classifies each question via an LLM call, expected to return exactly
`"structured"` or `"semantic"`. In practice, the model sometimes appended trailing
punctuation (`"structured."`). The routing decision used a substring check (`"structured"
in route`) and worked fine — but `check_node` used exact equality (`route == "structured"`),
which silently failed on the punctuated string. This caused correct, deterministic
structured answers to fall through into the LLM-judged confidence check, get marked "not
confident," and get overwritten by a semantic retry — which then hallucinated invoice
totals from raw context instead of using the already-correct structured lookup.

**Fix:** normalize the LLM's routing output once, immediately after the call, so every
downstream comparison operates on a guaranteed-clean value instead of re-deriving it in
multiple places.

This is the kind of bug that only shows up when two pieces of code that *should* agree on
a value use different comparison logic — worth normalizing untrusted LLM output at the
source rather than defensively re-checking it everywhere it's used.

## Project structure

```
rag-invoice-assistant/
├── app.py                       # Flask entry point
├── requirements.txt
├── data/
│   ├── Sample-Pdf-invoices/     # raw invoice PDFs
│   └── invoices_structured.json # extracted structured data
├── rag/
│   ├── nodes.py                 # RAGState + node functions (route, structured, semantic, check)
│   └── graph.py                 # LangGraph wiring
├── api/
│   └── invoiceAssistant_apis.py                # Flask routes (Blueprint)
└── scripts/
    ├── extract_structured.py    # offline pipeline: PDFs -> invoices_structured.json
    └── eval.py               # runs a small labeled eval set against the live graph
```

## Extraction pipeline

Each PDF is processed through:

1. **Text extraction** (`pypdf`) — handles most content, but reading order isn't always
   preserved (labels and values can end up separated), and table structures in particular
   don't extract cleanly.
2. **LLM structured extraction** — a schema-constrained prompt (JSON mode, temperature 0)
   pulls out `invoice_number`, `customer`, `date`, `ship_mode`, and `total`.
3. **Deterministic fallback for `total`** — a regex match against the `Total:` label (or
   the last dollar amount in the document) catches the cases where the LLM missed it due
   to jumbled text order. High-value numeric fields get a rule-based safety net rather
   than relying solely on model output.

## Evaluation

A small labeled eval set (`scripts/run_eval.py`) runs a handful of questions with known
correct answers against the live graph and reports pass/fail — this is what caught the
routing bug described above being fixed correctly, rather than relying on manual spot
checks.

## Running it locally

Requires [Ollama](https://ollama.com) running locally.

```bash
# pull the models used
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# set up the environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (re)generate structured data from the PDFs in data/Sample-Pdf-invoices/
python scripts/extract_structured.py

# run the eval set
python scripts/run_eval.py

# start the API
python app.py
```

Then:
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What items did Aaron Bergman order?"}'
```