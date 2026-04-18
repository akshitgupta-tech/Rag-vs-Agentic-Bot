# RAG vs Agentic AI: Customer Support Bot Benchmark

> A comparative study of two modern AI architectures for automated customer support — a **RAG-based bot** (Retrieval-Augmented Generation with FAISS) and an **Agentic bot** (LLM + tool-calling) — evaluated across 7 quantitative metrics on a 60-query held-out benchmark.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Notebooks](#notebooks)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup](#setup)
- [Results](#results)
- [Conclusion](#conclusion)
- [Project Structure](#project-structure)
- [References](#references)

---

## Project Overview

|  | RAG Bot | Agentic Bot |
|---|---|---|
| **Core idea** | Retrieve similar Q&A pairs from a vector store, then generate a response | Classify intent, select tools, execute Python functions that simulate backend APIs |
| **Vector store** | FAISS + `sentence-transformers/all-MiniLM-L6-v2` | None |
| **LLM** | Groq (`llama-3.1-8b-instant`) | Groq (`llama-3.1-8b-instant`) |
| **Dataset** | Bitext Customer Support (HuggingFace) | Same benchmark CSV from Notebook 1 |
| **Multi-step support** | ❌ No | ✅ Yes (tool chaining) |

---

## Architecture

```
Notebook 1 — RAG Bot
  HuggingFace Bitext Dataset
         │
    80/20 split
    ┌────┴────┐
  Train    Holdout (benchmark)
    │           │
FAISS index  benchmark_queries.csv
    │           │
    └──[query]──► RAG pipeline ──► rag_results.csv

Notebook 2 — Agentic Bot
  benchmark_queries.csv
         │
  Groq LLM (intent classification + response generation)
         │
  Tool selection
  ┌──────┬──────┬────────┬──────────┐
  │      │      │        │          │
check  process cancel  escalate  track
order  refund  order   to human   order
         │
  agentic_results.csv

Notebook 3 — Evaluation
  rag_results.csv + agentic_results.csv
         │
  7-metric evaluation (LLM judge + lexical scorers)
         │
  5 comparison charts + benchmark_summary.csv
```

---

## Notebooks

| # | File | What it does |
|---|---|---|
| 1 | `Notebook_1_RAG_Bot.ipynb` | Loads the Bitext dataset, builds an 80/20 split, embeds training responses into FAISS, runs the RAG pipeline on the held-out benchmark, saves `rag_results.csv` |
| 2 | `Notebook_2_Agentic_Bot.ipynb` | Implements intent classification + tool selection + tool execution (order status, refunds, cancellations, escalation), saves `agentic_results.csv` |
| 3 | `Notebook_3_Evaluation.ipynb` | Loads both result CSVs, computes 7 metrics using an LLM judge + lexical scorers, produces 5 comparison charts |

---

## Evaluation Metrics

| Metric | Method | Notes |
|---|---|---|
| **Intent Accuracy** | LLM judge | Same prompt for both bots — no bias |
| **Response Relevance** | LLM judge, 0–10 score normalized to 0–1 | Judged against shared POLICIES dict |
| **Hallucination Rate** | Rule-based check against `POLICIES` dict | Lower is better |
| **BLEU Score** | Lexical n-gram overlap with ground truth | Favours RAG by design (same corpus) |
| **ROUGE-L** | Longest common subsequence overlap | Same caveat as BLEU |
| **Latency (s/query)** | Wall-clock time per query | Lower is better |
| **Tool Call Accuracy** | Correct tool selected vs expected intent | Agentic bot only |

---

## Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (free tier works)
- Google Colab (recommended) or a local CPU/GPU environment

### Running on Google Colab

1. Upload all three notebooks to Colab.
2. Store your Groq API key as a Colab secret named `CST_API`:
   - Colab sidebar → 🔑 Key icon → **Add secret** → Name: `CST_API`
3. Run notebooks **in order**: Notebook 1 → Notebook 2 → Notebook 3.
4. Notebook 1 saves `benchmark_queries.csv` and `rag_results.csv`. Notebook 2 reads these and saves `agentic_results.csv`. Notebook 3 reads both result files and generates all plots.

### Dependencies (auto-installed in each notebook)

```
datasets
langchain
langchain-community
faiss-cpu
sentence-transformers
groq
gradio
tqdm
rouge-score
sacrebleu
```

---

## Results

### Summary Table

| Metric | RAG Bot | Agentic Bot | Winner |
|---|---|---|---|
| Intent Accuracy (%) | 40.0 | **83.3** | ✅ Agentic |
| Response Relevance (0–1) | 0.580 | **0.792** | ✅ Agentic |
| Hallucination Rate (%) | 18.3 | **16.7** | ✅ Agentic |
| Multi-step Success (%) | N/A | **40.0** | ✅ Agentic only |
| Avg Latency (s) | 5.70 | **4.49** | ✅ Agentic |
| BLEU Score | **12.92** | 3.92 | ✅ RAG |
| ROUGE-L Score | **0.295** | 0.221 | ✅ RAG |
| Tool Invocation Accuracy (%) | N/A | **96.7** | ✅ Agentic only |

---

### Plot 1 — Performance Comparison

![Performance Comparison](results/plots/plot1_performance.png)

The Agentic bot leads decisively on intent accuracy (**83.3% vs 40.0%**) and response relevance (0.792 vs 0.580). The RAG bot scores higher on BLEU and ROUGE-L because its responses closely mirror the Bitext ground-truth phrasing — a lexical similarity advantage that does not reflect actual correctness or helpfulness.

---

### Plot 2 — Hallucination Rate

![Hallucination Rate](results/plots/plot2_hallucination.png)

The RAG bot hallucinates at **18.3%** while the Agentic bot achieves **16.7%**. The Agentic bot's grounding against the structured `POLICIES` dictionary constrains its outputs, while the RAG bot is susceptible to generating plausible-sounding but unverified claims from retrieved chunks.

---

### Plot 3 — Latency per Query

![Latency per Query](results/plots/plot3_latency.png)

Average latency: **RAG 5.70s vs Agentic 4.49s**. The RAG bot's FAISS embedding + retrieval step adds consistent overhead that the Agentic bot avoids. Both bots use a single Groq LLM call per query, making retrieval the primary differentiating factor in latency.

---

### Plot 4 — Tool Invocation Accuracy (Agentic Bot Only)

![Tool Invocation Accuracy](results/plots/plot4_tool_accuracy.png)

The Agentic bot selects the correct tool on **96.7%** of queries overall. Perfect accuracy (100%) is achieved on `cancel_order`, `complaint`, `contact_human_agent`, and `track_order` intents. The `get_refund` and `payment_issue` intents reach 90%, reflecting the inherent ambiguity between these two closely related task types.

---

### Plot 5 — NLG Metrics (BLEU and ROUGE)

![NLG Metrics](results/plots/plot5_nlg_metrics.png)

The RAG bot's higher BLEU (12.92 vs 3.92) and ROUGE scores (ROUGE-1: 0.483 vs 0.352; ROUGE-L: 0.295 vs 0.221) are an artifact of retrieval: its responses are phrased similarly to the Bitext ground truth because both originate from the same corpus. This lexical overlap does not indicate higher quality — the **43-point intent accuracy gap** tells the opposite story.

---

## Key Design Choices

- **No data leakage** — The FAISS index is built only from the training 80%. Benchmark queries come exclusively from the held-out 20%, so retrieval cannot trivially match ground truth.
- **Unified LLM judge** — Both bots are scored with the exact same judge prompt, preventing evaluation bias from prompt asymmetry.
- **Reproducible tools** — The Agentic bot operates on a local copy of mock orders per query; no state bleeds between benchmark runs.
- **Groq for speed** — Using `llama-3.1-8b-instant` on Groq keeps latency low enough to benchmark 60 queries without hitting free-tier rate limits.
- **Shared policy grounding** — Both hallucination and relevance scores are anchored to the same `POLICIES` dictionary, making cross-architecture comparisons fair.

---

## Conclusion

The Agentic bot outperformed the Traditional RAG bot across the metrics that matter most for real-world customer support deployment.

The Agentic bot achieved significantly higher **intent accuracy** (83.3% vs 40.0%), **response relevance** (0.792 vs 0.580), and **tool invocation accuracy** (96.7%). It also produced a lower hallucination rate (16.7% vs 18.3%) and faster average latency (4.49s vs 5.70s). Its inclusion of a planning module further enables multi-step conversation handling at a 40.0% success rate — a capability the RAG architecture cannot support.

The RAG bot retains an advantage in surface-level language fluency (BLEU: 12.92 vs 3.92; ROUGE-L: 0.295 vs 0.221), which is an artifact of lexical overlap with the training corpus rather than a reflection of response quality or correctness.

For **task-oriented customer support**, the Agentic architecture is the stronger choice. For applications where fluency and naturalness of generated text are the primary requirement — such as content generation or document summarisation — RAG remains competitive.

---

## Project Structure

```
.
├── notebooks/
│   ├── Notebook_1_RAG_Bot.ipynb         # RAG pipeline
│   ├── Notebook_2_Agentic_Bot.ipynb     # Agentic pipeline
│   └── Notebook_3_Evaluation.ipynb      # Evaluation & plots
├── results/
│   ├── data/
│   │   ├── benchmark_queries.csv        # 60-query held-out benchmark
│   │   ├── rag_results.csv              # RAG bot responses + scores
│   │   ├── agentic_results.csv          # Agentic bot responses + scores
│   │   └── benchmark_summary.csv        # Aggregated 7-metric comparison
│   ├── plots/
│   │   ├── plot1_performance.png        # Intent accuracy, relevance, BLEU, ROUGE
│   │   ├── plot2_hallucination.png      # Hallucination rate comparison
│   │   ├── plot3_latency.png            # Per-query latency time series
│   │   ├── plot4_tool_accuracy.png      # Agentic tool invocation by intent
│   │   └── plot5_nlg_metrics.png        # BLEU, ROUGE-1, ROUGE-L detail
│   └── conclusion.txt
├── benchmark_summary.csv
├── requirements.txt
└── README.md
```

---

## References

- [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API](https://console.groq.com/docs)
- [LangChain](https://python.langchain.com/)
- Lewis et al., 2020 — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Yao et al., 2023 — *ReAct: Synergizing Reasoning and Acting in Language Models*
- Neha & Bhati, 2025 — *Evaluation framework for RAG and Agentic systems*

---

## Author

**Akshit Gupta**
