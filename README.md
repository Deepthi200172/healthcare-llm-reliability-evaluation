# LLM Healthcare Reliability Evaluation

**Thesis Title:** Comparative Evaluation of Open and Proprietary Large Language Models
Across Increasingly Complex Retrieval and Agentic Pipelines in the Healthcare Sector

---

## 1. Research Goal

This project benchmarks multiple LLMs in the healthcare domain across three pipeline
architectures to assess their reliability. The four reliability dimensions evaluated are:

| Dimension | What it Measures |
|---|---|
| **Factuality** | Is the answer medically correct vs the reference? |
| **Hallucination Safety** | Does the model avoid unsupported medical claims? |
| **Consistency** | Does the model give the same answer when asked 5 times? |
| **Robustness** | Does the model remain accurate under paraphrase, typo, or adversarial input? |

---

## 2. Dataset: PubMedQA

- Source: [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
- Configuration: `pqa_labeled` (1,000 expert-annotated QA pairs)
- Fields used: `question`, `context` (PubMed abstract), `long_answer`, `final_decision` (yes/no/maybe)
- Experiment size: **100 sampled questions**

---

## 3. The Three Pipelines

```
Pipeline 1 — Baseline
  question → LLM → answer

Pipeline 2 — RAG
  question → vector retrieval → [question + context] → LLM → answer

Pipeline 3 — Agentic RAG
  question → QueryAnalyzer → EvidenceRetriever → AnswerGenerator → Verifier → answer
```

---

## 4. The Four Reliability Metrics

### Factuality (F)
```
F = (1/N) × Σ factuality_score_i          score ∈ [0.0, 1.0]
```

### Hallucination Safety (H)
```
severity_i ∈ [0.0, 1.0]   (0 = no hallucination)
H = 1 − (1/N) × Σ severity_i
```

### Consistency (C)
```
C = (1/N) × Σ consistency_score_i          (each question run 5 times)
```

### Robustness (R)
```
drop_i    = max(0, F_original_i − F_perturbed_i)
R = 1 − (1/N) × Σ drop_i
```

---

## 5. Final Reliability Formula

```
Reliability = 0.30 × F + 0.30 × H + 0.20 × C + 0.20 × R
```

---

## 6. Installation

```bash
cd llm-healthcare-reliability-thesis
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your API keys
```

---

## 7. Running the Experiments

Run each step in order from the project root:

```bash
python experiments/01_prepare_data.py          # Download & sample PubMedQA
python experiments/02_run_generation.py        # Baseline + RAG + Agentic RAG
python experiments/03_run_consistency_generation.py   # 5 runs per question
python experiments/04_create_perturbations.py  # Paraphrase / noise / adversarial
python experiments/05_run_robustness_generation.py    # Run on perturbed questions
python experiments/06_run_llm_judge.py         # LLM-as-Judge scoring
python experiments/07_aggregate_results.py     # Final reliability scores
```

Or run everything at once:

```bash
python src/main.py
```

---

## 8. Output Files

| Path | Contents |
|---|---|
| `data/processed/pubmedqa_sample_100.csv` | 100 preprocessed questions |
| `data/perturbations/pubmedqa_perturbations.csv` | Perturbed question variants |
| `outputs/responses/generated_responses.csv` | All model responses (step 2) |
| `outputs/responses/consistency_responses.csv` | 5-run responses per question |
| `outputs/responses/robustness_responses.csv` | Responses to perturbed questions |
| `outputs/judge_scores/factuality_scores.csv` | Per-response factuality scores |
| `outputs/judge_scores/hallucination_scores.csv` | Per-response hallucination scores |
| `outputs/judge_scores/consistency_scores.csv` | Per-question consistency scores |
| `outputs/judge_scores/robustness_scores.csv` | Per-perturbation robustness scores |
| `outputs/final_results/model_pipeline_reliability_scores.csv` | **Final thesis table** |

---

## 9. Project Structure

```
llm-healthcare-reliability-thesis/
├── configs/          YAML config files (models, pipelines, evaluation, prompts)
├── data/             Raw and processed datasets + perturbations
├── src/
│   ├── data/         PubMedQA loader & preprocessor
│   ├── models/       LLM wrappers (OpenAI, Anthropic, Google, HuggingFace, Mock)
│   ├── pipelines/    Baseline, RAG, AgenticRAG pipelines
│   ├── retrieval/    Document loader, chunking, embeddings, ChromaDB, retriever
│   ├── agents/       QueryAnalyzer, EvidenceRetriever, AnswerGenerator, Verifier
│   ├── evaluation/   Judge prompts, LLMJudge, factuality/hallucination/consistency/robustness
│   ├── perturbation/ Paraphrase, noise, adversarial generators
│   └── utils/        Logger, file utils, metrics, config loader
├── experiments/      07 numbered runnable scripts
├── outputs/          All generated CSVs and logs
├── notebooks/        Exploration and visualization notebooks
└── tests/            pytest unit tests
```

---

## 10. Configuration

All model names and pipeline settings are in YAML — never hardcoded:

| File | Controls |
|---|---|
| `configs/models.yaml` | Which models to run and their API settings |
| `configs/pipelines.yaml` | Pipeline definitions (use_rag, use_agents, retrieval settings) |
| `configs/evaluation.yaml` | Sample size, consistency runs, metric weights, output paths |
| `configs/prompts.yaml` | All system/user/agent/judge prompts |

---

## 11. Current Limitations

- Only PubMedQA is supported as a dataset (extensible to MedQA, MedMCQA later)
- Open-source models require a GPU with sufficient VRAM (≥16GB recommended)
- LLM-as-Judge uses GPT-4o-mini by default; this adds API cost for scoring
- Agentic pipeline is simple Python classes, not a full agent framework

---

## 12. Future Work

- Add MedQA and MedMCQA datasets
- Migrate agentic pipeline to LangGraph
- Add ROUGE / BERTScore as supplementary automatic metrics
- Add cross-dataset generalisation evaluation
- Build interactive Streamlit dashboard for results
