# Statute grounding for legal provision citation under India's BNS

Code, data and per-query outputs for the paper

> H. Kag, S. K. Puthanveettil, K. Ryagnirvesh, S. Panchal. *Do language models know India's new criminal code? Statute grounding, model scale and leakage-free evaluation for legal provision citation.* Manuscript submitted to *Data & Knowledge Engineering*, 2026.

India's Bharatiya Nyaya Sanhita (BNS) replaced the Indian Penal Code (IPC) on 1 July 2024 and renumbered its provisions: murder moved from IPC 302 to BNS 103, theft from 379 to 303, cheating from 420 to 318. A language model answering from memory may therefore name the right offence and cite a repealed section. This repository contains:

- **BNS-Facts-135**, a leakage-free benchmark of 135 fact-pattern queries, each grounded in a specific BNS section
- a **statute-grounded knowledge graph** of the BNS, with edges read from each section's marginal title
- every script, retrieval context and per-query model output behind the paper's tables and figures

## Key result

Correct BNS section cited on BNS-Facts-135 v1.1 (n = 135):

| Model | No grounding | + retrieved statute | + statute + graph |
|---|---|---|---|
| Claude Opus 5.5 | 99.3% | — | — |
| Claude Haiku 4.5 | 31.9% | 88.1% | 90.4% |
| Qwen 2.5 7B | 0.7% | 79.3% | 79.3% |
| Llama 3.1 8B | 0.0% | 81.5% | 80.0% |
| Llama 3.2 3B | 0.0% | 80.0% | 76.3% |

Without grounding, small models cite repealed IPC numbers (up to 67% of queries) or confabulate sections. Retrieved statute text fixes this (every none-vs-statute McNemar test p < 1e-19). Given correct evidence they cite correctly 93–98% of the time, so their residual error is retrieval error. Adding graph structure to the text never helps significantly.

## Layout

```
data/            BNS-Facts-135, the statute-grounded graph, the grounding contexts
results/         every table in the paper and every per-query model output
scripts/         numbered in execution order
figures/         Figures 1-3 of the paper (Figure_2.tex is the diagram's source)
corpus/          BNS Bill text, used only to read the marginal titles
original_data/   inputs from the authors' earlier GraphRAG-Law system: statute
                 chunks, its FAISS index, its keyword offence links and its
                 110-query benchmark (read, never modified)
```

## Benchmark

`data/bns_facts_135_v1_1.csv` is the version used in the paper. `data/bns_facts_135.csv` (v1.0) is kept for provenance, and `data/bns_facts_135_v1_1_changelog.csv` lists the nine corrected items and the reason for each.

| column | meaning |
|---|---|
| `query_id` | F001–F135 |
| `query` | a lay description of a situation; it never contains the offence name |
| `expected_offence` | one of nine: cheating, extortion, forgery, murder, rape, robbery, sexual harassment, stalking, theft |
| `difficulty` | `plain` (90) or `boundary` (45, close to a neighbouring offence) |
| `grounding_section` | the BNS section whose conduct the item describes |

The authors drafted the items from the statutory text and audited them against it. They are research data, not legal advice.

## Knowledge graph

| file | content |
|---|---|
| `data/bns_act_titled.csv` | the 350 distinct BNS sections with marginal titles aligned onto Act numbering |
| `data/graph_defines.csv` | statute-grounded DEFINES edges for the nine offence families (39 edges) |
| `data/graph_defines_expanded.csv`, `data/offence_taxonomy.csv` | the full taxonomy: 272 sections, 271 offences |
| `data/grounding_contexts_v11.json` | the top-5 retrieved sections shown to every model, per query |

## Where each result comes from

| Paper | Script | Output in `results/` |
|---|---|---|
| Sec. 4, re-examination of the original benchmark | `03_evaluate_on_hard.py`, `13_gate_experiment.py` | `hard_benchmark_eval.csv`, `gate_results.csv` |
| Table 2, leakage | `02_validate_benchmark.py`, `03_evaluate_on_hard.py` | `../data/bns_facts_135_validated.csv`, `hard_benchmark_eval.csv` |
| Table 3, Fig. 1, retrieval on v1.1 | `25_eval_local_v11.py` | `final_table_v11.csv`, `per_query_outcomes_v11.csv` |
| Table 4, keyword vs statute edges | `32_graph_ablation_v11.py` | `graph_ablation_v11.csv`, `graph_edges_mcnemar_v11.csv` |
| Sec. 7 diagnostics on v1.0 | `14_`, `16_`–`20_` | `scope_results.csv`, `retrieval_variants.csv`, `error_diagnosis.csv`, `encoder_comparison.csv`, `final_table.csv`, `bm25_results.csv` |
| Benchmark audit (Opus zero-shot) | `23_llm_baseline.py` | `llm_baseline_outputs.csv`, `llm_vs_systems.csv` |
| Table 5, Fig. 3, model scale × grounding | `26_`, `28_`, `29_`, then `30_scale_table.py` | `verifiability_outputs.csv`, `scale_haiku_outputs.csv`, `scale_local_outputs.csv`, `scale_final_table.csv`, `scale_final_long.csv` |
| Table 6, error decomposition | `31_error_decomposition.py` | `error_decomposition.csv` |
| Figures | `33_make_figures.py`, `34_crop_figure.py` | `../figures/` |

`results/llm_ledger.json` records every API charge; all API experiments together cost $1.05.

## Reproducing

**Environment** (Python 3.10; tested on Windows 11, CPU only):

```
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
```

Run scripts from the repository root, for example `python scripts/25_eval_local_v11.py`. Every path is relative to the repository.

**Retrieval experiments** run locally and cost nothing:

| step | script |
|---|---|
| parse the BNS and align marginal titles | `10_parse_bns.py`, `11_align_titles.py` |
| build the statute-grounded graph | `12_build_graph.py`, `15_expand_taxonomy.py` |
| build, validate and correct the benchmark | `01_build_hard_benchmark.py`, `02_validate_benchmark.py`, `24_benchmark_v11.py` |
| retrieval systems on v1.1 | `25_eval_local_v11.py` |
| keyword vs statute edges | `32_graph_ablation_v11.py` |

**Claude models** need an Anthropic API key in a `.env` file in the repository root:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_WORKSPACE_ID=wrkspc_...   # only for organisation-level keys
```

`.env` is listed in `.gitignore`; never commit it. Each script has `pilot`, `submit` and `collect` steps and refuses to submit a batch whose measured cost would exceed its budget.

| step | script | cost (Message Batches API) |
|---|---|---|
| zero-shot classification | `23_llm_baseline.py` | ≈ $0.16 |
| Opus 5.5 citation | `26_llm_verifiability.py` | ≈ $0.61 |
| shared grounding contexts | `27_build_contexts.py` | free |
| Haiku 4.5, three conditions | `28_haiku_scale.py` | ≈ $0.28 |

**Open models** run through [Ollama](https://ollama.com) as Q4_K_M builds with temperature 0. `29_local_scale.py` expects a server at `http://127.0.0.1:11435`; set `OLLAMA_HOST_URL` to use another one. The three models need about 11 GB of disk.

Results for hosted models describe the versions available in October 2026 and may not reproduce exactly later.

## Licence

- Code (`scripts/`): MIT, see `LICENSE`.
- The authors' data and outputs (`data/`, `results/`, `figures/`): CC BY 4.0, see `DATA_LICENSE.md`.
- Statutory text (`corpus/`, and the statute text in `original_data/`) is the text of Acts and Bills of the Parliament of India, included for reproducibility. Neither licence above applies to it.

## Citation

Until the article is published, please cite this repository; `CITATION.cff` has the details, and GitHub's "Cite this repository" button formats them.

This is an information-retrieval research artefact, not a substitute for advice from a qualified lawyer.
