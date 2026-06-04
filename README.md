# Search Engines 2026 – Project

Code for the Search Engines 2026 project (Natural Questions document ranking).

## Repository layout

```
project_handout/   docs, training queries, qrels, unseen test queries
notebooks/         se_project_2 … se_project_8 (one notebook per task)
indexes/           built locally
results/           tuning caches + per-task output
```

## Requirements

- Python 3.10+
- Java 11 (for PyTerrier / Terrier)
- A GPU is strongly recommended for `se_project_6` (LLM prompting) and
  `se_project_8` (biencoder reranking).

Python dependencies are installed from inside each notebook (first cell, `%pip install ...`).

## How to run

1. Place the dataset files in `project_handout/`:
   - `docs2.jsonl`
   - `train_queries.csv`, `train_qrels.csv`
   - `test_queries.csv`

2. Open the notebooks in order:

   | Notebook              | Task                                            |
   |-----------------------|-------------------------------------------------|
   | `se_project_2.ipynb`  | Dataset analysis & indexing (builds `indexes/`) |
   | `se_project_3.ipynb`  | BM25 & Hiemstra_LM tuning + evaluation          |
   | `se_project_4.ipynb`  | Pseudo-relevance feedback (RM3) & word2vec QE   |
   | `se_project_6.ipynb`  | Query expansion with an LLM                     |
   | `se_project_7.ipynb`  | Runs on the unseen queries                      |
   | `se_project_8.ipynb`  | Biencoder reranking (Sentence-BERT)             |

   Each notebook resolves its paths relative to its own location, so just
   run all cells top-to-bottom.

3. Notebooks 4, 6, 7 and 8 read tuning caches from `results/` produced by
   earlier notebooks – do not skip ahead without running them first.