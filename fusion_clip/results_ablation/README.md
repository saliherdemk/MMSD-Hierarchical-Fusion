# CLIP attribute ablation results

- `descriptive_only/`: the three completed runs (`seed_42`–`seed_44`) and
  `summary.json`. Each seed keeps configuration, history, results, and test and
  validation predictions.
- `threshold_screening/`: seed-42 validation evidence for margins 0, 0.01,
  0.02, and 0.04, plus `selection.json`. Zero won; there is no separate selected
  threshold model. These are validation experiments, not four test results.
- `clip_cache/`: the frozen CLIP score cache and extraction manifest. Scores
  retain the original five descriptions and all nine affect/neutral scores,
  allowing every attribute rule to be reconstructed without another CLIP pass.
- Notebook: `../clip-attribute-ablation.ipynb`.

The fresh sentiment-aware runs produced exactly the same test probabilities
as `../results/runs/hierarchical_finetuned_clip_sentiment_packed/` for all three
seeds. Those existing files are the comparison baseline; duplicate copies are
not stored here. Descriptive-only uses the same expanded embeddings and HFM.

Descriptive-only F1 is **79.74 ± 0.10%**, versus **80.32 ± 0.38%** for
sentiment-aware CLIP. The mean difference is **+0.58 percentage points**.
`python scripts/report_results.py` from the repository root verifies predictions,
checkpoint selection, summaries, and threshold selection and regenerates the
paper tables.

Generated case lists, duplicate aggregate tables, reconstructed attribute files,
download metadata, and checkpoints are omitted from this compact research
record. Hashes referring to original checkpoints remain in saved run metadata;
the compact layout is for analysis, not checkpoint-based training resume.
