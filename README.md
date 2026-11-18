# MMSD Hierarchical Fusion

This repository studies multimodal sarcasm detection from tweets and images on
MMSD. It reproduces the hierarchical fusion model (HFM) of Cai et al. (ACL 2019)
and tests whether changing its image attributes improves classification.

HFM combines tweet text, regional image features, and five predicted image
attributes. The proposed variant replaces COCO-based attribute prediction with
frozen CLIP selection of descriptive and affective words, while retaining the
local HFM fusion architecture. Experiments include notebooks and saved results.

## Experiments

| Directory | What it tests |
| --- | --- |
| `text_only/` | A text-only Bi-LSTM baseline: how much sarcasm can be detected without images. |
| `concat2/` | Concatenation of text and task-fine-tuned regional image features, without attributes. |
| `concat3_base_model/` | Three-way concatenation of text, ImageNet-pretrained regional features, and COCO attributes, without task-specific image fine-tuning. |
| `concat3_paper/` | Three-way concatenation using task-fine-tuned regional features and COCO attributes; the matched concatenation control for HFM. |
| `fusion_full/` | The reproduced HFM with COCO attributes, combining attribute-guided text encoding, representation attention, and learned modality weighting. |
| `fusion_clip/` | HFM with frozen CLIP attributes: four descriptive words and one affective word when affect beats neutral, otherwise five descriptive words. |
| `fusion_full_filtered/` | Whether removing common stopwords from COCO attribute rankings improves HFM. |
| `fusion_attribute_dropout/` | Whether dropping all attributes for some training examples reduces reliance on them; also evaluates missing and shuffled attributes. |
| `fusion_gated/` | Whether a tweet-conditioned gate improves control over attribute-guided text initialization. |

### CLIP controls and follow-up experiments

`fusion_clip/clip-attribute-ablation.ipynb` compares two attribute selection rules
within the same CLIP-HFM architecture:

- **Descriptive only:** always uses the five highest-ranked descriptive words.
- **Sentiment aware:** replaces the fifth description with an affective word when
  its score exceeds neutral, testing whether affective replacement helps.

It also tests stricter affect-versus-neutral margins of `0.01`, `0.02`, and
`0.04` against the default `0`, selecting the margin using validation F1.
Results are retained in `fusion_clip/results_ablation/`.

## Supporting Files

The `notebooks/` directory contains image-classifier fine-tuning, regional feature
extraction, and COCO attribute-predictor training. These prepare the visual inputs
used by the experiments.

`report.pdf` and `report.tex` present the literature review, methods, results, and
limitations. `scripts/report_results.py` verifies saved predictions and generates
the report tables in `tables/`; `figures/` contains the report diagrams.
