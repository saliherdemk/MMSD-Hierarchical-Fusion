# MMSD-Hierarchical-Fusion

Pipeline-level reproduction of Cai et al. (ACL 2019), with **sentiment-aware CLIP
image attributes** as the proposed improvement. CLIP selects five words for the
existing HFM; it does not replace the regional ResNet features or text encoder.

## Paper and checked results

- `report.tex`: report following the eight sections in the YZM5110 brief.
- `report.pdf`: compiled paper. Fill the author block in `report.tex`, rebuild,
  and name the submitted PDF `<LastName>_YZM5110_Research.pdf`.
- `scripts/report_results.py`: standard-library-only verification and table generation.
- `tables/`: generated tables and a SHA-256 audit of the prediction files.

From the repository root:

```sh
python scripts/report_results.py
tectonic report.tex
```

Install [Tectonic](https://tectonic-typesetting.github.io/) to build the PDF;
its first build downloads TeX packages. The schematic is editable TikZ source
in `figures/clip_pipeline.tex`. Source-paper figure credits are in
`figures/SOURCES.md` and the captions.

The verification command needs no dataset, GPU, or training libraries. It checks
36 test prediction files (30 clean and six corruption conditions), seven
ablation validation prediction files, labels and IDs,
threshold decisions, four metrics, summary statistics, and validation-selected
epochs. It does **not** certify the historical producing code.

## Experiments

| Directory | Experiment |
| --- | --- |
| `text_only/` | Text Bi-LSTM |
| `concat2/` | Text + fine-tuned regional image features |
| `concat3_base_model/` | Text + ImageNet regional features + COCO attributes |
| `concat3_paper/` | Text + fine-tuned regional features + COCO attributes |
| `fusion_full/` | HFM with COCO attributes |
| `fusion_clip/` | HFM with frozen CLIP descriptive/affective attributes |
| `fusion_full_filtered/` | Inference-time stopword filtering |
| `fusion_attribute_dropout/` | Whole-attribute dropout and corruption evaluation |
| `fusion_gated/` | Tweet-conditioned gate on early attribute fusion |

The retained CLIP runs report 80.32% mean test F1 versus 80.16% for the archived
COCO-HFM baseline, with slightly lower accuracy. This is a preliminary descriptive
difference, not a verified causal or statistically significant improvement.
The experiment author confirms that the retained COCO-HFM baseline results were
regenerated after the padding correction; both principal HFM variants use
packed recurrence and masked text pooling/attention.
The descriptive-only CLIP control reaches 79.74% mean F1, versus 80.32% for
sentiment-aware CLIP. The +0.58-point difference is positive across all three
seeds. Validation screening selects the original zero-margin rule over stricter
affective replacement. Compact evidence is in `fusion_clip/results_ablation/`.

## Data and training setup

1. Obtain the original MMSD assets from
   <https://github.com/headacheboy/data-of-multimodal-sarcasm-detection>, revision
   `f42b16510208624d91fa545ca9bb64c6335f971e`. Follow its image-download instructions.
   Keep the assets local: the inspected upstream repository has no license file,
   so public availability alone does not authorize redistribution.
2. Arrange the text/vocabulary assets under `data/raw/mmsd/` and images under
   `data/raw/dataset_image/`. Additional source notes are in local `data/SOURCE.md`.
3. Run `notebooks/train_resnet50v2_image_classifier.ipynb`, then set its checkpoint
   in `notebooks/extract_resnet50v2_image_regions.ipynb` and execute all three parts.
   Regional arrays are cached under `data/features/resnetv2_50_a1h_in1k_finetuned/`.
   The float32 arrays require about 39.5 GB before headers.
4. For the COCO-HFM baseline, obtain COCO 2017 images/captions from
   <https://cocodataset.org/>, run
   `notebooks/train_resnet101_coco_attribute_predictor.ipynb`, and set its output
   as `ATTRIBUTE_PREDICTIONS` in `fusion_full/fusion-full.ipynb`.
5. For CLIP-HFM, open `fusion_clip/fusion-clip.ipynb`. Set `DATA_ROOT`, `ASSET_ROOT`,
   `FEATURE_DIR`, and `IMAGE_DIR` to your mounts, and choose fresh
   `CLIP_ATTRIBUTE_DIR` and `RUN_ROOT` paths. Run the notebook from top to bottom.
   It extracts attributes with pinned frozen CLIP, then trains seeds 42–44.
   The COCO predictor is not required for this branch; the supplied attribute
    vocabulary and embedding matrix are reused.

### CLIP attribute follow-up experiments

`fusion_clip/clip-attribute-ablation.ipynb` is a standalone notebook for the
descriptive-only control and validation-selected affective-margin experiment.
Edit the mounts and `OUTPUT_ROOT`, then run top to bottom. It caches the original
five descriptive words and all affect/neutral scores in one frozen CLIP pass,
screens margins `0, 0.01, 0.02, 0.04` on seed-42 validation F1, and freezes the
selection before evaluating the finalists across seeds 42–44. A fresh zero-margin
baseline is included. This takes 9–11 classifier fits, with completed screening
fits reused. It saves predictions, insertion frequencies, paired errors,
mean/SD tables, and `findings.md` under the new output root. Interrupted fits
are preserved rather than overwritten; move an incomplete run aside before
retrying, or use a fresh output root. Existing experiment outputs are unchanged.

The imported results have a compact repository layout: `results_ablation/`
contains `descriptive_only/seed_42`–`seed_44` and `summary.json`, seed-42
threshold-screening evidence, and the reusable CLIP score cache. Duplicate
sentiment-aware predictions reuse the original result directory. See
`fusion_clip/results_ablation/README.md` for retained files and interpretation.

Training notebooks target Kaggle GPU mounts. Saved multimodal configurations
record Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3, CUDA 13.0, and cuDNN
92700. The CLIP notebook installs Transformers 4.57.1 and tqdm 4.67.1; its
extraction manifest records Pillow 13.0.0. HFM runs list two Tesla T4 GPUs.
These are recorded environments, not a validated lockfile; root
`requirements.txt` alone is insufficient to recreate training. Preparation also
requires the `timm`/`torchvision` stack used in the respective notebooks.

After installing Jupyter and the notebook's training dependencies, and editing
the paths, a headless execution command is:

```sh
mkdir -p executed
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir=executed fusion_clip/fusion-clip.ipynb
```

For the control, substitute `fusion_full/fusion-full.ipynb`. Use fresh output
directories: the CLIP notebook refuses to overwrite extracted attributes or
existing seed directories. Its synthetic `check_padding` function supplies a
small model check during execution; the prediction audit above is the cheaper
CPU-only verification path. Full extraction/training needs downloaded assets,
substantial storage, and preferably a GPU. The paper reports preparation and
task-stage costs separately.

CLIP results are retained in
`fusion_clip/results/runs/hierarchical_finetuned_clip_sentiment_packed/`;
the extraction manifest, words, and scores are in
`fusion_clip/results/features/clip_sentiment_attributes/`.
Large training checkpoints are excluded by `.gitignore`.

## Reproducibility boundaries

Per-run configs and histories are retained, but historical source commits and
fine-tuned visual/COCO checkpoint hashes are incomplete. Some supporting
notebooks (concatenation and filtered attributes) still warn about older saved
results; the main COCO-HFM rerun has been confirmed by the experiment author.
The report distinguishes
the original paper's numbers from local measurements and documents these gaps.
Publish the final files with a commit or tag and record that revision when
submitting; the current report identifies its inspected base revision rather
than claiming a historical training revision.
