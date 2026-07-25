# CLAUDE.md — Seeing Through Deepfakes

M.Sc. dissertation proof-of-concept: multi-signal AI-image/deepfake detection (metadata heuristics + fine-tuned ViT + reliability-weighted fusion + conditional YuNet/DINOv2 deepfake stage), framed against EU AI Act Article 50.

- **Architecture, data flow, design rationale:** see [PROJECT.md](PROJECT.md).
- **Known bugs, tech debt, prioritized fixes:** see [GAPS.md](GAPS.md) — read it before "improving" anything; several oddities are documented empirical choices, and #1/#2 are real bugs with prescribed fixes.
- A Claude skill `deepfake-detection-framework` (outside this repo) holds the report's exact fusion formulas and report writing style — load it when touching fusion math or writing thesis prose.

## Commands

```bash
pip install -r requirements.txt        # NOTE: also needs faiss-cpu, matplotlib, pandas, seaborn (GAPS.md #4)
brew install exiftool                  # external binary, required by metadata module

python -m src.integration_pipeline.app                 # run the demo web UI (localhost, auto-opens browser)
python src/metadata_module/metadata_extraction.py IMG  # metadata analysis of one image, JSON to stdout
python src/deepfake_module/initialise_index.py         # (re)build FAISS landmark index — slow, downloads dataset
python src/visual_module/build_combined_dataset.py     # (re)build training corpus — downloads many GB
python -m src.visual_module.build_commfor_eval_index   # (re)scan CF eval shard index — committed, rarely needed
```

There is no build, lint, or test command. Tests don't exist yet (GAPS.md #3 specifies the intended pytest layout under `tests/`). Training and evaluation run through notebooks: `visual_classifier_finetuning.ipynb` (edit the config cell, run), `*_eval.ipynb` per module, `integration_pipeline/bulk_evaluation.ipynb` for end-to-end runs over `data/sample_images/`.

**The repo path contains spaces (iCloud Drive) — always quote paths in shell commands.**

## Conventions

- Layout: one package per evidence stream under `src/` (`metadata_module`, `visual_module`, `deepfake_module`, `integration_pipeline`). Reusable logic in `.py` modules; notebooks only orchestrate and display. Each package re-exports its public API in `__init__.py`.
- Run artefacts go to the module's `outputs/` dir, named `{run_name}_{what}.{json|png|csv}`. Run names follow `run_NN_<description>` (e.g. `run_01_stage2A_ft_combined`).
- British English identifiers: `analyse_image`, `initialise_index`, "Visualisation".
- Module outputs are dicts/Pydantic models carrying `probability` + `rationale`/`explanation` — every signal must expose a probability AND a reason, never a bare boolean. This is a thesis requirement (Article 50 explainability), not style.
- Imports from project root use the `src.` prefix; scripts/notebooks bootstrap `sys.path` with the project root first (see top of `app.py`).
- Type hints use modern syntax (`float | None`, `list[str]`); errors in the request path are caught broadly and returned as JSON `{"error": ...}` with a traceback printed.

## Gotchas

- **The canonical base model is `dima806/ai_vs_real_image_detection` everywhere** (David, 2026-07-19), paired with the `run_01_stage2A` delta. The other repo (`ai_vs_human_generated_image_detection`) only survives inside the run_02–04 delta checkpoints. `load_weight_delta` now RAISES on a base mismatch (`force=True` to override); `app.py` derives the base from the delta via `get_delta_base_model`.
- **The canonical fused decision threshold is 0.55** (David, 2026-07-19), set in `src/integration_pipeline/config.py` and the notebook config cells. The report's printed 0.5 predates this; historic 0.25/0.45 values were testing artefacts from the mispaired-model era. Don't change it without David.
- `gradcam_*_analysis.py` files implement **occlusion saliency**, not Grad-CAM. And `gradcam_landmark_analysis.py` can only be run from inside `src/deepfake_module/` (bare local import, GAPS.md #8).
- `DeepfakeClassifier.predict()` without a `visual_classifier` arg always runs the analysis (its internal gate is a no-op); the real proportionality gate is `fusion_result.is_ai` in `app.py`.
- Fine-tuned models are stored ONLY as int8 weight deltas in `src/visual_module/fine_tuned_model_delta/`; full checkpoints under `outputs/models/` are gitignored and may not exist on a fresh clone.
- Sample images encode ground truth in the filename: `{real|ai|deepfake}{±}metadata{±}place{±}face.ext`. `bulk_evaluation.ipynb` parses the prefix — keep the convention when adding samples.
- Missing/stripped metadata must score **≈0.5 (uncertain)**, never low ("real"). `score_features()` starts at 0.5 by design.
- The visual classifier's label strings differ per base model; `extract_visual_ai_probability()` in `fusion.py` normalises them — route any new prediction consumption through it.
- **`combined_dataset` test is in-distribution for the run_01 delta** — `build_combined_dataset.py` shuffles all three sources together *before* the 80/10/10 split, so train and test share generators. Scores there measure corpus fit, not generalisation; the Community Forensics eval set (Test 3 in `visual_module_eval.ipynb`) is the external check.
- The CF eval set is 206 GB over 413 shards, and **a shard is not one generator with its paired reals** — a generator spans many shards and any single shard is often all-real or all-AI. Select shards via `select_commfor_eval_shards()` off the committed index; evenly spaced sampling yields generators with one class and undefined AP. Shards range 4.7 MB–2.9 GB, so always keep a size cap.
- `data/` is ~88 GB and gitignored; never `git add` anything under `data/visual/`. Everything under `**/outputs/models/`, `**/fine_tuned_model/` is gitignored too.

## Rules

- **Never change fusion formulas, weights (`w_m`,`w_v`), accuracy scalars (`a_m`,`a_v`), or the D1/D2/D3 threshold logic without David's sign-off** — they are the thesis contribution and reported in his submitted work. Evolutions must stay interpretable weighted-evidence combinations (no black-box learned combiners) and any change must be reported, not silently applied.
- New detection signals must feed **into fusion** with their own `w`/`a` pair (numerator and denominator), never bypass it as a standalone verdict. Expensive analysis stages must sit behind the fused-positive gate (proportionality).
- Never invent hyperparameters, thresholds, or dataset numbers that aren't in the code or the skill references — ask David; a plausible fabricated value silently corrupts results.
- When retraining: keep the 80/10/10 split, 50/50 class balance, and seed 42 (`build_combined_dataset.py`) or results stop being comparable to the report's figures.
- Don't delete or overwrite anything in `fine_tuned_model_delta/`, `src/deepfake_module/models/`, or `outputs/` — committed experiment artefacts are expensive/impossible to regenerate.
- Known limitations (visual-classifier dominance in fusion, LEGO-face detections, controlled-test-split-only evaluation) are documented findings — report them in write-ups, don't quietly patch them away.
- Evaluation figures should match the report's visual language (labels "Real"/"AI-Generated", side-by-side stage comparisons) — reuse `training.py:evaluate_model()` or the skill's `eval_report.py`, don't hand-roll new chart styles.
