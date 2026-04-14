# Travel Photo Curation Pipeline

This project turns a large travel-photo folder into a smaller, more usable shortlist for:

- preserving good memories
- reducing duplicate or near-duplicate shots
- surfacing printworthy photos
- generating faithful enhancement prompts for final upscaling or cleanup

The current workflow is built around:

- Stage 1 local curation with Python + OpenCV + lightweight heuristics
- Stage 2 ranking with vision-language models
- optional Vertex-based image improvement for the final ranked shortlist

The workflow described here reflects the current working setup and intentionally ignores older experimental paths that we are no longer documenting.

## Current Pipeline

### Stage 1: Local Mechanical Curation

Stage 1 processes a local folder directly. This is the recommended mode when your Google Drive photos are already synced to disk.

What Stage 1 does:

- discovers supported image files in a folder or its child albums
- normalizes EXIF orientation before scoring
- supports common image formats plus `HEIC` / `HEIF`
- rejects images below `1 MP`
- splits outputs into:
  - `Above_8MP`
  - `Between_1MP_and_8MP`
- uses a soft blur strategy:
  - very blurry images are hard rejected
  - moderately soft images stay eligible and are ranked instead
- applies exposure checks
- builds strict duplicate groups for winner selection
- builds looser `scene_family_id` metadata for “same moment / zoom variant / reframed variant” comparison
- copies matching `.RAF` files for selected `.jpg` / `.jpeg` / `.png` files when present
- writes a full `selection_log.csv`

### Stage 2: Vision-Language Ranking

Stage 2 works on Stage 1 outputs.

It runs in two passes:

1. `scene family reduction`
   Within each `scene_family_id`, the model keeps only the strongest images.

2. `portfolio reranking`
   From the family-reduced survivors, the model creates a final ranked shortlist of printworthy images.

The current recommended local models are:

- `gemma3:27b`
- `gemma4:31b`

Stage 2 also asks the ranking model for structured print-improvement guidance for each ranked image, including:

- whether the image is worth enhancing
- image type
- strengths
- enhancement goal
- safe edits
- conditional edits
- edits to avoid
- risk flags

Those fields are stored in `portfolio_summary.json` and are later used to build image-improvement prompts.

### Optional Final Improvement

After Stage 2 ranking, you can run a Vertex image-improvement pass on the ranked shortlist.

This step:

- reads the ranked images
- reads the stored improvement guidance from `portfolio_summary.json`
- creates improved print-ready outputs in `ranked_improved/`
- writes per-image metadata and a summary file

## Repository Layout

- [main.py](/Users/aditya/Documents/trial_pics/main.py): Stage 1 CLI entry point
- [config.py](/Users/aditya/Documents/trial_pics/config.py): Stage 1 thresholds and grouping config
- [pipeline.py](/Users/aditya/Documents/trial_pics/pipeline.py): Stage 1 local curation logic
- [image_filters.py](/Users/aditya/Documents/trial_pics/image_filters.py): decode, blur, exposure, and HEIC handling
- [stage2_scene_family_pipeline.py](/Users/aditya/Documents/trial_pics/stage2_scene_family_pipeline.py): Stage 2 local VLM ranking via Ollama
- [stage2_gemini_model_suite.py](/Users/aditya/Documents/trial_pics/stage2_gemini_model_suite.py): Gemini-based Stage 2 suite
- [vertex_ranked_photo_improver2.py](/Users/aditya/Documents/trial_pics/vertex_ranked_photo_improver2.py): primary Vertex image-improvement pass (recommended)
- [vertex_ranked_photo_improver.py](/Users/aditya/Documents/trial_pics/vertex_ranked_photo_improver.py): older Vertex variant
- [gemini_ranked_photo_improver.py](/Users/aditya/Documents/trial_pics/gemini_ranked_photo_improver.py): legacy direct Gemini improver (not the primary path)
- [ALGORITHM_LOGIC.md](/Users/aditya/Documents/trial_pics/ALGORITHM_LOGIC.md): deeper algorithm notes

## Requirements

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Key Python dependencies:

- `opencv-python-headless`
- `Pillow`
- `numpy`
- `open-clip-torch`
- `torch`
- `transformers`

For local Stage 2 with Ollama, you also need:

- [Ollama](https://ollama.com/)
- the local model you want to use, for example:
  - `ollama pull gemma3:27b`
  - `ollama pull gemma4:31b`

For Gemini-based ranking or Vertex improvement, set:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## Stage 1 Usage

Run Stage 1 on a local album folder:

```bash
python3 main.py \
  --local-root "/path/to/album" \
  --local-output-name Curated_Best_Stage2Ready
```

Useful flags:

- `--min-mp`
- `--medium-min-mp`
- `--laplacian-threshold`
- `--similarity-threshold`
- `--check-background-people`
- `--local-output-name`

Example with a custom output name:

```bash
python3 main.py \
  --local-root "/Users/aditya/Library/CloudStorage/GoogleDrive-adityasengariitd@gmail.com/My Drive/Pictures/Italy June 2025" \
  --local-output-name Curated_Best_Stage2Ready
```

### Stage 1 Output Structure

Inside each processed album folder, Stage 1 creates a new output folder like:

```text
Curated_Best_Stage2Ready/
  Above_8MP/
  Between_1MP_and_8MP/
  selection_log.csv
```

The CSV includes:

- every input JPG row
- selection status
- rejection reason
- strict duplicate group information
- scene family information
- the selected winner for each strict group
- quality metrics such as blur and exposure values
- copied RAW path when a matching `.RAF` was found

## One-Command Album Run (Recommended)

If you want the same output naming every time and do not want to keep editing folder names manually, use:

[run_album_pipeline.py](/Users/aditya/Documents/trial_pics/run_album_pipeline.py)

Typical usage:

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py
```

What it does by default:

- Stage 1:
  - processes the current album folder
  - writes to `Curated_Best_Stage2Ready`
- Stage 2:
  - runs model `gemma4:31b`
  - runs family reduction and final portfolio ranking
  - writes to model-specific `Stage2_...` folders
  - if portfolio ranking fails after family reduction, it automatically retries from family outputs with a smaller batch size

Useful options:

- `--model gemma4:31b`
- `--reuse-stage1` to skip recomputing Stage 1 if already present
- with `--reuse-stage1`, the launcher automatically reuses existing family reduction if available
- `--force-family-rerun` if you explicitly want to recompute family reduction
- `--skip-stage2` to run Stage 1 only
- `--batch-size`, `--family-divisor`, `--portfolio-divisor`, `--timeout`

Example reusing existing Stage 1:

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --reuse-stage1
```

### Common Run Modes

Fresh full run (default model `gemma4:31b`):

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py
```

Fresh full run with a different model:

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --model gemma3:27b
```

Resume from existing Stage 1 output:

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --reuse-stage1
```

Resume and continue from existing family reduction (automatic with `--reuse-stage1`):

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --reuse-stage1
```

Force family-reduction recalculation (even if an old `Stage2_FamilyReduced_*` exists):

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --reuse-stage1 --force-family-rerun
```

Run only Stage 1 (skip Stage 2):

```bash
cd "/path/to/one/album/folder"
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --skip-stage2
```

## Stage 2 Usage: Local Ollama Models

Stage 2 can start from a full Stage 1 folder or resume from an existing family-reduced folder.

### Full Stage 2 run from Stage 1

Example with `gemma3:27b`:

```bash
python3 /Users/aditya/Documents/trial_pics/stage2_scene_family_pipeline.py \
  --stage1-dir "/path/to/Curated_Best_Stage2Ready" \
  --model gemma3:27b
```

Example with `gemma4:31b`:

```bash
python3 /Users/aditya/Documents/trial_pics/stage2_scene_family_pipeline.py \
  --stage1-dir "/path/to/Curated_Best_Stage2Ready" \
  --model gemma4:31b
```

### Resume only the portfolio step

If family reduction already exists:

```bash
python3 /Users/aditya/Documents/trial_pics/stage2_scene_family_pipeline.py \
  --resume-family-dir "/path/to/Stage2_FamilyReduced_gemma3_27b" \
  --model gemma3:27b
```

### Important Stage 2 flags

- `--family-divisor`
  Controls how aggressively each scene family is reduced. Default: `4`
- `--portfolio-divisor`
  Controls how aggressively the final survivor pool is reduced. Default: `4`
- `--batch-size`
  Number of images per portfolio batch. Default: `12`
- `--scene-family-ids`
  Target only specific scene families
- `--max-families`
  Smoke-test on a small subset
- `--timeout`
  Per-model-call timeout in seconds
- `--family-output-dir`
  Custom family-reduction folder
- `--portfolio-output-dir`
  Custom final portfolio folder

### Stage 2 Output Structure

Family-reduction folder:

```text
Stage2_FamilyReduced_<model_slug>/
  decisions/
  images/
  family_reduction.csv
  family_reduction_summary.json
```

Portfolio folder:

```text
Stage2_PortfolioTop_<model_slug>/
  portfolio_batch_01.json
  portfolio_batch_02.json
  ...
  portfolio_summary.json
  ranked/
```

`portfolio_summary.json` is the main artifact. It contains:

- final ranked filenames
- the top filename
- batch reports
- final reasons
- print-improvement guidance per ranked image
- a ready `print_improvement_prompt` for downstream enhancement

## Stage 2 Usage: Gemini Ranking Suite

If you want to run Stage 2 using Gemini instead of Ollama:

```bash
python3 /Users/aditya/Documents/trial_pics/stage2_gemini_model_suite.py \
  --stage1-dir "/path/to/Curated_Best_Stage2Ready" \
  --model gemini-2.5-pro \
  --suite-output-dir "/path/to/Gemini_Stage2"
```

You can pass `--model` multiple times to compare Gemini models in one run.

Useful Gemini suite flags:

- `--family-divisor`
- `--portfolio-divisor`
- `--batch-size`
- `--scene-family-ids`
- `--max-families`
- `--timeout`
- `--suite-output-dir`
- `--api-key-env`
- `--review-max-side`
- `--review-jpeg-quality`
- `--review-max-bytes`
- `--dry-run`

## Final Improvement with Vertex (Recommended Defaults)

If you already ran:

```bash
python3 /Users/aditya/Documents/trial_pics/run_album_pipeline.py --reuse-stage1
```

and you are still in that album folder, run:

```bash
python3 /Users/aditya/Documents/trial_pics/vertex_ranked_photo_improver2.py --overwrite
```

This uses default settings (`gemini-3-pro-image-preview`, `4K`, unified prompt path) and writes:

```text
Stage2_PortfolioTop_<model_slug>.../
  ranked_improved_vertex2_unified_full_final/
    meta/
    improvement_summary.json
```

### Run from a particular Stage 2 folder

If you want to target one specific portfolio folder explicitly:

```bash
python3 /Users/aditya/Documents/trial_pics/vertex_ranked_photo_improver2.py \
  --portfolio-dir "/absolute/path/to/Stage2_PortfolioTop_..._printplans" \
  --overwrite
```

### Quick smoke test before full run

```bash
python3 /Users/aditya/Documents/trial_pics/vertex_ranked_photo_improver2.py \
  --portfolio-dir "/absolute/path/to/Stage2_PortfolioTop_..._printplans" \
  --limit 5 \
  --overwrite
```

### Useful Vertex options

- `--project-id`
- `--location` (default: `global`)
- `--model` (default: `gemini-3-pro-image-preview`)
- `--analysis-model` (default: `gemini-2.5-flash`)
- `--image-size` (default: `4K`)
- `--pad-percent`
- `--input-max-side`
- `--input-jpeg-quality`
- `--start-rank`
- `--limit`
- `--overwrite`

### Legacy direct Gemini improver (optional)

If you ever need the older direct Gemini path, the script is:

[gemini_ranked_photo_improver.py](/Users/aditya/Documents/trial_pics/gemini_ranked_photo_improver.py)

This is no longer the primary enhancement route in this project.

## Current Selection Philosophy

This project intentionally separates three ideas:

1. `keepable`
   Good enough to preserve in the album.

2. `printworthy`
   Strong enough to print or frame.

3. `enhanceable`
   Worth sending through a careful improvement step.

That means:

- a photo can be keepable but not printworthy
- a photo can be printworthy because it is a beautiful family photo
- a photo can also be printworthy because it is beautiful as a travel / city / architectural image

## Practical Notes

- Stage 1 works best on locally synced photo folders.
- RAF files are copied as companion RAWs when a matching selected JPG or PNG exists.
- HEIC support is handled locally through the current decode path.
- Stage 2 is conservative at the scene-family level and more subjective at the final portfolio level.
- `scene_family_id` is broader than the strict Stage 1 duplicate grouping and is meant for human-style comparison.

## Known Limitations

- RAW files are not yet processed as primary ranking inputs.
- The face-based scene checker is still experimental and can misread statues or artwork.
- CLIP is not a major decision-maker in the current travel-album setup.
- Generative image improvement can drift if prompts are too semantic or too permissive.
- Some museum or artwork shots can still be overrated by ranking models.

## Recommended Working Flow

1. Run Stage 1 on the album.
2. Run Stage 2 with `gemma4:31b` or `gemma3:27b`.
3. Inspect `portfolio_summary.json` and the `ranked/` folder.
4. Run Gemini improvement only on the top ranked subset you actually want to polish.
5. Manually spot-check a few improved outputs before processing the whole shortlist.

## More Detail

For a lower-level explanation of grouping, thresholds, and CSV fields, see:

[ALGORITHM_LOGIC.md](/Users/aditya/Documents/trial_pics/ALGORITHM_LOGIC.md)
