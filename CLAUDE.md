# Photo Curation Pipeline

Travel photo curation pipeline: Stage 1 (quality filtering + dedup) → Stage 2 (VLM scene ranking via Ollama) → Stage 3 (Vertex AI image improvement). Also includes a standalone quick-fix tool for processing individual images through Vertex AI.

## Quick Start

All commands run from this directory via `./run.sh`:

```bash
./run.sh check                              # verify environment health (run this first)
./run.sh stage1 /path/to/album              # Stage 1: quality filtering
./run.sh stage2 /path/to/Curated_Best_Stage2Ready   # Stage 2: VLM ranking
./run.sh stage3 /path/to/Stage2_PortfolioTop_gemma4_31b  # Stage 3: Vertex polish
./run.sh fix photo.JPG                      # quick-fix: default auto-prompt
./run.sh fix photo.JPG --gemma              # quick-fix: gemma writes the prompt
./run.sh fix photo.JPG --gemma --prompt "enhance sky"  # gemma + your custom prompt
```

## Python Environments

Two separate Python interpreters are used:

- **Stage 1 & 2**: system `python3` (Anaconda 3.8) — has cv2, open_clip, torch
- **Stage 3 & quick-fix**: `.venv/bin/python3` (Python 3.10) — has google.auth, pillow-heif, certifi

The `run.sh` wrapper handles this automatically. If running scripts manually:
```bash
SSL_CERT_FILE=.venv/lib/python3.10/site-packages/certifi/cacert.pem \
  .venv/bin/python3 vertex_quick_fix.py --images photo.JPG
```

## Google Cloud

- Project: `project-e7328f60-223b-43f8-95d`
- Auth: `gcloud auth application-default login` (run once, credentials cached at ~/.config/gcloud/)
- No `--project-id` needed — auto-discovered from ADC

## Ollama

- Model: `gemma4:31b` (used for Stage 2 ranking and gemma-analyze in quick-fix)
- Must be running before Stage 2 or `--gemma` flag: `brew services start ollama`
- Pull model: `ollama pull gemma4:31b`

## Quick-Fix Modes

`./run.sh fix <images> [flags]`

| Mode | Flags | Use case |
|------|-------|----------|
| Default | (none) | Auto-prompt from Vertex analysis |
| Gemma | `--gemma` | gemma4:31b analyzes image, writes prompt |
| Gemma + custom | `--gemma --prompt "text"` | Your prompt is priority, gemma's is secondary |
| Selective edit | `--gemma --prompt "text" --selective-edit` | Background-only inpainting, subject untouched |
| Ultra preserve | `--gemma --prompt "text" --ultra-preserve` | Museum-quality pixel-level subject preservation |

Common extra flags: `--overwrite`, `--image-size 2K`

## Key Files

- `main.py` — Stage 1 CLI entry point
- `pipeline.py` — Stage 1 curation logic
- `config.py` — Stage 1 thresholds
- `image_filters.py` — decode, blur, exposure, HEIC
- `stage2_scene_family_pipeline.py` — Stage 2 VLM ranking (Ollama)
- `stage2_gemini_model_suite.py` — Stage 2 via Gemini API (alternative)
- `run_album_pipeline.py` — Stage 1+2 combined launcher
- `vertex_ranked_photo_improver2.py` — Stage 3 Vertex improvement
- `vertex_quick_fix.py` — standalone single-image Vertex fix
- `run.sh` — wrapper that handles SSL/venv boilerplate

## Output Structure

```
album/
  Curated_Best_Stage2Ready/          # Stage 1 output
    Above_8MP/
    Between_1MP_and_8MP/
    selection_log.csv
  Stage2_FamilyReduced_gemma4_31b/   # Stage 2 intermediate
  Stage2_PortfolioTop_gemma4_31b/    # Stage 2 final
    ranked/
    portfolio_summary.json
  _vertex_fixed/                     # Quick-fix output (next to source images)
    meta/
```

## Conventions

- Never add Co-Authored-By lines in git commits
- Stage 2 output folder names encode the model slug (e.g., `gemma4_31b`)
- HEIC/HEIF supported via pillow-heif (venv only)
- For HEIC-heavy folders, use lower blur thresholds: `--hard-blur-threshold 5 --singleton-blur-threshold 10`
