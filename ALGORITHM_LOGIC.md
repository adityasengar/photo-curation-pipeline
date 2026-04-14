# Current Selection Logic

This document describes how the project currently works end to end.

It reflects the code in:
- [main.py](main.py)
- [config.py](config.py)
- [image_filters.py](image_filters.py)
- [pipeline.py](pipeline.py)
- [ollama_review_trial.py](ollama_review_trial.py)
- [gemma_rank_all_outputs.py](gemma_rank_all_outputs.py)

## 1. Entry Point

The CLI starts in [main.py](main.py).

The main relevant modes are:
- `--folder-id`
  For Google Drive mode.
- `--local-root`
  For local-folder mode. This is the mode used for your synced Google Drive folders.

Important CLI settings:
- `--min-mp`
  Default `8.0`
- `--medium-min-mp`
  Default `1.0`
- `--laplacian-threshold`
  Default `80.0`
  This is now a ranking reference, not the hard blur reject.
- `--similarity-threshold`
  Default `0.97`
- `--check-background-people`
  Optional face-based semantic check

The CLI builds a `PipelineConfig` and passes it to `CurationPipeline`.

## 2. Current Config Defaults

The most important current defaults in [config.py](config.py) are:

- `min_megapixels = 8.0`
- `medium_min_megapixels = 1.0`
- `hard_blur_threshold = 25.0`
- `laplacian_threshold = 80.0`
- `black_clip_pct = 15.0`
- `white_clip_pct = 15.0`
- `similarity_threshold = 0.97`
- `duplicate_color_threshold = 0.14`
- `duplicate_hash_threshold = 24`
- `require_sequence_proximity = True`
- `sequence_group_window = 20`
- `scene_family_window = 4`
- `scene_family_similarity_threshold = 0.93`
- `scene_family_color_threshold = 0.28`
- `scene_family_hash_threshold = 20`

Interpretation:
- below `1 MP` is rejected
- `1 MP` to `< 8 MP` goes to the medium bucket
- `>= 8 MP` goes to the high bucket
- blur below `25` is hard rejected
- blur above `25` remains eligible and is handled by ranking
- strict duplicate grouping stays narrow
- scene-family grouping is looser and exists for comparison metadata

## 3. Local Folder Flow

When `--local-root` is used, [pipeline.py](pipeline.py) runs `_run_local()`.

That flow is:

1. Discover local folders
2. Build candidate images
3. Group similar images
4. Pick one winner per group
5. Copy winners into output folders
6. Copy matching RAF files
7. Write `selection_log.csv`

## 4. Folder Discovery

`_discover_local_folders()` does this:

- if the given root itself contains supported images, it is curated
- each child subfolder is also curated if it contains supported images
- folders named the same as the output folder are skipped

Supported image suffixes are:
- `.jpg`
- `.jpeg`
- `.png`
- `.tif`
- `.tiff`
- `.bmp`
- `.webp`
- `.heic`
- `.heif`

RAF files are not processed as primary images yet.
They are only copied as companion RAW files when a matching JPG/PNG is selected.

## 5. Candidate Building

Each file is processed in `_build_local_candidates()`.

For every supported image file:

1. Create a `LocalImageDecision`
2. Decode the image
3. Read original dimensions
4. Compute megapixels
5. Assign a resolution tier
6. Apply OpenCV filters
7. Optionally run the face-based background checker
8. Optionally run CLIP
9. Compute a quality score
10. Build grouping fingerprints
11. Save all of this into:
   - a decision row for CSV
   - a candidate object for grouping and winner selection

## 6. Decoding

Decoding happens in [image_filters.py](image_filters.py).

Current behavior:
- JPG, JPEG, PNG, TIFF:
  decoded through Pillow first
- EXIF orientation is normalized with `ImageOps.exif_transpose(...)`
- fallback is OpenCV if Pillow fails
- HEIC / HEIF:
  decoded via `ffmpeg`
- HEIC dimensions:
  read via `sips`

So orientation is normalized before later scoring and grouping.

## 7. Resolution Filtering

Resolution tiering is handled by `_resolution_tier()` in [pipeline.py](pipeline.py).

Rules:
- `>= 8 MP` -> `above_8mp`
- `>= 1 MP and < 8 MP` -> `between_1mp_and_8mp`
- `< 1 MP` -> `below_1mp`

Effect:
- below `1 MP` is rejected immediately
- everything else stays in the pipeline
- final winners are copied into:
  - `Above_8MP`
  - `Between_1MP_and_8MP`

## 8. OpenCV Filters

`apply_filters()` in [image_filters.py](image_filters.py) computes:

- `laplacian_var`
  Sharpness proxy
- `black_clip_pct`
  Dark clipping proxy
- `white_clip_pct`
  Bright clipping proxy

Current hard reject rules:

### Blur
- reject if `laplacian_var < hard_blur_threshold`
- current default hard threshold: `25`

### Exposure
- reject if `black_clip_pct > black_clip_pct threshold`
- reject if `white_clip_pct > white_clip_pct threshold`

Current interpretation:
- only clearly blurred images are thrown out
- moderately soft images remain eligible

This was changed so that acceptable but not ultra-sharp travel shots are not discarded too aggressively.

## 9. Optional Scene Check

If `enable_background_people_check` is on, the face-based scene checker runs.

This is the experimental semantic checker that tries to detect:
- extra background faces
- dominant face size
- possible background-people distractions

It can:
- add metadata
- add a ranking penalty
- optionally reject an image if configured that way

In practice, this has been conservative and somewhat brittle.
It can misread statues or art as faces.

## 10. CLIP Scoring

If CLIP is available and the image survived the earlier checks, the pipeline runs CLIP aesthetic scoring.

CLIP output:
- `clip_score`
- `passed_clip`

Important detail:
- CLIP is currently not the main decision-maker
- it mostly acts as a bonus feature in the quality score
- on your travel albums, CLIP has generally been weak and low-range

## 11. Quality Score

`_quality_score()` in [pipeline.py](pipeline.py) computes:

`pass_bonus + sharpness_score + resolution_score + clip_bonus - exposure_penalty - scene_penalty`

Where:

- `pass_bonus`
  `1.0` if OpenCV filters passed
- `sharpness_score`
  based on `laplacian_var / laplacian_threshold`
  capped at `3.0`
- `resolution_score`
  based on `megapixels / min_megapixels`
  capped at `2.0`
- `clip_bonus`
  CLIP score if CLIP passed
- `exposure_penalty`
  penalty based on dark and bright clipping
- `scene_penalty`
  optional semantic penalty

Important detail:
- `laplacian_threshold = 80` is still used here as a ranking reference
- it is not the hard blur rejection threshold

## 12. How Grouping Works

Grouping is done by `_group_similar_images()` in [pipeline.py](pipeline.py).

This is the current grouping logic.

There are now two grouping layers:

1. `strict duplicate groups`
   Used for actual Stage 1 winner selection.
2. `scene families`
   Looser metadata-only groupings for “same moment / zoom variant / reframed variant” comparisons.

### 12.1 Sequence Compatibility

Before any visual comparison, two images must pass `_sequence_compatible()`.

That means:
- both filenames must end with a number
- the non-numeric prefix must match
- the numeric distance must be within `sequence_group_window`

Current default:
- `require_sequence_proximity = True`
- `sequence_group_window = 20`

So chronology does matter, but only as a gate.

It is not enough by itself to group images.

### 12.2 Visual Similarity Tests

For two images to group together, all of these must pass:

1. grayscale cosine similarity
   from `_image_vector()`
2. color thumbnail distance
   from `_color_thumb()`
3. perceptual hash distance
   from `_perceptual_hash()`

The exact checks are:

- `similarity >= similarity_threshold`
- `color_distance <= duplicate_color_threshold`
- `hash_distance <= duplicate_hash_threshold`

Current defaults:
- grayscale cosine similarity `>= 0.97`
- color distance `<= 0.14`
- pHash Hamming distance `<= 24`

These checks are intentionally conservative so only near-duplicates collapse together.

## 12.5 Scene Family Grouping

Scene families are built with the same general mechanics as strict groups:
- sequence proximity gate
- grayscale similarity
- color thumbnail distance
- perceptual hash distance

But the thresholds are looser:
- sequence window `<= 4`
- grayscale cosine similarity `>= 0.93`
- color distance `<= 0.28`
- pHash Hamming distance `<= 20`

Purpose:
- catch same-scene variants that are not true duplicates
- keep strict Stage 1 selection unchanged
- provide better structure for later Gemma comparison

Example:
- `DSCF6624.JPG` and `DSCF6625.JPG` do not meet the strict duplicate thresholds
- but they do meet the scene-family thresholds
- so they remain separate Stage 1 winners while still being marked as part of the same family

### 12.3 Union-Find Grouping

If two images match, they are unioned into the same connected component.

This means groups are built transitively:
- if A matches B
- and B matches C
- then all three become one group

This is what earlier caused some over-grouping problems.
The color and pHash guards reduced that problem significantly.

### 12.4 Important Consequence

If nearby frames do not pass those visual checks, they remain singleton groups.

This is exactly why images like:
- `DSCF7404.JPG`
- `DSCF7405.JPG`
- `DSCF7406.JPG`
- `DSCF7407.JPG`

were not compared against each other in Stage 1.

They were adjacent in time, but visually different enough that the current grouping logic left them as separate singleton groups.

## 13. Winner Selection Inside a Group

Once a group exists, `_run_local()` selects exactly one winner per group.

Eligible images are:
- images whose decision status is still one of:
  - `pending`
  - `eligible`
  - `selected`
- and whose `FilterResult.passed` is true

If no eligible images remain in a group:
- all images in that group are marked rejected
- reason becomes `no_eligible_image_in_group`

If eligible images exist:
- the winner is:
  `max(eligible, key=(passed_clip, quality_score, megapixels))`

So winner ranking order is:

1. `passed_clip`
2. `quality_score`
3. `megapixels`

Important note:
- this is only inside an already formed group
- singleton groups now face an extra check after winner selection

### 13.1 Singleton Guard

After the group winner is chosen, `_run_local()` applies extra rules if `group_size == 1`.

Current singleton defaults:
- `singleton_hard_blur_threshold = 50.0`
- `singleton_min_quality_score = 2.75`
- `singleton_group_penalty = 0.6`

Effect:
- reject singleton winners if `laplacian_var < 50`
- reject singleton winners if `quality_score < 2.75`
- if they survive both checks, keep them selected but subtract `0.6` from the stored ranking score

This means Stage 1 is now stricter on visually isolated images than on images that had to beat similar alternatives.

## 14. Status Values in the CSV

Each image gets a decision row in `selection_log.csv`.

Common statuses:
- `selected`
  the group winner
- `eligible`
  survived the hard checks but lost to another image in the same group
- `rejected`
  failed decode, resolution, blur, exposure, scene check, or group had no eligible winner

Important detail:
- singleton groups no longer get a free pass
- they can now be:
  - `selected` after the singleton guard
  - `rejected` with reasons like `singleton_blur (...)` or `singleton_quality (...)`

## 15. RAF Copying

If a selected file is:
- `.jpg`
- `.jpeg`
- `.png`

the pipeline looks for a same-stem:
- `.RAF`
- `.raf`

If found, it copies the RAW file into the same output bucket.

This is handled by `_copy_matching_raw()`.

## 16. Output Folders

For each curated local album folder, the pipeline creates:

- `<album>/Curated_Best/Above_8MP`
- `<album>/Curated_Best/Between_1MP_and_8MP`

or the equivalent folder name passed with `--local-output-name`.

Selected winners are copied there.

## 17. CSV Output

`_write_local_csv()` writes `selection_log.csv` in the local output folder.

Important CSV fields:
- `filename`
- `source_path`
- `resolution_tier`
- `status`
- `rejection_reason`
- `group_id`
- `group_size`
- `scene_family_id`
- `scene_family_size`
- `singleton_group`
- `singleton_penalty_applied`
- `selected_filename`
- `selected_tier`
- `selected_saved_path`
- `laplacian_var`
- `clip_score`
- `quality_score`
- `matching_raw_found`
- `copied_raw_path`

This CSV is the best source of truth for:
- why an image was kept or rejected
- what group it belonged to
- which image won that group

## 18. Stage 2 Gemma Logic

Gemma is not yet integrated directly into the main pipeline.

Instead, we built trial scripts:

- [ollama_review_trial.py](ollama_review_trial.py)
- [gemma_rank_all_outputs.py](gemma_rank_all_outputs.py)

### 18.1 `ollama_review_trial.py`

This sends batches of images to local Ollama models.

We tested:
- `moondream`
  too weak
- `gemma3:4b`
  much better

Modes:
- `describe`
- `compare`
- `portfolio`

The `portfolio` mode asks Gemma to decide:
- which photos are keepable
- which are printworthy

The current prompt is intentionally stricter than before:
- assume an image is not printworthy unless it is clearly stronger than average
- do not punish deliberate shallow depth of field
- do not punish a small subject in frame if the composition feels intentional
- reject ordinary, awkward, accidental, or flat-looking photos more aggressively

### 18.2 `gemma_rank_all_outputs.py`

This runs a hierarchical second-stage ranking:

1. take all Stage 1 curated JPGs
2. split into small batches
3. ask Gemma to score each batch
4. keep top candidates from each batch
5. rerank finalists
6. produce a top-N shortlist

The current final rerank prompt is also stricter than before:
- it treats `printworthy` as selective rather than generous
- it explicitly pushes down boring, awkward, soft, or accidentally bad family frames

This is how the stricter post-singleton `Top 30` shortlist was created.

## 19. Current Practical Weaknesses

These are the main known issues in the current logic.

### 19.1 Singleton Risk Is Reduced, Not Gone

The new singleton guard removes many weak one-off survivors.
In the Italy album test:
- old Stage 1 selected count: `248`
- singleton-strict Stage 1 selected count: `175`
- dropped previous singleton winners: `73`

It successfully removed previous problem cases like:
- `DSCF7406.JPG`
- `DSCF6697.JPG`
- `DSCF7413.JPG`

But some singleton images still remain by design, because some truly good photos are one-off shots rather than burst variants.

### 19.2 Visual Grouping Is Stronger Than Human-Moment Grouping

The current grouping is:
- sequence-compatible
- then visually similar

But it is not yet:
- burst-aware in a strong human sense
- or “same moment” aware beyond those thumbnail-like features

### 19.3 Gemma Is Better With A Stricter Prompt, But Still Imperfect

Gemma can surface strong candidates, but on larger batches it:
- becomes repetitive
- can still produce contradictory reasons across batch and final passes
- sometimes produces internally inconsistent summary lists

The stricter prompt improved things materially:
- obvious bad inclusions like `DSCF7406.JPG` no longer surfaced in the strict top 30
- weak singleton survivors we audited by eye also stayed out of the strict top 30

But Gemma still works best as:
- a selective second-stage reviewer
- not a completely trusted final authority

## 20. Best Current Mental Model

The current system is best understood as:

### Stage 1

Mechanical cleanup:
- decode
- size floor
- blur / exposure hard rejects
- duplicate-style grouping
- one winner per visual group

### Stage 2

Human-taste reranking:
- family photo quality
- scenic beauty
- printworthiness

That means:
- Stage 1 is a shortlist generator
- Stage 2 should be treated as a taste/review layer
- neither stage alone is perfect

## 21. What To Improve Next

If you continue iterating, the most useful next improvements are:

1. strengthen grouping for nearby filename bursts
   so obviously related frames get compared together
2. separate Stage 2 into:
   - `keepable`
   - `printworthy`
3. make Gemma compare smaller, more coherent groups
   before broad portfolio reranking
4. optionally add one more lightweight singleton filter
   for clearly accidental crowd/interior snapshots that still survive
5. separate:
   - `keepable`
   - `printworthy`

This is the current end-to-end algorithm as of now.
