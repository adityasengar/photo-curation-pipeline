#!/usr/bin/env python3
"""
Zero-Cost Local Photo Curation Agent
=====================================
Scans a Google Drive folder, applies resolution / blur / exposure / aesthetic
filters, and extracts only the images worth printing at A4/A3.

Usage:
    python main.py --folder-id <DRIVE_FOLDER_ID> [OPTIONS]

Run `python main.py --help` for the full option list.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import PipelineConfig
from pipeline import CurationPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate Google Drive photos for high-quality printing.",
    )
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--folder-id",
        help="Google Drive folder ID to scan.",
    )
    input_group.add_argument(
        "--local-root",
        help="Local folder to scan. If it contains subfolders, each subfolder is curated separately.",
    )
    p.add_argument(
        "--credentials", default="credentials.json",
        help="Path to OAuth 2.0 client-secret JSON (default: credentials.json).",
    )
    p.add_argument(
        "--min-mp", type=float, default=8.0,
        help="Minimum megapixels for A4 print suitability (default: 8.0).",
    )
    p.add_argument(
        "--medium-min-mp", type=float, default=1.0,
        help="Minimum megapixels for the lower review bucket in local mode (default: 1.0).",
    )
    p.add_argument(
        "--laplacian-threshold", type=float, default=80.0,
        help="Laplacian variance threshold for blur detection (default: 80.0).",
    )
    p.add_argument(
        "--black-clip", type=float, default=15.0,
        help="Max %% of pixels in darkest 5 bins (default: 15.0).",
    )
    p.add_argument(
        "--white-clip", type=float, default=15.0,
        help="Max %% of pixels in brightest 5 bins (default: 15.0).",
    )
    p.add_argument(
        "--clip-model", default="ViT-B-32",
        help="OpenCLIP model architecture (default: ViT-B-32).",
    )
    p.add_argument(
        "--clip-pretrained", default="openai",
        help="OpenCLIP pretrained weights tag (default: openai).",
    )
    p.add_argument(
        "--clip-threshold", type=float, default=0.55,
        help="Minimum CLIP positive-similarity score (default: 0.55).",
    )
    p.add_argument(
        "--output-dir", default="Curated_For_Print",
        help="Output directory for Drive mode (default: Curated_For_Print).",
    )
    p.add_argument(
        "--local-output-name", default="Curated_Best",
        help="Folder name created inside each local photo folder (default: Curated_Best).",
    )
    p.add_argument(
        "--similarity-threshold", type=float, default=0.97,
        help="Cosine similarity threshold for grouping similar images in local mode (default: 0.97).",
    )
    p.add_argument(
        "--check-background-people", action="store_true",
        help="Enable an optional face-based check that penalizes or rejects shots with likely background people.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig(
        folder_id=args.folder_id or "",
        local_root=Path(args.local_root).expanduser() if args.local_root else None,
        credentials_path=Path(args.credentials),
        min_megapixels=args.min_mp,
        medium_min_megapixels=args.medium_min_mp,
        laplacian_threshold=args.laplacian_threshold,
        black_clip_pct=args.black_clip,
        white_clip_pct=args.white_clip,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_score_threshold=args.clip_threshold,
        output_dir=Path(args.output_dir),
        csv_path=Path(args.output_dir) / "curation_log.csv",
        local_output_dirname=args.local_output_name,
        similarity_threshold=args.similarity_threshold,
        enable_background_people_check=args.check_background_people,
    )

    pipeline = CurationPipeline(cfg)
    stats = pipeline.run()
    print(stats.summary())

    sys.exit(0 if stats.records else 1)


if __name__ == "__main__":
    main()
