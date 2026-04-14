#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
STAGE2_SCRIPT = PROJECT_ROOT / "stage2_scene_family_pipeline.py"


def run_command(command: list[str], *, cwd: Path) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n[run] (cwd={cwd}) {printable}")
    subprocess.run(command, cwd=str(cwd), check=True)


def build_stage2_command(
    *,
    model: str,
    family_divisor: int,
    portfolio_divisor: int,
    batch_size: int,
    timeout: int,
    stage1_dir: Path | None = None,
    resume_family_dir: Path | None = None,
) -> list[str]:
    command = [
        "python3",
        str(STAGE2_SCRIPT),
    ]
    if stage1_dir is not None:
        command.extend(["--stage1-dir", str(stage1_dir)])
    if resume_family_dir is not None:
        command.extend(["--resume-family-dir", str(resume_family_dir)])
    command.extend(
        [
            "--model",
            model,
            "--family-divisor",
            str(family_divisor),
            "--portfolio-divisor",
            str(portfolio_divisor),
            "--batch-size",
            str(batch_size),
            "--timeout",
            str(timeout),
        ]
    )
    return command


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage 1 + Stage 2 for a travel album with stable default output names. "
            "Run this from inside the target album folder, or pass --album-dir."
        )
    )
    parser.add_argument(
        "--album-dir",
        default=".",
        help="Album directory to process (default: current directory).",
    )
    parser.add_argument(
        "--stage1-output-name",
        default="Curated_Best_Stage2Ready",
        help="Stage 1 output folder name created inside the album (default: Curated_Best_Stage2Ready).",
    )
    parser.add_argument(
        "--model",
        default="gemma4:31b",
        help="Stage 2 model (default: gemma4:31b).",
    )
    parser.add_argument(
        "--family-divisor",
        type=int,
        default=4,
        help="Stage 2 family reduction divisor (default: 4).",
    )
    parser.add_argument(
        "--portfolio-divisor",
        type=int,
        default=4,
        help="Stage 2 portfolio reduction divisor (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Stage 2 batch size (default: 12).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-model-call timeout in seconds for Stage 2 (default: 3600).",
    )
    parser.add_argument(
        "--reuse-stage1",
        action="store_true",
        help="Skip Stage 1 if the Stage 1 output folder already exists.",
    )
    parser.add_argument(
        "--force-family-rerun",
        action="store_true",
        help="Force Stage 2 to recompute family reduction even if family outputs already exist.",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Run only Stage 1.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging in Stage 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    album_dir = Path(args.album_dir).expanduser().resolve()
    if not album_dir.exists() or not album_dir.is_dir():
        raise FileNotFoundError(f"Album directory not found: {album_dir}")

    stage1_dir = album_dir / args.stage1_output_name
    slug = model_slug(args.model)
    stage2_family_dir = album_dir / f"Stage2_FamilyReduced_{slug}"
    stage2_portfolio_dir = album_dir / f"Stage2_PortfolioTop_{slug}"

    should_run_stage1 = True
    if args.reuse_stage1 and stage1_dir.exists():
        should_run_stage1 = False
        print(f"[info] Reusing existing Stage 1 output: {stage1_dir}")

    if should_run_stage1:
        stage1_cmd = [
            "python3",
            str(MAIN_SCRIPT),
            "--local-root",
            str(album_dir),
            "--local-output-name",
            args.stage1_output_name,
        ]
        if args.verbose:
            stage1_cmd.append("--verbose")
        run_command(stage1_cmd, cwd=album_dir)
    else:
        if not stage1_dir.exists():
            raise FileNotFoundError(
                f"Stage 1 output not found at {stage1_dir}. Remove --reuse-stage1 or run Stage 1 first."
            )

    if not args.skip_stage2:
        existing_family_csv = stage2_family_dir / "family_reduction.csv"
        should_resume_family = (
            args.reuse_stage1
            and not args.force_family_rerun
            and existing_family_csv.exists()
        )
        if should_resume_family:
            print(f"[info] Reusing existing family reduction: {stage2_family_dir}")
            first_stage2_cmd = build_stage2_command(
                resume_family_dir=stage2_family_dir,
                model=args.model,
                family_divisor=args.family_divisor,
                portfolio_divisor=args.portfolio_divisor,
                batch_size=args.batch_size,
                timeout=args.timeout,
            )
        else:
            first_stage2_cmd = build_stage2_command(
                stage1_dir=stage1_dir,
                model=args.model,
                family_divisor=args.family_divisor,
                portfolio_divisor=args.portfolio_divisor,
                batch_size=args.batch_size,
                timeout=args.timeout,
            )
        try:
            run_command(first_stage2_cmd, cwd=album_dir)
        except subprocess.CalledProcessError as first_error:
            family_csv = stage2_family_dir / "family_reduction.csv"
            if not family_csv.exists():
                raise

            print(
                "[warn] Stage 2 failed after/around family reduction. "
                "Retrying portfolio phase from family outputs with smaller batch sizes."
            )
            retry_sizes = []
            for size in (6, 4):
                if size < args.batch_size:
                    retry_sizes.append(size)
            if not retry_sizes:
                retry_sizes = [max(2, args.batch_size // 2)]

            recovered = False
            last_retry_error: subprocess.CalledProcessError | None = None
            for retry_batch in retry_sizes:
                retry_cmd = build_stage2_command(
                    resume_family_dir=stage2_family_dir,
                    model=args.model,
                    family_divisor=args.family_divisor,
                    portfolio_divisor=args.portfolio_divisor,
                    batch_size=retry_batch,
                    timeout=args.timeout,
                )
                try:
                    run_command(retry_cmd, cwd=album_dir)
                    recovered = True
                    print(f"[info] Stage 2 recovered with batch-size={retry_batch}")
                    break
                except subprocess.CalledProcessError as retry_error:
                    last_retry_error = retry_error
                    print(f"[warn] Retry failed with batch-size={retry_batch}")

            if not recovered:
                if last_retry_error is not None:
                    raise last_retry_error
                raise first_error

    summary = {
        "album_dir": str(album_dir),
        "stage1_output_dir": str(stage1_dir),
        "stage2_family_output_dir": str(stage2_family_dir),
        "stage2_portfolio_output_dir": str(stage2_portfolio_dir),
        "model": args.model,
    }
    print("\n[done] Pipeline summary")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
