"""Zip large run JSONLs for download (GitHub rejects files >100MB).

Uses Python's stdlib zipfile (no system ``zip`` binary required).
Writes to ``~/`` by default so Jupyter ``FileLink`` / file browser can download.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# GitHub hard limit is 100MB; pack anything near that for off-git transfer.
GITHUB_SAFE_MB = 90


def _iter_jsonl(run_dir: Path, *, include_checkpoints: bool, include_masks: bool) -> list[Path]:
    files: list[Path] = []
    if include_masks:
        files.extend(sorted(run_dir.glob("*_predictions_with_masks.jsonl")))
    if include_checkpoints:
        ckpt = run_dir / "checkpoints"
        if ckpt.is_dir():
            files.extend(sorted(ckpt.glob("*.jsonl")))
    return [p for p in files if p.is_file()]


def pack_run_jsonl(
    run_dir: Path,
    *,
    out_zip: Path | None = None,
    include_checkpoints: bool = True,
    include_masks: bool = True,
    only_large: bool = False,
) -> Path:
    """Create a zip of mask/checkpoint JSONLs with unique archive names."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    files = _iter_jsonl(
        run_dir,
        include_checkpoints=include_checkpoints,
        include_masks=include_masks,
    )
    if files:
        print("JSONL sizes:")
        for p in files:
            print(f"  {p.relative_to(run_dir)}  {p.stat().st_size / (1024*1024):.1f} MB")

    if only_large:
        lim = GITHUB_SAFE_MB * 1024 * 1024
        large = [p for p in files if p.stat().st_size >= lim]
        if not large and files:
            print(
                f"No files ≥ {GITHUB_SAFE_MB} MB; packing all {len(files)} JSONL(s) instead "
                f"(omit --only-large to silence this fallback)."
            )
        else:
            files = large

    if not files:
        raise FileNotFoundError(
            f"No JSONL files to pack under {run_dir} "
            f"(checkpoints={include_checkpoints}, masks={include_masks}, only_large={only_large})"
        )

    if out_zip is None:
        out_zip = Path.home() / f"{run_dir.name}_jsonl.zip"
    else:
        out_zip = Path(out_zip).expanduser().resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    print(f"Packing {len(files)} file(s) → {out_zip}")
    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED, allowZip64=True) as zf:
        for src in files:
            # Unique names so pods_predictions_with_masks.jsonl from different
            # runs/tasks never collide inside the archive.
            if src.parent.name == "checkpoints":
                arcname = f"{run_dir.name}__checkpoints__{src.name}"
            else:
                arcname = f"{run_dir.name}__{src.name}"
            mb = src.stat().st_size / (1024 * 1024)
            print(f"  + {src.relative_to(run_dir)}  ({mb:.1f} MB) → {arcname}")
            zf.write(src, arcname=arcname)

    out_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_zip} ({out_mb:.1f} MB)")
    print("Download via Jupyter file browser or:")
    print(f'  from IPython.display import FileLink; FileLink("{out_zip}")')
    return out_zip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Zip checkpoint / mask JSONLs from a run folder for off-GitHub download "
            "(avoids the 100MB GitHub file limit)."
        )
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run folder, e.g. outputs/20261124_okr_4_3",
    )
    p.add_argument(
        "--out-zip",
        type=Path,
        default=None,
        help="Output zip path (default: ~/<run_dir_name>_jsonl.zip)",
    )
    p.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Omit checkpoints/*.jsonl",
    )
    p.add_argument(
        "--no-masks",
        action="store_true",
        help="Omit *_predictions_with_masks.jsonl",
    )
    p.add_argument(
        "--only-large",
        action="store_true",
        help=f"Only include files ≥ {GITHUB_SAFE_MB} MB",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pack_run_jsonl(
        args.run_dir,
        out_zip=args.out_zip,
        include_checkpoints=not args.no_checkpoints,
        include_masks=not args.no_masks,
        only_large=args.only_large,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
