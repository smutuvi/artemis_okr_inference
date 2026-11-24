"""Split run JSONLs for GitHub vs home zip download.

GitHub rejects files >100MB. This tool:
  - lists / optionally reports JSONLs that are safe to ``git add -f``
  - zips only the oversized ones to ``~/`` for Jupyter download

Uses Python's stdlib zipfile (no system ``zip`` binary required).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

GITHUB_LIMIT_MB = 100
GITHUB_LIMIT_BYTES = GITHUB_LIMIT_MB * 1024 * 1024


def _iter_jsonl(run_dir: Path, *, include_checkpoints: bool, include_masks: bool) -> list[Path]:
    files: list[Path] = []
    if include_masks:
        files.extend(sorted(run_dir.glob("*_predictions_with_masks.jsonl")))
    if include_checkpoints:
        ckpt = run_dir / "checkpoints"
        if ckpt.is_dir():
            files.extend(sorted(ckpt.glob("*.jsonl")))
    return [p for p in files if p.is_file()]


def _arcname(run_dir: Path, src: Path) -> str:
    if src.parent.name == "checkpoints":
        return f"{run_dir.name}__checkpoints__{src.name}"
    return f"{run_dir.name}__{src.name}"


def _write_zip(out_zip: Path, run_dir: Path, files: list[Path]) -> Path:
    out_zip = Path(out_zip).expanduser().resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Packing {len(files)} oversized file(s) → {out_zip}")
    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED, allowZip64=True) as zf:
        for src in files:
            arcname = _arcname(run_dir, src)
            mb = src.stat().st_size / (1024 * 1024)
            print(f"  + {src.relative_to(run_dir)}  ({mb:.1f} MB) → {arcname}")
            zf.write(src, arcname=arcname)
    out_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_zip} ({out_mb:.1f} MB)")
    print("Download via Jupyter file browser or:")
    print(f'  from IPython.display import FileLink; FileLink("{out_zip}")')
    return out_zip


def pack_run_jsonl(
    run_dir: Path,
    *,
    out_zip: Path | None = None,
    include_checkpoints: bool = True,
    include_masks: bool = True,
    only_large: bool = False,
    github_split: bool = False,
) -> Path | None:
    """Zip JSONLs. With ``github_split``, only files >100MB go to the home zip."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    files = _iter_jsonl(
        run_dir,
        include_checkpoints=include_checkpoints,
        include_masks=include_masks,
    )
    if not files:
        raise FileNotFoundError(
            f"No JSONL files under {run_dir} "
            f"(checkpoints={include_checkpoints}, masks={include_masks})"
        )

    print("JSONL sizes:")
    for p in files:
        mb = p.stat().st_size / (1024 * 1024)
        flag = "ZIP→home" if p.stat().st_size > GITHUB_LIMIT_BYTES else "git-ok"
        print(f"  [{flag}] {p.relative_to(run_dir)}  {mb:.1f} MB")

    if github_split or only_large:
        large = [p for p in files if p.stat().st_size > GITHUB_LIMIT_BYTES]
        small = [p for p in files if p.stat().st_size <= GITHUB_LIMIT_BYTES]

        if small:
            print(f"\nGitHub-safe JSONLs (≤{GITHUB_LIMIT_MB} MB) — you can force-add:")
            for p in small:
                print(f"  git add -f {p}")
        else:
            print(f"\nNo JSONLs ≤{GITHUB_LIMIT_MB} MB.")

        if not large:
            print(f"No files >{GITHUB_LIMIT_MB} MB — nothing to zip to home.")
            return None

        if out_zip is None:
            out_zip = Path.home() / f"{run_dir.name}_jsonl_large.zip"
        return _write_zip(Path(out_zip), run_dir, large)

    # Default: zip everything requested
    if out_zip is None:
        out_zip = Path.home() / f"{run_dir.name}_jsonl.zip"
    return _write_zip(Path(out_zip), run_dir, files)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Zip run JSONLs for download. Use --github-split to zip only files "
            f">{GITHUB_LIMIT_MB} MB to ~/ and list smaller ones for git add."
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
        help="Output zip path (default: ~/<run>_jsonl.zip or _jsonl_large.zip)",
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
        help=f"Alias for --github-split (zip only files >{GITHUB_LIMIT_MB} MB)",
    )
    p.add_argument(
        "--github-split",
        action="store_true",
        help=(
            f"Zip only JSONLs >{GITHUB_LIMIT_MB} MB to home; print git add lines "
            f"for files ≤{GITHUB_LIMIT_MB} MB"
        ),
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
        github_split=args.github_split or args.only_large,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
