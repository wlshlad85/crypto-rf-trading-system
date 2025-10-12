#!/usr/bin/env python3
"""Utility to collect data and result artifacts into a consolidated archive."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

# Default directories to gather artifacts from
DEFAULT_SOURCE_DIRECTORIES: Sequence[Path] = (
    Path("data/4h_training"),
    Path("phase1/data"),
    Path("phase1/cpcv_results"),
    Path("phase1/walkforward_results"),
    Path("phase2"),
    Path("phase2b"),
    Path("analysis"),
    Path("results"),
    Path("logs/24hr_trading"),
    Path("logs/enhanced_24hr_trading"),
    Path("logs/minute_backtest"),
)

# Patterns to scan from the repository root
DEFAULT_ROOT_PATTERNS: Sequence[str] = (
    "*results*.json",
    "btc_*session*.json",
)

# File extensions considered data artifacts
DEFAULT_EXTENSIONS: Sequence[str] = (
    ".csv",
    ".json",
    ".log",
    ".txt",
    ".html",
    ".pdf",
    ".png",
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ArtifactRecord:
    source: Path
    destination: Path
    size_bytes: int
    modified_time: str

    def to_dict(self) -> dict:
        return {
            "source": str(self.source),
            "destination": str(self.destination),
            "size_bytes": self.size_bytes,
            "modified_time": self.modified_time,
        }


def iter_artifact_files(
    directories: Sequence[Path],
    extensions: Sequence[str],
    root_patterns: Sequence[str],
) -> Iterator[tuple[Path, Path]]:
    """Yield tuples of (source_root, file_path) for matching artifacts."""
    normalized_exts = {ext.lower() for ext in extensions}

    for src_root in directories:
        root_path = (REPO_ROOT / src_root).resolve()
        if not root_path.exists():
            logging.debug("Skipping missing source directory: %s", root_path)
            continue
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in normalized_exts:
                yield root_path, file_path

    # Patterns searched from repository root
    for pattern in root_patterns:
        for file_path in REPO_ROOT.glob(pattern):
            if file_path.is_file():
                yield REPO_ROOT, file_path


def copy_artifacts(
    files: Iterable[tuple[Path, Path]],
    destination: Path,
) -> List[ArtifactRecord]:
    """Copy artifacts to destination and return manifest records."""
    manifest: List[ArtifactRecord] = []
    destination.mkdir(parents=True, exist_ok=True)
    seen_sources = set()

    for source_root, file_path in files:
        resolved_path = file_path.resolve()
        if resolved_path in seen_sources:
            logging.debug("Skipping duplicate artifact %s", file_path)
            continue
        seen_sources.add(resolved_path)

        relative_path = file_path.relative_to(source_root)
        dest_path = destination / source_root.relative_to(REPO_ROOT) / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)

        stat = dest_path.stat()
        manifest.append(
            ArtifactRecord(
                source=file_path.relative_to(REPO_ROOT),
                destination=dest_path.relative_to(destination),
                size_bytes=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            )
        )
        logging.debug("Copied %s -> %s", file_path, dest_path)

    return manifest


def write_manifest(destination: Path, records: Sequence[ArtifactRecord]) -> Path:
    manifest_path = destination / "manifest.json"
    payload = [record.to_dict() for record in records]
    manifest_path.write_text(json.dumps(payload, indent=2))
    logging.info("Wrote manifest with %d records to %s", len(records), manifest_path)
    return manifest_path


def compress_archive(destination: Path, archive_format: str) -> Path:
    archive_base = destination.with_suffix("")
    archive_path = shutil.make_archive(str(archive_base), archive_format, root_dir=destination)
    logging.info("Created %s archive at %s", archive_format, archive_path)
    return Path(archive_path)


def clean_destination(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        logging.info("Removed existing destination %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("artifacts/data_archive"),
        help="Destination directory for collected artifacts.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the destination before copying artifacts.",
    )
    parser.add_argument(
        "--compress",
        choices=["zip", "gztar", "tar"],
        help="Optionally compress the resulting archive using the selected format.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=list(DEFAULT_EXTENSIONS),
        help="Override the list of file extensions to collect.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    destination: Path = (REPO_ROOT / args.dest).resolve()
    if args.clean:
        clean_destination(destination)

    files = list(
        iter_artifact_files(
            directories=DEFAULT_SOURCE_DIRECTORIES,
            extensions=args.extensions,
            root_patterns=DEFAULT_ROOT_PATTERNS,
        )
    )

    if not files:
        logging.warning("No artifacts found with the current configuration.")

    records = copy_artifacts(files, destination)
    manifest_path = write_manifest(destination, records)

    if args.compress:
        compress_archive(destination, args.compress)
        logging.info(
            "Compression complete. Archive and manifest available in %s",
            destination.parent,
        )
    else:
        logging.info("Artifacts collected at %s", destination)
        logging.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
