"""Ingest documents into the vector store.

Usage:
    python scripts/ingest.py                  # ingest data/sample_docs
    python scripts/ingest.py --dir path/to/dir
    python scripts/ingest.py --file a.pdf --file b.md
    python scripts/ingest.py --reset          # wipe the collection first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import build_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest docs into the RAG store")
    parser.add_argument("--dir", action="append", help="Directory to ingest")
    parser.add_argument("--file", action="append", help="Single file to ingest")
    parser.add_argument(
        "--reset", action="store_true", help="Drop the collection before ingesting"
    )
    args = parser.parse_args()

    pipeline = build_pipeline(get_settings())

    if args.reset:
        pipeline.reset()
        print("Collection reset.")

    targets_dir = args.dir or ([] if args.file else [get_settings().data_dir])
    total_chunks = 0

    for d in targets_dir:
        report = pipeline.ingest_directory(d)
        print(
            f"{d}: {report.files_ingested}/{report.files_seen} files, "
            f"{report.chunks_added} chunks"
        )
        for skipped in report.skipped:
            print(f"  skipped: {skipped}")
        total_chunks += report.chunks_added

    for f in args.file or []:
        added = pipeline.ingest_file(f)
        print(f"{f}: {added} chunks")
        total_chunks += added

    print(f"\nDone. {total_chunks} chunks added, {pipeline.stats()['chunks']} total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
