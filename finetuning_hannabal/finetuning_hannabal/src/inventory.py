"""Read-only inventory and exact-duplicate detection for the source corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from normalize import stable_id


MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
EXPECTED_COUNTS = {".pdf": 955, ".doc": 297, ".docx": 24}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_posix(path: Path, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def inventory_corpus(templates_dir: Path, repository_root: Path | None = None) -> list[dict[str, Any]]:
    templates_dir = templates_dir.resolve()
    repository_root = (repository_root or templates_dir.parent).resolve()
    paths = sorted(
        (path for path in templates_dir.rglob("*") if path.is_file() and path.suffix.casefold() in MEDIA_TYPES),
        key=lambda path: relative_posix(path, repository_root).casefold(),
    )
    rows: list[dict[str, Any]] = []
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        relative_path = relative_posix(path, repository_root)
        digest = sha256_file(path)
        stat = path.stat()
        row = {
            "schema_version": "manifest-record-v1",
            "source_id": stable_id("src", relative_path),
            "relative_path": relative_path,
            "sha256": digest,
            "media_type": MEDIA_TYPES[path.suffix.casefold()],
            "extension": path.suffix.casefold()[1:],
            "size_bytes": stat.st_size,
            "folder_hint": path.relative_to(templates_dir).parts[0],
            "exact_duplicate": {
                "group_id": None,
                "is_primary": True,
                "member_paths": [relative_path],
            },
            "inventory_status": "inventoried",
            "pipeline_version": "1.0.0",
        }
        rows.append(row)
        by_hash[digest].append(row)
    for digest, members in by_hash.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda row: row["relative_path"].casefold())
        paths_in_group = [row["relative_path"] for row in members]
        group_id = stable_id("dup", digest)
        for index, row in enumerate(members):
            row["exact_duplicate"] = {
                "group_id": group_id,
                "is_primary": index == 0,
                "member_paths": paths_in_group,
            }
    return rows


def extension_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(f".{record['extension']}" for record in records)
    return dict(sorted(counts.items()))


def assert_expected_counts(records: Iterable[dict[str, Any]]) -> None:
    actual = extension_counts(records)
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"Corpus counts differ: expected {EXPECTED_COUNTS}, got {actual}")


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-counts", action="store_true")
    args = parser.parse_args()
    records = inventory_corpus(args.templates_dir)
    if args.strict_counts:
        assert_expected_counts(records)
    write_jsonl(records, args.output)
    print(json.dumps({"total": len(records), "counts": extension_counts(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
