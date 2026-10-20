"""Format-aware, read-only extraction with stable evidence coordinates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree as ET

from inventory import MEDIA_TYPES, sha256_file
from normalize import normalization_map, normalize_text, sha256_text, stable_id


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


class ExtractionFailure(RuntimeError):
    def __init__(self, status: str, warning: str):
        super().__init__(warning)
        self.status = status
        self.warning = warning


def _xml_text(element: ET.Element) -> str:
    pieces: list[str] = []
    for node in element.iter():
        if node.tag == f"{WORD_NS}t" and node.text:
            pieces.append(node.text)
        elif node.tag == f"{WORD_NS}tab":
            pieces.append("\t")
        elif node.tag in {f"{WORD_NS}br", f"{WORD_NS}cr"}:
            pieces.append("\n")
    return "".join(pieces)


def _block(kind: str, ordinal: int, raw_text: str, location: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_text(raw_text)
    return {
        "block_id": f"block_{ordinal:06d}",
        "kind": kind,
        "ordinal": ordinal,
        "location": location,
        "raw_text": raw_text,
        "normalized_text": normalized,
        "raw_sha256": sha256_text(raw_text),
        "normalized_sha256": sha256_text(normalized),
        "raw_global_start": 0,
        "raw_global_end": 0,
        "normalized_global_start": 0,
        "normalized_global_end": 0,
        "normalization_map": normalization_map(raw_text, normalized),
    }


def _finalize_blocks(blocks: list[dict[str, Any]]) -> tuple[str, str]:
    raw_parts: list[str] = []
    normalized_parts: list[str] = []
    raw_cursor = 0
    normalized_cursor = 0
    for index, block in enumerate(blocks):
        if index:
            raw_parts.append("\n\n")
            normalized_parts.append("\n\n")
            raw_cursor += 2
            normalized_cursor += 2
        block["raw_global_start"] = raw_cursor
        block["raw_global_end"] = raw_cursor + len(block["raw_text"])
        block["normalized_global_start"] = normalized_cursor
        block["normalized_global_end"] = normalized_cursor + len(block["normalized_text"])
        raw_parts.append(block["raw_text"])
        normalized_parts.append(block["normalized_text"])
        raw_cursor = block["raw_global_end"]
        normalized_cursor = block["normalized_global_end"]
    return "".join(raw_parts), "".join(normalized_parts)


def _extract_pdf(path: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[str], str]:
    try:
        from PyPDF2 import PdfReader
        from PyPDF2.errors import DependencyError, PdfReadError
    except ImportError as exc:
        raise ExtractionFailure("unreadable", "pypdf2_unavailable") from exc
    warnings: list[str] = []
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:
        raise ExtractionFailure("malformed", f"pdf_open_failed:{type(exc).__name__}") from exc
    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except DependencyError as exc:
            raise ExtractionFailure("encrypted", "pdf_encrypted_crypto_dependency_unavailable") from exc
        except Exception as exc:
            raise ExtractionFailure("encrypted", f"pdf_decryption_failed:{type(exc).__name__}") from exc
        if not decrypted:
            raise ExtractionFailure("encrypted", "pdf_encrypted_password_required")
        warnings.append("pdf_encrypted_empty_password_decrypted")
    blocks: list[dict[str, Any]] = []
    failed_pages = 0
    try:
        pages = list(reader.pages)
    except (PdfReadError, DependencyError, Exception) as exc:
        status = "encrypted" if reader.is_encrypted else "malformed"
        raise ExtractionFailure(status, f"pdf_pages_unreadable:{type(exc).__name__}") from exc
    for page_number, page in enumerate(pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            text = ""
            failed_pages += 1
            warnings.append(f"pdf_page_extract_failed:{page_number}:{type(exc).__name__}")
        blocks.append(_block("page", len(blocks) + 1, text, {"page": page_number}))
    counts = {"pages": len(pages), "paragraphs": 0, "tables": 0, "headers": 0, "footers": 0}
    method = "pypdf2_page_text"
    if failed_pages:
        warnings.append("pdf_partial_page_extraction")
    return blocks, counts, warnings, method


def _docx_blocks(path: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[str], str]:
    blocks: list[dict[str, Any]] = []
    counts = {"pages": 0, "paragraphs": 0, "tables": 0, "headers": 0, "footers": 0}
    warnings: list[str] = []
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ExtractionFailure("malformed", f"docx_open_failed:{type(exc).__name__}") from exc
    with archive:
        names = set(archive.namelist())
        if "word/document.xml" not in names:
            raise ExtractionFailure("malformed", "docx_missing_document_xml")

        def parse_part(part_name: str, part_kind: str) -> None:
            try:
                root = ET.fromstring(archive.read(part_name))
            except (ET.ParseError, KeyError) as exc:
                warnings.append(f"docx_part_parse_failed:{part_name}:{type(exc).__name__}")
                return
            if part_kind == "body":
                body = root.find(f"{WORD_NS}body")
                children = list(body) if body is not None else []
            else:
                children = list(root)
            paragraph_index = 0
            table_index = 0
            for child in children:
                if child.tag == f"{WORD_NS}p":
                    paragraph_index += 1
                    text = _xml_text(child)
                    if not text:
                        continue
                    kind = "paragraph" if part_kind == "body" else part_kind
                    counts["paragraphs" if kind == "paragraph" else f"{kind}s"] += 1
                    blocks.append(_block(kind, len(blocks) + 1, text, {"part": part_name, "paragraph": paragraph_index}))
                elif child.tag == f"{WORD_NS}tbl":
                    table_index += 1
                    counts["tables"] += 1
                    for row_index, row in enumerate(child.findall(f"{WORD_NS}tr"), 1):
                        for cell_index, cell in enumerate(row.findall(f"{WORD_NS}tc"), 1):
                            text = _xml_text(cell)
                            if not text:
                                continue
                            blocks.append(_block("table_cell", len(blocks) + 1, text, {"part": part_name, "table": table_index, "row": row_index, "cell": cell_index}))

        parse_part("word/document.xml", "body")
        for part_name in sorted(name for name in names if name.startswith("word/header") and name.endswith(".xml")):
            parse_part(part_name, "header")
        for part_name in sorted(name for name in names if name.startswith("word/footer") and name.endswith(".xml")):
            parse_part(part_name, "footer")
    return blocks, counts, warnings, "docx_ooxml_xml"


def _clean_word_text(value: str) -> str:
    return value.replace("\x07", "").replace("\x0b", "\n")


def _extract_doc_with_word(path: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[str], str]:
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise ExtractionFailure("unreadable", "word_automation_dependency_unavailable") from exc
    pythoncom.CoInitialize()
    word = None
    document = None
    blocks: list[dict[str, Any]] = []
    counts = {"pages": 0, "paragraphs": 0, "tables": 0, "headers": 0, "footers": 0}
    warnings: list[str] = []
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        word.AutomationSecurity = 3
        word.Options.UpdateLinksAtOpen = False
        document = word.Documents.Open(
            FileName=str(path.resolve()),
            ConfirmConversions=False,
            ReadOnly=True,
            AddToRecentFiles=False,
            Visible=False,
            OpenAndRepair=False,
            NoEncodingDialog=True,
        )
        counts["pages"] = int(document.ComputeStatistics(2))
        for index in range(1, document.Paragraphs.Count + 1):
            text = _clean_word_text(str(document.Paragraphs(index).Range.Text)).rstrip("\r")
            if not text:
                continue
            counts["paragraphs"] += 1
            blocks.append(_block("paragraph", len(blocks) + 1, text, {"paragraph": index}))
        for table_index in range(1, document.Tables.Count + 1):
            table = document.Tables(table_index)
            counts["tables"] += 1
            for row_index in range(1, table.Rows.Count + 1):
                for cell_index in range(1, table.Columns.Count + 1):
                    try:
                        text = _clean_word_text(str(table.Cell(row_index, cell_index).Range.Text)).rstrip("\r")
                    except Exception:
                        continue
                    if text:
                        blocks.append(_block("table_cell", len(blocks) + 1, text, {"table": table_index, "row": row_index, "cell": cell_index}))
        for section_index in range(1, document.Sections.Count + 1):
            section = document.Sections(section_index)
            for collection_name, kind in (("Headers", "header"), ("Footers", "footer")):
                collection = getattr(section, collection_name)
                for item_index in range(1, collection.Count + 1):
                    item = collection(item_index)
                    if not item.Exists:
                        continue
                    text = _clean_word_text(str(item.Range.Text)).rstrip("\r")
                    if text:
                        counts[f"{kind}s"] += 1
                        blocks.append(_block(kind, len(blocks) + 1, text, {"section": section_index, "variant": item_index}))
        warnings.append("word_paragraphs_may_include_table_text")
        document.Saved = True
    except Exception as exc:
        raise ExtractionFailure("unreadable", f"word_automation_failed:{type(exc).__name__}") from exc
    finally:
        if document is not None:
            try:
                document.Close(SaveChanges=0)
            except Exception:
                pass
        if word is not None:
            try:
                word.Quit(SaveChanges=0)
            except Exception:
                pass
        pythoncom.CoUninitialize()
    return blocks, counts, warnings, "microsoft_word_com_read_only"


def _extract_doc_with_libreoffice(path: Path, executable: str | None = None) -> tuple[list[dict[str, Any]], dict[str, int], list[str], str]:
    executable = executable or shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        raise ExtractionFailure("unreadable", "libreoffice_fallback_unavailable")
    with tempfile.TemporaryDirectory(prefix="hannibal_doc_extract_") as temp_dir:
        command = [executable, "--headless", "--safe-mode", "--convert-to", "docx", "--outdir", temp_dir, str(path.resolve())]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        converted = Path(temp_dir) / f"{path.stem}.docx"
        if result.returncode or not converted.exists():
            raise ExtractionFailure("unreadable", f"libreoffice_conversion_failed:{result.returncode}")
        blocks, counts, warnings, _ = _docx_blocks(converted)
        warnings.append("legacy_doc_converted_in_temporary_directory")
        return blocks, counts, warnings, "libreoffice_safe_mode_temporary_docx"


def extract_document(
    path: Path,
    manifest_record: dict[str, Any] | None = None,
    *,
    word_extractor: Callable[[Path], tuple[list[dict[str, Any]], dict[str, int], list[str], str]] | None = None,
    allow_libreoffice_fallback: bool = True,
) -> dict[str, Any]:
    path = path.resolve()
    extension = path.suffix.casefold()
    if extension not in MEDIA_TYPES:
        raise ValueError(f"Unsupported extension: {extension}")
    digest = sha256_file(path)
    source_id = (manifest_record or {}).get("source_id", stable_id("src", path.as_posix()))
    relative_path = (manifest_record or {}).get("relative_path", path.as_posix())
    warnings: list[str] = []
    try:
        if extension == ".pdf":
            blocks, counts, warnings, method = _extract_pdf(path)
        elif extension == ".docx":
            blocks, counts, warnings, method = _docx_blocks(path)
        else:
            try:
                blocks, counts, warnings, method = (word_extractor or _extract_doc_with_word)(path)
            except ExtractionFailure as word_error:
                if not allow_libreoffice_fallback:
                    raise
                try:
                    blocks, counts, fallback_warnings, method = _extract_doc_with_libreoffice(path)
                    warnings = [word_error.warning, *fallback_warnings]
                except ExtractionFailure as fallback_error:
                    raise ExtractionFailure("unreadable", f"{word_error.warning};{fallback_error.warning}") from fallback_error
        raw_text, normalized_text = _finalize_blocks(blocks)
        non_whitespace = sum(not char.isspace() for char in normalized_text)
        pages = counts.get("pages", 0)
        extraction_status = "success"
        if not normalized_text.strip():
            extraction_status = "empty"
            warnings.append("no_extracted_text")
        elif pages and non_whitespace / pages < 40:
            extraction_status = "likely_ocr_required"
            warnings.append("low_text_density_likely_ocr_required")
        elif any("failed" in warning for warning in warnings):
            extraction_status = "partial"
        status = "success" if extraction_status == "success" else "review_required"
    except ExtractionFailure as exc:
        blocks = []
        counts = {"pages": 0, "paragraphs": 0, "tables": 0, "headers": 0, "footers": 0}
        raw_text = normalized_text = ""
        extraction_status = exc.status
        status = "failed"
        method = "none"
        warnings = [exc.warning]
    return {
        "schema_version": "extraction-record-v1",
        "extraction_id": stable_id("ext", source_id, digest, "1.0.0"),
        "source": {
            "source_id": source_id,
            "relative_path": relative_path,
            "sha256": digest,
            "media_type": MEDIA_TYPES[extension],
            "size_bytes": path.stat().st_size,
        },
        "status": status,
        "extraction_status": extraction_status,
        "method": method,
        "warnings": sorted(set(warnings)),
        "counts": counts,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "raw_text_sha256": sha256_text(raw_text),
        "normalized_text_sha256": sha256_text(normalized_text),
        "blocks": blocks,
        "versions": {
            "pipeline": "1.0.0",
            "extractor": "1.0.0",
            "normalizer": "1.0.0",
            "schema": "extraction-record-v1",
        },
    }


def failed_extraction_record(
    path: Path,
    manifest_record: dict[str, Any] | None,
    *,
    extraction_status: str = "unreadable",
    warning: str,
) -> dict[str, Any]:
    """Build a schema-compatible failure when extraction is stopped externally."""
    path = path.resolve()
    extension = path.suffix.casefold()
    if extension not in MEDIA_TYPES:
        raise ValueError(f"Unsupported extension: {extension}")
    digest = sha256_file(path)
    source_id = (manifest_record or {}).get("source_id", stable_id("src", path.as_posix()))
    relative_path = (manifest_record or {}).get("relative_path", path.as_posix())
    return {
        "schema_version": "extraction-record-v1",
        "extraction_id": stable_id("ext", source_id, digest, "1.0.0"),
        "source": {
            "source_id": source_id,
            "relative_path": relative_path,
            "sha256": digest,
            "media_type": MEDIA_TYPES[extension],
            "size_bytes": path.stat().st_size,
        },
        "status": "failed",
        "extraction_status": extraction_status,
        "method": "none",
        "warnings": [warning],
        "counts": {"pages": 0, "paragraphs": 0, "tables": 0, "headers": 0, "footers": 0},
        "raw_text": "",
        "normalized_text": "",
        "raw_text_sha256": sha256_text(""),
        "normalized_text_sha256": sha256_text(""),
        "blocks": [],
        "versions": {
            "pipeline": "1.0.0",
            "extractor": "1.0.0",
            "normalizer": "1.0.0",
            "schema": "extraction-record-v1",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-record", type=Path)
    parser.add_argument("--no-libreoffice-fallback", action="store_true")
    args = parser.parse_args()
    manifest_record = (
        json.loads(args.manifest_record.read_text(encoding="utf-8"))
        if args.manifest_record
        else None
    )
    record = extract_document(
        args.path,
        manifest_record,
        allow_libreoffice_fallback=not args.no_libreoffice_fallback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "extraction_status": record["extraction_status"], "method": record["method"]}))
    return 0 if record["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
