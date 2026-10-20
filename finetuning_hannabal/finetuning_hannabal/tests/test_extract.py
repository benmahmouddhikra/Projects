from __future__ import annotations

import hashlib
from pathlib import Path

from PyPDF2 import PdfWriter

from extract import _block, extract_document
from inventory import sha256_file


def test_pdf_doc_and_docx_routing(tmp_path, minimal_docx):
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as stream:
        writer.write(stream)
    pdf = extract_document(pdf_path)
    assert pdf["method"] == "pypdf2_page_text"
    assert pdf["counts"]["pages"] == 1

    docx = extract_document(minimal_docx)
    assert docx["method"] == "docx_ooxml_xml"
    assert docx["counts"]["paragraphs"] == 4

    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes(b"test fixture")

    def fake_word(path):
        return [_block("paragraph", 1, "Legacy text", {"paragraph": 1})], {"pages": 1, "paragraphs": 1, "tables": 0, "headers": 0, "footers": 0}, [], "microsoft_word_com_read_only"

    doc = extract_document(doc_path, word_extractor=fake_word, allow_libreoffice_fallback=False)
    assert doc["method"] == "microsoft_word_com_read_only"
    assert doc["normalized_text"] == "Legacy text"


def test_encrypted_and_malformed_pdf_are_routed_to_review(tmp_path):
    encrypted_path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    with encrypted_path.open("wb") as stream:
        writer.write(stream)
    encrypted = extract_document(encrypted_path)
    assert encrypted["status"] == "failed"
    assert encrypted["extraction_status"] == "encrypted"

    malformed_path = tmp_path / "malformed.pdf"
    malformed_path.write_bytes(b"not a pdf")
    malformed = extract_document(malformed_path)
    assert malformed["status"] == "failed"
    assert malformed["extraction_status"] == "malformed"


def test_unicode_diacritics_and_directionality_round_trip(minimal_docx):
    record = extract_document(minimal_docx)
    assert "ÉCHANTILLON" in record["normalized_text"]
    assert "العربية" in record["normalized_text"]
    assert "[NOM_CLIENT]" in record["normalized_text"]


def test_read_only_extraction_does_not_change_source(minimal_docx):
    before = sha256_file(minimal_docx)
    extract_document(minimal_docx)
    after = sha256_file(minimal_docx)
    assert before == after

