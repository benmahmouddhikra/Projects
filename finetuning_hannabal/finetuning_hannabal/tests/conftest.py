from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest


FEATURE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FEATURE_ROOT
SRC_ROOT = FEATURE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@pytest.fixture
def feature_root() -> Path:
    return FEATURE_ROOT


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


def write_minimal_docx(path: Path, text: str) -> None:
    escaped = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    paragraphs = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{line}</w:t></w:r></w:p>'
        for line in escaped.split("\n")
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{paragraphs}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '</Types>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("word/document.xml", document)


@pytest.fixture
def minimal_docx(tmp_path: Path) -> Path:
    path = tmp_path / "test-contract.docx"
    write_minimal_docx(path, "CONTRAT ÉCHANTILLON\nArticle 1 - Objet\nTexte arabe: العربية\nNom: [NOM_CLIENT]")
    return path


@pytest.fixture
def taxonomy(feature_root: Path) -> dict:
    return json.loads((feature_root / "config" / "taxonomy.json").read_text(encoding="utf-8"))
