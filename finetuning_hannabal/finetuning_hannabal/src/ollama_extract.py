"""Sequential Ollama structured-output boundary for coordinate-only labeling."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from normalize import sha256_text


_REQUEST_LOCK = threading.Lock()


@dataclass(frozen=True)
class OllamaConfig:
    model_tag: str = "qwen3.5:4b-q4_K_M"
    num_ctx: int = 4096
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 300
    malformed_retries: int = 1


def load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def installed_models(base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 5) -> list[str]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []
    return sorted({item.get("name") for item in payload.get("models", []) if item.get("name")})


def model_available(model_tag: str, base_url: str = "http://127.0.0.1:11434") -> bool:
    models = installed_models(base_url)
    return model_tag in models or any(name.split(":", 1)[0] == model_tag for name in models)


def source_blocks_payload(extraction_record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "block_id": block["block_id"],
            "kind": block["kind"],
            "location": block["location"],
            "normalized_text": block["normalized_text"],
            "normalized_sha256": block["normalized_sha256"],
        }
        for block in extraction_record["blocks"]
    ]


def validate_label_spans(response: dict[str, Any], extraction_record: dict[str, Any]) -> list[str]:
    blocks = {block["block_id"]: block for block in extraction_record["blocks"]}
    errors: list[str] = []

    def visit(value: Any, path: str = "$") -> None:
        if isinstance(value, dict):
            if set(value) == {"block_id", "start", "end", "text_sha256"}:
                block = blocks.get(value["block_id"])
                if block is None:
                    errors.append(f"{path}: unknown block_id")
                    return
                start, end = value["start"], value["end"]
                text = block["normalized_text"]
                if start < 0 or end <= start or end > len(text):
                    errors.append(f"{path}: invalid offsets")
                elif sha256_text(text[start:end]) != value["text_sha256"]:
                    errors.append(f"{path}: text hash mismatch")
                return
            for key, item in value.items():
                visit(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    visit(response)
    return errors


class OllamaStructuredLabeler:
    def __init__(self, schema_path: Path, prompt_path: Path, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self.schema = load_schema(schema_path)
        self.validator = Draft202012Validator(self.schema)
        self.prompt = prompt_path.read_text(encoding="utf-8")
        self.prompt_version = sha256_text(self.prompt)[:16]

    def label(self, extraction_record: dict[str, Any], taxonomy: dict[str, Any]) -> dict[str, Any]:
        if not model_available(self.config.model_tag, self.config.base_url):
            raise RuntimeError(f"Ollama model is not installed or reachable: {self.config.model_tag}")
        user_payload = {
            "source_id": extraction_record["source"]["source_id"],
            "taxonomy": taxonomy,
            "blocks": source_blocks_payload(extraction_record),
        }
        full_prompt = self.prompt + "\nINPUT_JSON:\n" + json.dumps(user_payload, ensure_ascii=False, sort_keys=True)
        body = {
            "model": self.config.model_tag,
            "prompt": full_prompt,
            "stream": False,
            "format": self.schema,
            "options": {"temperature": 0, "num_ctx": self.config.num_ctx},
        }
        malformed_attempts = 0
        while True:
            try:
                with _REQUEST_LOCK:
                    request = urllib.request.Request(
                        f"{self.config.base_url.rstrip('/')}/api/generate",
                        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as http_response:
                        envelope = json.load(http_response)
                response = json.loads(envelope["response"])
                schema_errors = sorted(self.validator.iter_errors(response), key=lambda error: list(error.path))
                if schema_errors:
                    raise ValueError("; ".join(error.message for error in schema_errors))
                span_errors = validate_label_spans(response, extraction_record)
                if span_errors:
                    raise RuntimeError("Rejected ungrounded model response: " + "; ".join(span_errors))
                return response
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                if malformed_attempts >= self.config.malformed_retries:
                    raise RuntimeError(f"Malformed structured output after retry: {exc}") from exc
                malformed_attempts += 1
                body["prompt"] = full_prompt + "\nYour prior output was malformed. Return only schema-valid JSON."
            except (urllib.error.URLError, TimeoutError, RuntimeError):
                raise

