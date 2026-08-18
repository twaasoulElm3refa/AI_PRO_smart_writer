from __future__ import annotations

import json
import mimetypes
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.settings import get_settings


@dataclass(frozen=True)
class StoredMediaFile:
    file_id: str
    path: Path
    filename: str
    content_type: str
    size_bytes: int
    metadata: dict[str, Any]


def _media_root() -> Path:
    settings = get_settings()
    root = Path(settings.GENERATED_MEDIA_FILES_DIR).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _validate_file_id(file_id: str) -> str:
    try:
        return str(uuid.UUID(str(file_id)))
    except Exception as exc:
        raise ValueError("Invalid generated file id") from exc


def _media_category(content_type: str, extension: str) -> str:
    """Return the storage folder for a generated media file."""
    normalized_type = (content_type or "").strip().lower()
    normalized_extension = (extension or "").strip().lower()

    if normalized_type.startswith("image/"):
        return "images"
    if normalized_type.startswith("audio/") or normalized_extension in {
        ".mp3", ".wav", ".pcm", ".opus", ".ogg", ".aac", ".flac", ".m4a",
    }:
        return "audio"
    if normalized_type.startswith("video/"):
        return "video"
    return "files"


def safe_download_name(filename: str, fallback: str = "generated-file") -> str:
    filename = Path(filename or "").name.strip()
    if not filename:
        filename = fallback
    filename = re.sub(r"[^A-Za-z0-9._()\- ]+", "_", filename)
    filename = re.sub(r"\s+", " ", filename).strip(" .")
    return filename[:180] or fallback


def save_media_bytes(
    data: bytes,
    *,
    extension: str,
    filename: str,
    content_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StoredMediaFile:
    if not data:
        raise ValueError("Generated file is empty")

    extension = extension.lower().strip()
    if not extension.startswith("."):
        extension = f".{extension}"
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
        raise ValueError("Invalid generated file extension")

    file_id = str(uuid.uuid4())
    final_filename = safe_download_name(filename, fallback=f"generated{extension}")
    if not Path(final_filename).suffix:
        final_filename = f"{final_filename}{extension}"

    final_content_type = content_type or mimetypes.guess_type(final_filename)[0] or "application/octet-stream"
    final_metadata = dict(metadata or {})

    root = _media_root()
    category = _media_category(final_content_type, extension)
    media_dir = root / category
    metadata_dir = root / "metadata"
    media_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = media_dir / f"{file_id}{extension}"
    meta_path = metadata_dir / f"{file_id}.json"

    path.write_bytes(data)
    payload = {
        "file_id": file_id,
        "filename": final_filename,
        "content_type": final_content_type,
        "size_bytes": len(data),
        "extension": extension,
        "category": category,
        "metadata": final_metadata,
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return StoredMediaFile(
        file_id=file_id,
        path=path,
        filename=final_filename,
        content_type=final_content_type,
        size_bytes=len(data),
        metadata=final_metadata,
    )


def get_media_file(file_id: str) -> StoredMediaFile:
    file_id = _validate_file_id(file_id)
    root = _media_root()
    # New files keep sidecar metadata separate from the generated media. Fall
    # back to the old root-level layout so previously generated URLs still work.
    meta_path = root / "metadata" / f"{file_id}.json"
    legacy_layout = False
    if not meta_path.exists():
        meta_path = root / f"{file_id}.json"
        legacy_layout = True
    if not meta_path.exists():
        raise FileNotFoundError("Generated media file metadata was not found")

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("Generated media file metadata is invalid") from exc

    extension = str(payload.get("extension") or "").lower()
    if legacy_layout:
        path = root / f"{file_id}{extension}"
    else:
        category = str(payload.get("category") or "").strip().lower()
        if category not in {"images", "audio", "video", "files"}:
            raise ValueError("Generated media file category is invalid")
        path = root / category / f"{file_id}{extension}"
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("Generated media file was not found")

    return StoredMediaFile(
        file_id=file_id,
        path=path,
        filename=safe_download_name(str(payload.get("filename") or path.name), fallback=path.name),
        content_type=str(payload.get("content_type") or "application/octet-stream"),
        size_bytes=int(payload.get("size_bytes") or path.stat().st_size),
        metadata=dict(payload.get("metadata") or {}),
    )


def media_download_url(file_id: str) -> str:
    return f"/tasks/generated-files/download/{file_id}"
