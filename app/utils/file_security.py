"""File security utilities for validating uploads."""

import re
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Allowed MIME types and their expected file extensions
ALLOWED_TYPES = {
    "application/pdf": [".pdf"],
    "text/plain": [".txt"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
        ".docx"
    ],
}

# Maximum filename length
MAX_FILENAME_LENGTH = 255


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and other security issues.

    Args:
        filename: Original filename from upload

    Returns:
        Safe filename with dangerous characters removed

    Raises:
        ValueError: If filename is empty or too long
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Get just the filename, no directory components
    safe_name = Path(filename).name

    # Remove any remaining path separators
    safe_name = safe_name.replace("/", "_").replace("\\", "_")

    # Remove dangerous characters, keep alphanumeric, dots, dashes, underscores
    safe_name = re.sub(r'[^\w\s.-]', '_', safe_name)

    # Remove multiple dots (except for file extension)
    parts = safe_name.rsplit(".", 1)
    if len(parts) == 2:
        name_part, ext_part = parts
        name_part = name_part.replace(".", "_")
        safe_name = f"{name_part}.{ext_part}"

    # Limit length
    if len(safe_name) > MAX_FILENAME_LENGTH:
        # Keep extension, truncate name
        name_parts = safe_name.rsplit(".", 1)
        if len(name_parts) == 2:
            name, ext = name_parts
            name = name[: MAX_FILENAME_LENGTH - len(ext) - 1]
            safe_name = f"{name}.{ext}"
        else:
            safe_name = safe_name[:MAX_FILENAME_LENGTH]

    # Ensure not empty after sanitization
    if not safe_name or safe_name == ".":
        raise ValueError("Filename invalid after sanitization")

    logger.debug(f"Sanitized filename: {filename} -> {safe_name}")
    return safe_name


def validate_file_extension(
    filename: str, content_type: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate that file extension matches the declared content type.

    Args:
        filename: Filename with extension
        content_type: Declared MIME type

    Returns:
        Tuple of (is_valid, error_message)
    """
    if content_type not in ALLOWED_TYPES:
        return False, f"Content type '{content_type}' not allowed"

    file_ext = Path(filename).suffix.lower()
    allowed_exts = ALLOWED_TYPES[content_type]

    if file_ext not in allowed_exts:
        return (
            False,
            f"File extension '{file_ext}' doesn't match content type '{content_type}'",
        )

    return True, None


def validate_file_content(content: bytes, content_type: str) -> Tuple[bool, Optional[str]]:
    """
    Validate file content matches declared type using magic bytes.

    Args:
        content: File content bytes
        content_type: Declared MIME type

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(content) < 4:
        return False, "File content too short"

    if content_type == "application/pdf":
        # PDF files start with %PDF
        if not content.startswith(b"%PDF"):
            return False, "File content doesn't match PDF format"

    elif content_type == "text/plain":
        # Text files should be valid UTF-8 or ASCII
        try:
            content[:1024].decode("utf-8")
        except UnicodeDecodeError:
            return False, "File doesn't appear to be valid text"

    elif (
        content_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        # DOCX files are ZIP archives (PK header)
        if not content.startswith(b"PK\x03\x04"):
            return False, "File content doesn't match DOCX format"

    return True, None


def validate_upload_path(file_path: Path, upload_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that the upload path is within the allowed directory.

    Args:
        file_path: Absolute path where file will be saved
        upload_dir: Absolute path of allowed upload directory

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Resolve to absolute paths
        file_path_resolved = file_path.resolve()
        upload_dir_resolved = upload_dir.resolve()

        # Check if file path is within upload directory
        if not file_path_resolved.is_relative_to(upload_dir_resolved):
            return False, "File path is outside allowed upload directory"

        return True, None

    except (ValueError, OSError) as e:
        return False, f"Path validation error: {str(e)}"
