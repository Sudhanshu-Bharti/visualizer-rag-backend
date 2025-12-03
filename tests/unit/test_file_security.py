"""Unit tests for file security utilities."""
import pytest
from app.utils.file_security import (
    sanitize_filename,
    validate_file_extension,
    validate_file_content,
    validate_upload_path,
)


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_sanitize_basic_filename(self):
        """Test sanitizing a basic filename."""
        result = sanitize_filename("test.pdf")
        assert result == "test.pdf"

    def test_sanitize_path_traversal(self):
        """Test preventing path traversal attacks."""
        result = sanitize_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result

    def test_sanitize_dangerous_characters(self):
        """Test removing dangerous characters."""
        result = sanitize_filename("test<>file.pdf")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_multiple_dots(self):
        """Test handling multiple dots in filename."""
        result = sanitize_filename("test..file...pdf")
        assert result == "test__file__.pdf"

    def test_sanitize_long_filename(self):
        """Test truncating overly long filenames."""
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_sanitize_empty_filename(self):
        """Test handling empty filename."""
        with pytest.raises(ValueError, match="Filename cannot be empty"):
            sanitize_filename("")


class TestValidateFileExtension:
    """Test file extension validation."""

    def test_valid_pdf_extension(self):
        """Test validating PDF extension."""
        is_valid, error = validate_file_extension("test.pdf", "application/pdf")
        assert is_valid is True
        assert error is None

    def test_valid_txt_extension(self):
        """Test validating text file extension."""
        is_valid, error = validate_file_extension("test.txt", "text/plain")
        assert is_valid is True
        assert error is None

    def test_invalid_extension_mismatch(self):
        """Test detecting extension/content-type mismatch."""
        is_valid, error = validate_file_extension("test.pdf", "text/plain")
        assert is_valid is False
        assert "doesn't match" in error

    def test_unsupported_content_type(self):
        """Test rejecting unsupported content types."""
        is_valid, error = validate_file_extension("test.exe", "application/exe")
        assert is_valid is False
        assert "not allowed" in error


class TestValidateFileContent:
    """Test file content validation."""

    def test_valid_pdf_magic_bytes(self):
        """Test validating PDF magic bytes."""
        pdf_content = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n"
        is_valid, error = validate_file_content(pdf_content, "application/pdf")
        assert is_valid is True
        assert error is None

    def test_invalid_pdf_magic_bytes(self):
        """Test detecting invalid PDF magic bytes."""
        fake_pdf = b"This is not a PDF"
        is_valid, error = validate_file_content(fake_pdf, "application/pdf")
        assert is_valid is False
        assert "PDF format" in error

    def test_valid_text_content(self):
        """Test validating text file content."""
        text_content = b"This is valid UTF-8 text content."
        is_valid, error = validate_file_content(text_content, "text/plain")
        assert is_valid is True
        assert error is None

    def test_invalid_text_encoding(self):
        """Test detecting invalid text encoding."""
        # Invalid UTF-8 sequence
        invalid_text = b"\xFF\xFE invalid utf-8"
        is_valid, error = validate_file_content(invalid_text, "text/plain")
        assert is_valid is False
        assert "valid text" in error

    def test_valid_docx_magic_bytes(self):
        """Test validating DOCX magic bytes (ZIP format)."""
        docx_content = b"PK\x03\x04" + b"\x00" * 100
        is_valid, error = validate_file_content(
            docx_content,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        assert is_valid is True
        assert error is None

    def test_file_too_short(self):
        """Test handling files that are too short."""
        short_content = b"ab"
        is_valid, error = validate_file_content(short_content, "application/pdf")
        assert is_valid is False
        assert "too short" in error


class TestValidateUploadPath:
    """Test upload path validation."""

    def test_valid_path_within_upload_dir(self, tmp_path):
        """Test validating path within upload directory."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        file_path = upload_dir / "test.pdf"

        is_valid, error = validate_upload_path(file_path, upload_dir)
        assert is_valid is True
        assert error is None

    def test_invalid_path_outside_upload_dir(self, tmp_path):
        """Test detecting path traversal outside upload directory."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        file_path = tmp_path / "outside" / "test.pdf"

        is_valid, error = validate_upload_path(file_path, upload_dir)
        assert is_valid is False
        assert "outside" in error
