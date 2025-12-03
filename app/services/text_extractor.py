import os
import logging
from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)


class TextExtractor:
    def __init__(self):
        pass

    async def extract_text(self, file_path: str) -> str:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".pdf":
                return await self._extract_from_pdf(file_path)
            elif file_extension == ".txt":
                return await self._extract_from_txt(file_path)
            elif file_extension == ".docx":
                return await self._extract_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    async def _extract_from_pdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"

            return text.strip()

        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise

    async def _extract_from_txt(self, file_path: str) -> str:
        """
        Extract text from a text file with automatic encoding detection.
        Supports multiple languages including Chinese, Japanese, Korean, etc.
        """
        # List of encodings to try, in order of preference
        encodings = [
            "utf-8",  # Most common, supports all languages
            "utf-8-sig",  # UTF-8 with BOM
            "gb2312",  # Simplified Chinese
            "gbk",  # Extended Simplified Chinese
            "gb18030",  # Chinese national standard
            "big5",  # Traditional Chinese
            "shift_jis",  # Japanese
            "euc-kr",  # Korean
            "iso-8859-1",  # Western European (latin-1)
            "cp1252",  # Windows Western European
        ]

        # First, try to detect encoding using chardet if available
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                if detected and detected["encoding"] and detected["confidence"] > 0.7:
                    detected_encoding = detected["encoding"]
                    logger.info(
                        f"Detected encoding: {detected_encoding} (confidence: {detected['confidence']:.2f})"
                    )
                    try:
                        return raw_data.decode(detected_encoding)
                    except (UnicodeDecodeError, LookupError):
                        logger.warning(
                            f"Failed to decode with detected encoding {detected_encoding}, trying alternatives"
                        )
        except ImportError:
            logger.debug("chardet not available, using fallback encoding detection")
        except Exception as e:
            logger.warning(f"Error during encoding detection: {e}")

        # Fallback: try each encoding in sequence
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as file:
                    content = file.read()
                    logger.info(f"Successfully read file with encoding: {encoding}")
                    return content
            except (UnicodeDecodeError, LookupError):
                continue
            except Exception as e:
                logger.error(f"Error reading text file with {encoding}: {e}")
                continue

        # If all encodings fail, read with utf-8 and replace errors
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as file:
                logger.warning(
                    f"All encodings failed, using UTF-8 with error replacement for {file_path}"
                )
                return file.read()
        except Exception as e:
            logger.error(f"Final fallback failed for {file_path}: {e}")
            raise

    async def _extract_from_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = ""

            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            return text.strip()

        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise
