from typing import List, Optional
import re
from app.core.config import settings


class TextSplitter:
    def __init__(
        self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.max_chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        text = self._clean_text(text)

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        sentences = self._split_into_sentences(text)

        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_chunk += sentence + " "
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if sentence_length > self.chunk_size:
                    word_chunks = self._split_long_sentence(sentence)
                    chunks.extend(word_chunks)
                    current_chunk = ""
                    current_length = 0
                else:
                    current_chunk = sentence + " "
                    current_length = sentence_length + 1

        if current_chunk:
            chunks.append(current_chunk.strip())

        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return [chunk for chunk in chunks if chunk.strip()]

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n+", " ", text)
        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_endings = r"[.!?]+\s+"
        sentences = re.split(sentence_endings, text)

        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _split_long_sentence(self, sentence: str) -> List[str]:
        words = sentence.split()
        chunks = []
        current_chunk = ""

        for word in words:
            if len(current_chunk + " " + word) <= self.chunk_size:
                current_chunk += " " + word if current_chunk else word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            prev_chunk = chunks[i - 1]
            overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)

            if overlap_text:
                overlapped_chunk = overlap_text + " " + chunk
            else:
                overlapped_chunk = chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks

    def _get_overlap_text(self, text: str, max_overlap_chars: int) -> str:
        if len(text) <= max_overlap_chars:
            return text

        words = text.split()
        overlap_text = ""

        for word in reversed(words):
            if len(word + " " + overlap_text) <= max_overlap_chars:
                overlap_text = word + " " + overlap_text if overlap_text else word
            else:
                break

        return overlap_text.strip()
