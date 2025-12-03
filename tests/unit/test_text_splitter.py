"""Unit tests for text splitter."""
from app.utils.text_splitter import TextSplitter


class TestTextSplitter:
    """Test text splitting functionality."""

    def test_split_short_text(self):
        """Test splitting text shorter than chunk size."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This is a short text."
        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Test splitting text longer than chunk size."""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is a sentence. " * 10  # ~200 characters
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 50 + 20  # Allow some flexibility

    def test_split_with_overlap(self):
        """Test that chunks have overlapping content."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = splitter.split_text(text)

        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            for i in range(len(chunks) - 1):
                # At least some words should overlap
                assert any(
                    word in chunks[i + 1]
                    for word in chunks[i].split()[-5:]
                )

    def test_split_empty_text(self):
        """Test handling empty text."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text("")

        assert len(chunks) == 0

    def test_clean_text_whitespace(self):
        """Test text cleaning removes extra whitespace."""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "This   has    extra   spaces."
        chunks = splitter.split_text(text)

        assert "  " not in chunks[0]

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        splitter = TextSplitter(chunk_size=200, chunk_overlap=20)
        text = "First sentence. Second sentence! Third sentence?"
        result = splitter._split_into_sentences(text)

        assert len(result) >= 3
