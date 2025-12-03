"""Pytest configuration and shared fixtures."""
import pytest
from typing import Generator
from unittest.mock import Mock, MagicMock
from neo4j import Driver


@pytest.fixture
def mock_neo4j_driver() -> Generator[Mock, None, None]:
    """Mock Neo4j driver for testing."""
    driver = MagicMock(spec=Driver)
    session = MagicMock()
    result = MagicMock()

    # Configure mock chain
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None
    session.run.return_value = result

    yield driver


@pytest.fixture
def sample_document_data() -> dict:
    """Sample document data for testing."""
    return {
        "id": "test-doc-123",
        "filename": "test.pdf",
        "content": "This is test content for document processing.",
        "size": 100,
    }


@pytest.fixture
def sample_chunks() -> list:
    """Sample text chunks for testing."""
    return [
        "This is the first chunk of text.",
        "This is the second chunk of text.",
        "This is the third chunk of text.",
    ]


@pytest.fixture
def sample_entities() -> list:
    """Sample entities for testing."""
    return [
        {
            "name": "Entity A",
            "type": "PERSON",
            "description": "A sample entity",
        },
        {
            "name": "Entity B",
            "type": "ORGANIZATION",
            "description": "Another sample entity",
        },
    ]


@pytest.fixture
def sample_embeddings() -> list:
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4] * 96,  # 384 dimensions
        [0.5, 0.6, 0.7, 0.8] * 96,
        [0.9, 0.8, 0.7, 0.6] * 96,
    ]
