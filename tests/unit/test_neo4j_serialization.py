"""Unit tests for Neo4j data serialization."""
from datetime import datetime
from app.utils.neo4j_serialization import serialize_neo4j_data


class TestSerializeNeo4jData:
    """Test Neo4j data serialization."""

    def test_serialize_simple_types(self):
        """Test serializing simple data types."""
        assert serialize_neo4j_data(42) == 42
        assert serialize_neo4j_data("hello") == "hello"
        assert serialize_neo4j_data(3.14) == 3.14
        assert serialize_neo4j_data(True) is True
        assert serialize_neo4j_data(None) is None

    def test_serialize_dict(self):
        """Test serializing dictionary."""
        data = {"key1": "value1", "key2": 42}
        result = serialize_neo4j_data(data)

        assert result == data
        assert isinstance(result, dict)

    def test_serialize_list(self):
        """Test serializing list."""
        data = [1, 2, "three", 4.0]
        result = serialize_neo4j_data(data)

        assert result == data
        assert isinstance(result, list)

    def test_serialize_nested_structures(self):
        """Test serializing nested data structures."""
        data = {
            "users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25},
            ],
            "metadata": {"count": 2, "active": True},
        }
        result = serialize_neo4j_data(data)

        assert result == data
        assert isinstance(result["users"], list)
        assert isinstance(result["metadata"], dict)

    def test_serialize_python_datetime(self):
        """Test serializing Python datetime objects."""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = serialize_neo4j_data(dt)

        assert isinstance(result, str)
        assert "2024-01-01" in result
        assert "12:00:00" in result

    def test_serialize_dict_with_datetime(self):
        """Test serializing dictionary containing datetime."""
        data = {
            "created_at": datetime(2024, 1, 1, 12, 0, 0),
            "name": "Test Document",
        }
        result = serialize_neo4j_data(data)

        assert isinstance(result["created_at"], str)
        assert result["name"] == "Test Document"

    def test_serialize_list_with_mixed_types(self):
        """Test serializing list with mixed types including datetime."""
        data = [
            "string",
            42,
            datetime(2024, 1, 1, 12, 0, 0),
            {"key": "value"},
        ]
        result = serialize_neo4j_data(data)

        assert len(result) == 4
        assert result[0] == "string"
        assert result[1] == 42
        assert isinstance(result[2], str)
        assert result[3] == {"key": "value"}
