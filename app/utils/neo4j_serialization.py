"""Neo4j data serialization utilities."""

from typing import Any
from datetime import datetime
from neo4j.time import DateTime as Neo4jDateTime


def serialize_neo4j_data(data: Any) -> Any:
    """
    Recursively convert Neo4j data types to JSON-serializable formats.

    Args:
        data: Any data structure potentially containing Neo4j types
            (DateTime, Node, Relationship, etc.)

    Returns:
        JSON-serializable version of the data

    Examples:
        >>> from neo4j.time import DateTime
        >>> neo4j_date = DateTime(2024, 1, 1, 12, 0, 0)
        >>> serialize_neo4j_data(neo4j_date)
        '2024-01-01T12:00:00'

        >>> data = {"date": neo4j_date, "values": [1, 2, neo4j_date]}
        >>> serialize_neo4j_data(data)
        {'date': '2024-01-01T12:00:00', 'values': [1, 2, '2024-01-01T12:00:00']}
    """
    if isinstance(data, dict):
        return {key: serialize_neo4j_data(value) for key, value in data.items()}

    elif isinstance(data, list):
        return [serialize_neo4j_data(item) for item in data]

    elif isinstance(data, Neo4jDateTime):
        # Neo4j DateTime objects
        return data.to_native().isoformat()

    elif hasattr(data, "to_native"):
        # Other Neo4j temporal types (Date, Time, Duration, etc.)
        native = data.to_native()
        if hasattr(native, "isoformat"):
            return native.isoformat()
        return native

    elif isinstance(data, datetime):
        # Standard Python datetime
        return data.isoformat()

    else:
        # Primitive types (str, int, float, bool, None)
        return data
