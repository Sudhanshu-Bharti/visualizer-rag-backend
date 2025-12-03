from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime
from neo4j.time import DateTime as Neo4jDateTime


def datetime_serializer(dt):
    """Convert datetime objects to ISO format strings"""
    if hasattr(dt, "to_native"):  # Neo4j DateTime
        return dt.to_native().isoformat()
    elif isinstance(dt, datetime):
        return dt.isoformat()
    return dt


def neo4j_serializer(obj):
    """Convert Neo4j objects to JSON-serializable formats"""
    if isinstance(obj, Neo4jDateTime) or hasattr(obj, "to_native"):
        return obj.to_native().isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: datetime_serializer})

    id: str
    filename: str
    size: int
    status: str
    created_at: datetime


class DocumentInfo(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: datetime_serializer})

    id: str
    filename: str
    size: int
    chunk_count: int
    entity_count: int
    created_at: datetime
    status: str
    progress: float = 0.0  # 0.0 to 1.0 for processing progress
    message: Optional[str] = None  # Progress message


class ChatMessage(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: datetime_serializer})

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Chat session identifier")
    parameters: Optional[Dict[str, Any]] = Field(
        default={}, description="RAG parameters"
    )


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: datetime_serializer,
            Neo4jDateTime: neo4j_serializer,
        },
        arbitrary_types_allowed=True,
    )

    response: str = Field(..., description="Assistant response")
    session_id: str = Field(..., description="Chat session identifier")
    retrieved_context: List[Dict[str, Any]] = Field(
        default=[], description="Retrieved context"
    )
    graph_data: Optional[Dict[str, Any]] = Field(
        None, description="Graph visualization data"
    )
    parameters_used: Dict[str, Any] = Field(
        default={}, description="Parameters used for generation"
    )

    def model_dump(self, **kwargs):
        """Custom serialization to handle Neo4j types"""
        data = super().model_dump(**kwargs)
        return self._serialize_neo4j_types(data)

    def _serialize_neo4j_types(self, obj):
        """Recursively serialize Neo4j types in nested structures"""
        if isinstance(obj, dict):
            return {k: self._serialize_neo4j_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_neo4j_types(item) for item in obj]
        elif isinstance(obj, Neo4jDateTime) or hasattr(obj, "to_native"):
            return obj.to_native().isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj


class RAGParameters(BaseModel):
    top_k: int = Field(5, ge=1, le=20, description="Number of top results to retrieve")
    similarity_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Similarity threshold"
    )
    max_hops: int = Field(2, ge=1, le=5, description="Maximum graph traversal hops")
    include_entities: bool = Field(True, description="Include entities in retrieval")
    include_relationships: bool = Field(True, description="Include relationships")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(
        512, ge=50, le=2000, description="Maximum tokens to generate"
    )
    model_name: str = Field("qwen2.5:7b-instruct", description="LLM model name")


class GraphNode(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: datetime_serializer,
            Neo4jDateTime: neo4j_serializer,
        }
    )

    id: str
    label: str
    type: str
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: datetime_serializer,
            Neo4jDateTime: neo4j_serializer,
        }
    )

    id: str
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class GraphData(BaseModel):
    model_config = ConfigDict(
        json_encoders={
            datetime: datetime_serializer,
            Neo4jDateTime: neo4j_serializer,
        }
    )

    nodes: List[GraphNode]
    edges: List[GraphEdge]


class GraphQueryRequest(BaseModel):
    query: Optional[str] = Field(None, description="Cypher query or natural language")
    node_limit: int = Field(100, ge=10, le=1000, description="Maximum nodes to return")
    include_embeddings: bool = Field(False, description="Include embedding vectors")


class Entity(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    properties: Dict[str, Any] = {}


class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = {}


class ProcessingStatus(BaseModel):
    document_id: str
    status: str  # processing, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    chunks_processed: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
