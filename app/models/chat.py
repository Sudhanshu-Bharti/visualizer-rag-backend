from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4


class ChatMessage(BaseModel):
    """Individual chat message model"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional message metadata"
    )

    # RAG-specific fields
    retrieved_chunks: Optional[List[str]] = Field(
        default=None, description="IDs of chunks used for this response"
    )
    source_documents: Optional[List[str]] = Field(
        default=None, description="IDs of documents referenced"
    )
    parameters_used: Optional[Dict[str, Any]] = Field(
        default=None, description="RAG parameters used"
    )


class ChatSession(BaseModel):
    """Chat session model with metadata"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: Optional[str] = Field(
        default=None, description="Session title (auto-generated or user-set)"
    )
    user_id: Optional[str] = Field(default=None, description="User identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0, description="Total messages in session")
    is_active: bool = Field(default=True, description="Whether session is active")
    tags: List[str] = Field(
        default_factory=list, description="Session tags for organization"
    )

    # Summary fields
    summary: Optional[str] = Field(
        default=None, description="AI-generated session summary"
    )
    last_activity: datetime = Field(default_factory=datetime.utcnow)


class ChatSessionCreate(BaseModel):
    """Request model for creating chat session"""

    title: Optional[str] = None
    user_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class ChatSessionUpdate(BaseModel):
    """Request model for updating chat session"""

    title: Optional[str] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None


class ChatSessionResponse(BaseModel):
    """Response model for chat session with messages"""

    session: ChatSession
    messages: List[ChatMessage]
    total_messages: int


class ChatSessionSummary(BaseModel):
    """Condensed session info for listings"""

    id: str
    title: Optional[str]
    created_at: datetime
    last_activity: datetime
    message_count: int
    is_active: bool
    tags: List[str]
    preview_message: Optional[str] = Field(
        default=None, description="Last user message preview"
    )
