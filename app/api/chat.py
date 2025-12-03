from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import logging

from app.models.schemas import ChatRequest
from app.models.chat import ChatSessionCreate, ChatSessionUpdate
from app.utils.neo4j_serialization import serialize_neo4j_data
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()
rag_service = RAGService()
# Fallback to in-memory chat sessions if Neo4j is unavailable
chat_sessions: Dict[str, list] = {}

# Try to initialize chat service, fallback to in-memory if Neo4j fails
try:
    from app.services.chat_service import ChatService

    chat_service = ChatService()
    use_neo4j_chat = True
    logger.info("✅ Neo4j chat service initialized successfully")
except Exception as e:
    logger.warning(f"⚠️ Neo4j chat service failed, using in-memory fallback: {e}")
    chat_service = None
    use_neo4j_chat = False
@router.post("/")
async def chat(request: ChatRequest):
    """Stateless chat endpoint - no session persistence"""
    try:
        logger.info(f"Chat request: {request.message[:50]}...")

        # Step 1: Get parameters
        parameters = request.parameters or {}

        # Step 2: Retrieve context using RAG
        rag_service_instance = RAGService()
        context = await rag_service_instance.retrieve_context(
            query=request.message,
            top_k=parameters.get("top_k", 5),
            similarity_threshold=parameters.get("similarity_threshold", 0.5),
            max_hops=parameters.get("max_hops", 2),
        )
        logger.info(f"Retrieved context: {len(context)} items")

        # Step 3: Generate response
        response_text = await rag_service_instance.generate_response(
            query=request.message,
            context=context,
            temperature=parameters.get("temperature", 0.7),
            max_tokens=parameters.get("max_tokens", 512),
            model_name=parameters.get("model_name"),  # Pass model_name parameter
        )
        logger.info(f"Generated response: {response_text[:50]}...")

        # Step 4: Return response (no session storage)
        response_data = {
            "response": response_text,
            "retrieved_context": context,
            "context_items": len(context),
            "parameters_used": parameters,
        }

        return response_data

    except Exception as e:
        logger.error(f"Chat error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
# Enhanced chat session management endpoints inspired by Vercel AI Chatbot
@router.post("/sessions")
async def create_chat_session(session_data: ChatSessionCreate):
    """Create a new chat session"""
    session = await chat_service.create_session(session_data)
    return JSONResponse(content=serialize_neo4j_data(session.dict()))
@router.get("/sessions")
async def list_chat_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of sessions to return"),
):
    """List chat sessions with optional filtering"""
    if use_neo4j_chat and chat_service:
        try:
            sessions = await chat_service.list_sessions(user_id=user_id, limit=limit)
            return JSONResponse(
                content=serialize_neo4j_data([s.dict() for s in sessions])
            )
        except Exception as e:
            logger.warning(f"Neo4j session list failed, using in-memory fallback: {e}")

    # Fallback to in-memory sessions
    return {
        "sessions": [
            {
                "session_id": sid,
                "message_count": len(messages),
                "last_activity": messages[-1]["timestamp"] if messages else None,
            }
            for sid, messages in chat_sessions.items()
        ]
    }
@router.get("/sessions/search")
async def search_chat_sessions(
    q: str = Query(..., description="Search query"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(20, ge=1, le=50, description="Number of results to return"),
):
    """Search chat sessions by title or message content"""
    sessions = await chat_service.search_sessions(query=q, user_id=user_id, limit=limit)
    return JSONResponse(content=serialize_neo4j_data([s.dict() for s in sessions]))
@router.get("/{session_id}")
async def get_chat_session(session_id: str):
    """Get a specific chat session with metadata"""
    session = await chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(content=serialize_neo4j_data(session.dict()))
@router.get("/{session_id}/messages")
async def get_chat_messages(
    session_id: str,
    limit: int = Query(100, ge=1, le=500, description="Number of messages to return"),
):
    """Get messages for a chat session"""
    # Verify session exists
    session = await chat_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await chat_service.get_session_messages(session_id, limit=limit)

    response_data = {
        "session_id": session_id,
        "messages": [m.dict() for m in messages],
        "total_messages": len(messages),
    }

    return JSONResponse(content=serialize_neo4j_data(response_data))
@router.put("/{session_id}")
async def update_chat_session(session_id: str, update_data: ChatSessionUpdate):
    """Update chat session metadata (title, tags, etc.)"""
    session = await chat_service.update_session(session_id, update_data)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse(content=serialize_neo4j_data(session.dict()))
@router.delete("/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session and all its messages"""
    deleted = await chat_service.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted successfully", "session_id": session_id}
# Legacy endpoint for backward compatibility
@router.get("/{session_id}/history")
async def get_chat_history(session_id: str):
    """Legacy endpoint - get chat history with fallback support"""
    if use_neo4j_chat and chat_service:
        try:
            return await get_chat_messages(session_id)
        except Exception as e:
            logger.warning(f"Neo4j history failed, using in-memory fallback: {e}")

    # Fallback to in-memory sessions
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "messages": chat_sessions[session_id]}
