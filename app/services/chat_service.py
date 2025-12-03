from typing import List, Optional
from datetime import datetime
import uuid
import logging

from app.core.database import neo4j_driver
from app.models.chat import (
    ChatSession,
    ChatMessage,
    ChatSessionCreate,
    ChatSessionUpdate,
    ChatSessionSummary,
)

logger = logging.getLogger(__name__)


class ChatService:
    """Service for managing chat sessions and messages with Neo4j persistence"""

    def __init__(self) -> None:
        self.driver = neo4j_driver.get_driver()
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        """Create necessary constraints and indexes for chat data"""
        with self.driver.session() as session:
            # Create constraints
            session.run("""
                CREATE CONSTRAINT chat_session_id IF NOT EXISTS
                FOR (s:ChatSession) REQUIRE s.id IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT chat_message_id IF NOT EXISTS
                FOR (m:ChatMessage) REQUIRE m.id IS UNIQUE
            """)

            # Create indexes for better performance
            session.run("""
                CREATE INDEX chat_session_user_id IF NOT EXISTS
                FOR (s:ChatSession) ON s.user_id
            """)

            session.run("""
                CREATE INDEX chat_session_created_at IF NOT EXISTS
                FOR (s:ChatSession) ON s.created_at
            """)

    async def create_session(self, session_data: ChatSessionCreate) -> ChatSession:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Auto-generate title if not provided
        title = session_data.title or f"Chat {now.strftime('%Y-%m-%d %H:%M')}"

        session_obj = ChatSession(
            id=session_id,
            title=title,
            user_id=session_data.user_id,
            created_at=now,
            updated_at=now,
            tags=session_data.tags,
        )

        with self.driver.session() as neo4j_session:
            # Convert datetime objects to ISO strings for Neo4j
            session_dict = session_obj.dict()
            session_dict["created_at"] = session_dict["created_at"].isoformat()
            session_dict["updated_at"] = session_dict["updated_at"].isoformat()
            session_dict["last_activity"] = session_dict["last_activity"].isoformat()

            neo4j_session.run(
                """
                CREATE (s:ChatSession {
                    id: $id,
                    title: $title,
                    user_id: $user_id,
                    created_at: datetime($created_at),
                    updated_at: datetime($updated_at),
                    message_count: $message_count,
                    is_active: $is_active,
                    tags: $tags,
                    last_activity: datetime($last_activity)
                })
            """,
                **session_dict,
            )

        logger.info(f"Created chat session: {session_id}")
        return session_obj

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID"""
        with self.driver.session() as neo4j_session:
            result = neo4j_session.run(
                """
                MATCH (s:ChatSession {id: $session_id})
                RETURN s
            """,
                session_id=session_id,
            )

            record = result.single()
            if not record:
                return None

            session_data = dict(record["s"])
            # Convert Neo4j datetime to Python datetime
            for key in ["created_at", "updated_at", "last_activity"]:
                if key in session_data:
                    session_data[key] = session_data[key].to_native()

            return ChatSession(**session_data)

    async def update_session(
        self, session_id: str, update_data: ChatSessionUpdate
    ) -> Optional[ChatSession]:
        """Update a chat session"""
        update_fields = []
        params = {"session_id": session_id, "updated_at": datetime.utcnow()}

        if update_data.title is not None:
            update_fields.append("s.title = $title")
            params["title"] = update_data.title

        if update_data.tags is not None:
            update_fields.append("s.tags = $tags")
            params["tags"] = update_data.tags

        if update_data.is_active is not None:
            update_fields.append("s.is_active = $is_active")
            params["is_active"] = update_data.is_active

        if not update_fields:
            return await self.get_session(session_id)

        update_fields.append("s.updated_at = datetime($updated_at)")

        with self.driver.session() as neo4j_session:
            query = f"""
                MATCH (s:ChatSession {{id: $session_id}})
                SET {", ".join(update_fields)}
                RETURN s
            """

            result = neo4j_session.run(query, **params)
            record = result.single()

            if not record:
                return None

            session_data = dict(record["s"])
            for key in ["created_at", "updated_at", "last_activity"]:
                if key in session_data:
                    session_data[key] = session_data[key].to_native()

            return ChatSession(**session_data)

    async def list_sessions(
        self, user_id: Optional[str] = None, limit: int = 50
    ) -> List[ChatSessionSummary]:
        """List chat sessions with optional user filtering"""
        query = """
            MATCH (s:ChatSession)
            WHERE ($user_id IS NULL OR s.user_id = $user_id) AND s.is_active = true
            OPTIONAL MATCH (s)-[:CONTAINS]->(m:ChatMessage {role: 'user'})
            WITH s, m ORDER BY m.timestamp DESC
            WITH s, collect(m.content)[0] as last_user_message
            RETURN s, last_user_message
            ORDER BY s.last_activity DESC
            LIMIT $limit
        """

        with self.driver.session() as neo4j_session:
            result = neo4j_session.run(query, user_id=user_id, limit=limit)

            sessions = []
            for record in result:
                session_data = dict(record["s"])
                for key in ["created_at", "updated_at", "last_activity"]:
                    if key in session_data:
                        session_data[key] = session_data[key].to_native()

                summary = ChatSessionSummary(
                    **session_data, preview_message=record["last_user_message"]
                )
                sessions.append(summary)

            return sessions

    async def add_message(self, session_id: str, message: ChatMessage) -> ChatMessage:
        """Add a message to a chat session"""
        with self.driver.session() as neo4j_session:
            # Add message and update session
            neo4j_session.run(
                """
                MATCH (s:ChatSession {id: $session_id})
                CREATE (m:ChatMessage {
                    id: $message_id,
                    role: $role,
                    content: $content,
                    timestamp: datetime($timestamp),
                    metadata: $metadata,
                    retrieved_chunks: $retrieved_chunks,
                    source_documents: $source_documents,
                    parameters_used: $parameters_used
                })
                CREATE (s)-[:CONTAINS]->(m)
                SET s.message_count = s.message_count + 1,
                    s.last_activity = datetime($timestamp),
                    s.updated_at = datetime($timestamp)
            """,
                session_id=session_id,
                message_id=message.id,
                role=message.role,
                content=message.content,
                timestamp=message.timestamp.isoformat(),
                metadata=message.metadata,
                retrieved_chunks=message.retrieved_chunks,
                source_documents=message.source_documents,
                parameters_used=message.parameters_used,
            )

        logger.info(f"Added {message.role} message to session {session_id}")
        return message

    async def get_session_messages(
        self, session_id: str, limit: int = 100
    ) -> List[ChatMessage]:
        """Get messages for a chat session"""
        with self.driver.session() as neo4j_session:
            result = neo4j_session.run(
                """
                MATCH (s:ChatSession {id: $session_id})-[:CONTAINS]->(m:ChatMessage)
                RETURN m
                ORDER BY m.timestamp ASC
                LIMIT $limit
            """,
                session_id=session_id,
                limit=limit,
            )

            messages = []
            for record in result:
                message_data = dict(record["m"])
                # Convert Neo4j datetime to Python datetime
                if "timestamp" in message_data:
                    message_data["timestamp"] = message_data["timestamp"].to_native()

                messages.append(ChatMessage(**message_data))

            return messages

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        with self.driver.session() as neo4j_session:
            result = neo4j_session.run(
                """
                MATCH (s:ChatSession {id: $session_id})
                OPTIONAL MATCH (s)-[:CONTAINS]->(m:ChatMessage)
                DETACH DELETE s, m
                RETURN count(s) as deleted_count
            """,
                session_id=session_id,
            )

            record = result.single()
            deleted_count = record["deleted_count"] if record else 0

            logger.info(
                f"Deleted chat session {session_id}, removed {deleted_count} sessions"
            )
            return deleted_count > 0

    async def search_sessions(
        self, query: str, user_id: Optional[str] = None, limit: int = 20
    ) -> List[ChatSessionSummary]:
        """Search chat sessions by title or message content"""
        search_query = """
            MATCH (s:ChatSession)
            WHERE ($user_id IS NULL OR s.user_id = $user_id)
                AND s.is_active = true
                AND (s.title CONTAINS $query
                     OR exists {
                         MATCH (s)-[:CONTAINS]->(m:ChatMessage)
                         WHERE m.content CONTAINS $query
                     })
            OPTIONAL MATCH (s)-[:CONTAINS]->(m:ChatMessage {role: 'user'})
            WITH s, m ORDER BY m.timestamp DESC
            WITH s, collect(m.content)[0] as last_user_message
            RETURN s, last_user_message
            ORDER BY s.last_activity DESC
            LIMIT $limit
        """

        with self.driver.session() as neo4j_session:
            result = neo4j_session.run(
                search_query, query=query, user_id=user_id, limit=limit
            )

            sessions = []
            for record in result:
                session_data = dict(record["s"])
                for key in ["created_at", "updated_at", "last_activity"]:
                    if key in session_data:
                        session_data[key] = session_data[key].to_native()

                summary = ChatSessionSummary(
                    **session_data, preview_message=record["last_user_message"]
                )
                sessions.append(summary)

            return sessions
