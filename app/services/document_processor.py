import os
from neo4j.time import DateTime as Neo4jDateTime
from typing import List, Dict, Any, Optional
import logging

from app.core.database import neo4j_driver
from app.models.schemas import DocumentInfo, ProcessingStatus
from app.services.text_extractor import TextExtractor
from app.services.embedding_service import EmbeddingService
from app.services.entity_extractor import EntityExtractor
from app.utils.text_splitter import TextSplitter
from app.utils.neo4j_serialization import serialize_neo4j_data

logger = logging.getLogger(__name__)


def convert_neo4j_datetime(neo4j_dt):
    """Convert Neo4j DateTime to Python datetime"""
    if isinstance(neo4j_dt, Neo4jDateTime):
        return neo4j_dt.to_native()
    elif hasattr(neo4j_dt, "to_native"):
        return neo4j_dt.to_native()
    return neo4j_dt


class DocumentProcessor:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.embedding_service = EmbeddingService()
        self.entity_extractor = EntityExtractor()
        self.text_splitter = TextSplitter()
        self.processing_status = {}

    async def create_document_placeholder(
        self, document_id: str, filename: str, size: int
    ):
        """Create a basic document record in Neo4j immediately after upload."""
        # Ensure connection is healthy
        if not neo4j_driver.driver:
            neo4j_driver.initialize()
        else:
            try:
                with neo4j_driver.driver.session() as test_session:
                    test_session.run("RETURN 1").single()
            except Exception:
                neo4j_driver.close()
                neo4j_driver.initialize()

        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            session.run(
                """
                CREATE (d:Document {
                    id: $document_id,
                    filename: $filename,
                    size: $size,
                    created_at: datetime(),
                    status: 'processing'
                })
            """,
                document_id=document_id,
                filename=filename,
                size=size,
            )

    async def process_document(
        self, document_id: str, file_path: str, original_filename: str
    ):
        print("========== STARTING DOCUMENT PROCESSING ==========")
        print(f"Document ID: {document_id}")
        print(f"File path: {file_path}")
        print(f"Original filename: {original_filename}")

        try:
            self.processing_status[document_id] = ProcessingStatus(
                document_id=document_id,
                status="processing",
                progress=0.0,
                message="Starting document processing",
            )
            print(f"Created processing status for {document_id}")

            text_content = await self.text_extractor.extract_text(file_path)
            self.processing_status[document_id].progress = 0.2
            self.processing_status[document_id].message = "Text extracted"

            chunks = self.text_splitter.split_text(text_content)
            self.processing_status[document_id].progress = 0.3
            self.processing_status[
                document_id
            ].message = f"Text split into {len(chunks)} chunks"

            chunk_embeddings = await self.embedding_service.embed_texts(chunks)
            self.processing_status[document_id].progress = 0.5
            self.processing_status[document_id].message = "Embeddings generated"

            (
                entities,
                relationships,
            ) = await self.entity_extractor.extract_entities_and_relations(text_content)
            self.processing_status[document_id].progress = 0.7
            self.processing_status[
                document_id
            ].message = f"Extracted {len(entities)} entities and {len(relationships)} relationships"

            entity_embeddings = await self.embedding_service.embed_texts(
                [e["name"] + " " + e.get("description", "") for e in entities]
            )
            self.processing_status[document_id].progress = 0.8
            self.processing_status[document_id].message = "Entity embeddings generated"

            await self._store_in_neo4j(
                document_id=document_id,
                filename=original_filename,
                text_content=text_content,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                entities=entities,
                entity_embeddings=entity_embeddings,
                relationships=relationships,
            )

            self.processing_status[document_id].progress = 1.0
            self.processing_status[document_id].status = "completed"
            self.processing_status[
                document_id
            ].message = "Document processing completed"
            self.processing_status[document_id].chunks_processed = len(chunks)
            self.processing_status[document_id].entities_extracted = len(entities)
            self.processing_status[document_id].relationships_created = len(
                relationships
            )

            os.remove(file_path)

        except Exception as e:
            logger.error(f"Error processing document {document_id}: {e}")
            self.processing_status[document_id].status = "failed"
            self.processing_status[document_id].message = f"Processing failed: {str(e)}"

    async def _store_in_neo4j(
        self,
        document_id: str,
        filename: str,
        text_content: str,
        chunks: List[str],
        chunk_embeddings: List[List[float]],
        entities: List[Dict[str, Any]],
        entity_embeddings: List[List[float]],
        relationships: List[Dict[str, Any]],
    ):
        # Ensure connection is healthy
        if not neo4j_driver.driver:
            neo4j_driver.initialize()
        else:
            # Test existing connection, reinitialize if needed
            try:
                with neo4j_driver.driver.session() as test_session:
                    test_session.run("RETURN 1").single()
            except Exception:
                neo4j_driver.close()
                neo4j_driver.initialize()

        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            doc_embedding = await self.embedding_service.embed_texts([text_content])

            # Update the existing document placeholder with full content and embedding
            session.run(
                """
                MATCH (d:Document {id: $document_id})
                SET d.content = $content,
                    d.size = $size,
                    d.embedding = $embedding,
                    d.status = 'completed'
            """,
                document_id=document_id,
                content=text_content,
                size=len(text_content),
                embedding=doc_embedding[0],
            )

            # Create chunks in batch to avoid N+1 queries
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_data.append({
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "index": i,
                    "embedding": embedding
                })

            if chunk_data:
                session.run(
                    """
                    MATCH (d:Document {id: $document_id})
                    UNWIND $chunks as chunk_info
                    CREATE (c:Chunk {
                        id: chunk_info.chunk_id,
                        content: chunk_info.content,
                        index: chunk_info.index,
                        embedding: chunk_info.embedding
                    })
                    CREATE (d)-[:HAS_CHUNK]->(c)
                """,
                    document_id=document_id,
                    chunks=chunk_data,
                )

            # Create entities in batch to avoid N+1 queries
            entity_nodes = {}
            entity_data = []
            for i, (entity, embedding) in enumerate(zip(entities, entity_embeddings)):
                entity_id = f"{document_id}_entity_{i}"
                entity_nodes[entity["name"]] = entity_id
                entity_data.append({
                    "entity_id": entity_id,
                    "name": entity["name"],
                    "type": entity.get("type", "UNKNOWN"),
                    "description": entity.get("description", ""),
                    "embedding": embedding
                })

            if entity_data:
                session.run(
                    """
                    MATCH (d:Document {id: $document_id})
                    UNWIND $entities as entity_info
                    CREATE (e:Entity {
                        id: entity_info.entity_id,
                        name: entity_info.name,
                        type: entity_info.type,
                        description: entity_info.description,
                        embedding: entity_info.embedding
                    })
                    CREATE (d)-[:HAS_ENTITY]->(e)
                """,
                    document_id=document_id,
                    entities=entity_data,
                )

            # Create Chunk->Entity relationships (MENTIONS)
            # Use batch processing to avoid N+1 queries
            print("========== CREATING MENTIONS RELATIONSHIPS ==========")
            print(f"Entities count: {len(entities)}")
            print(f"Chunks count: {len(chunks)}")
            logger.info(
                f"Creating chunk-entity relationships for {len(entities)} entities across {len(chunks)} chunks"
            )

            # Build list of relationships to create in a single batch query
            mentions_relationships = []
            for i, entity in enumerate(entities):
                entity_name = entity["name"].lower()
                entity_id = f"{document_id}_entity_{i}"
                print(f"Processing entity {i}: {entity_name}")

                # Find chunks that mention this entity
                for j, chunk in enumerate(chunks):
                    chunk_content = chunk.lower()
                    # Check if entity name appears in chunk (case-insensitive)
                    if entity_name in chunk_content:
                        chunk_id = f"{document_id}_chunk_{j}"
                        mentions_relationships.append({
                            "chunk_id": chunk_id,
                            "entity_id": entity_id
                        })
                        print(
                            f"  -> Found in chunk {j}: queued MENTIONS relationship"
                        )

            # Create all MENTIONS relationships in a single batch query
            if mentions_relationships:
                session.run(
                    """
                    UNWIND $relationships as rel
                    MATCH (c:Chunk {id: rel.chunk_id})
                    MATCH (e:Entity {id: rel.entity_id})
                    MERGE (c)-[:MENTIONS]->(e)
                """,
                    relationships=mentions_relationships,
                )

            mentions_count = len(mentions_relationships)
            print(
                f"========== CREATED {mentions_count} MENTIONS RELATIONSHIPS (BATCH) =========="
            )
            logger.info(
                f"Created {mentions_count} MENTIONS relationships between chunks and entities using batch processing"
            )

            # Create entity relationships in batch to avoid N+1 queries
            relationship_data = []
            for rel in relationships:
                source_id = entity_nodes.get(rel["source"])
                target_id = entity_nodes.get(rel["target"])

                if source_id and target_id:
                    relationship_data.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type": rel.get("type", "RELATED"),
                        "description": rel.get("description", "")
                    })

            if relationship_data:
                session.run(
                    """
                    UNWIND $relationships as rel
                    MATCH (s:Entity {id: rel.source_id})
                    MATCH (t:Entity {id: rel.target_id})
                    CREATE (s)-[r:RELATES_TO {
                        type: rel.rel_type,
                        description: rel.description
                    }]->(t)
                """,
                    relationships=relationship_data,
                )

    async def get_all_documents(self) -> List[DocumentInfo]:
        # Ensure connection is healthy
        if not neo4j_driver.driver:
            neo4j_driver.initialize()
        else:
            # Test existing connection, reinitialize if needed
            try:
                with neo4j_driver.driver.session() as test_session:
                    test_session.run("RETURN 1").single()
            except Exception:
                neo4j_driver.close()
                neo4j_driver.initialize()

        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity)
                RETURN d.id as id, d.filename as filename, d.size as size,
                       d.created_at as created_at, count(DISTINCT c) as chunk_count,
                       count(DISTINCT e) as entity_count
            """)

            documents = []
            for record in result:
                status = "completed"
                progress = 1.0
                message = None

                if record["id"] in self.processing_status:
                    proc_status = self.processing_status[record["id"]]
                    status = proc_status.status
                    progress = proc_status.progress
                    message = proc_status.message

                # Convert Neo4j DateTime to Python datetime
                created_at = record["created_at"]
                if hasattr(created_at, "to_native"):
                    created_at = created_at.to_native()

                documents.append(
                    DocumentInfo(
                        id=record["id"],
                        filename=record["filename"],
                        size=record["size"],
                        chunk_count=record["chunk_count"],
                        entity_count=record["entity_count"],
                        created_at=serialize_neo4j_data(created_at),
                        status=status,
                        progress=progress,
                        message=message,
                    )
                )

            return documents

    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        # Ensure connection is healthy
        if not neo4j_driver.driver:
            neo4j_driver.initialize()
        else:
            # Test existing connection, reinitialize if needed
            try:
                with neo4j_driver.driver.session() as test_session:
                    test_session.run("RETURN 1").single()
            except Exception:
                neo4j_driver.close()
                neo4j_driver.initialize()

        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity)
                RETURN d.id as id, d.filename as filename, d.size as size,
                       d.created_at as created_at, count(DISTINCT c) as chunk_count,
                       count(DISTINCT e) as entity_count
            """,
                document_id=document_id,
            )

            record = result.single()
            if not record:
                return None

            status = "completed"
            if document_id in self.processing_status:
                status = self.processing_status[document_id].status

            # Convert Neo4j DateTime to Python datetime
            created_at = record["created_at"]
            if hasattr(created_at, "to_native"):
                created_at = created_at.to_native()

            return DocumentInfo(
                id=record["id"],
                filename=record["filename"],
                size=record["size"],
                chunk_count=record["chunk_count"],
                entity_count=record["entity_count"],
                created_at=serialize_neo4j_data(created_at),
                status=status,
            )

    async def delete_document(self, document_id: str) -> bool:
        # Ensure connection is healthy
        if not neo4j_driver.driver:
            neo4j_driver.initialize()
        else:
            # Test existing connection, reinitialize if needed
            try:
                with neo4j_driver.driver.session() as test_session:
                    test_session.run("RETURN 1").single()
            except Exception:
                neo4j_driver.close()
                neo4j_driver.initialize()

        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (d)-[:HAS_ENTITY]->(e:Entity)
                OPTIONAL MATCH (e)-[r:RELATES_TO]-(other:Entity)
                WITH d, collect(DISTINCT c) as chunks, collect(DISTINCT e) as entities, collect(DISTINCT r) as relations
                FOREACH (chunk IN chunks | DETACH DELETE chunk)
                FOREACH (entity IN entities | DETACH DELETE entity)
                FOREACH (relation IN relations | DELETE relation)
                DETACH DELETE d
                RETURN count(d) as deleted
            """,
                document_id=document_id,
            )

            record = result.single()

            # Clean up uploaded files from filesystem
            try:
                import glob

                file_pattern = os.path.join("uploads", f"{document_id}_*")
                for file_path in glob.glob(file_pattern):
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not delete file for document {document_id}: {e}")

            if document_id in self.processing_status:
                del self.processing_status[document_id]

            return record and record["deleted"] > 0

    async def get_processing_status(
        self, document_id: str
    ) -> Optional[ProcessingStatus]:
        return self.processing_status.get(document_id)
