from typing import List, Dict, Any, Optional
import logging
import asyncio
import requests
import httpx

from app.core.database import neo4j_driver
from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService
from app.utils.neo4j_serialization import serialize_neo4j_data

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.ollama_base_url = settings.ollama_base_url
        self.gemini_service = GeminiService()

    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        max_hops: int = 2,
        include_entities: bool = True,
        include_relationships: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using vector similarity search and graph expansion.
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.embed_single_text(query)
            logger.info(f"Generated query embedding for: {query[:50]}...")

            context_items = []

            # Retrieve similar chunks
            chunks = await self._retrieve_similar_chunks(
                query_embedding, top_k, similarity_threshold
            )
            context_items.extend(chunks)
            logger.info(f"Retrieved {len(chunks)} similar chunks")

            # Retrieve similar entities if enabled
            if include_entities:
                entities = await self._retrieve_similar_entities(
                    query_embedding,
                    top_k // 2,  # Use fewer entities than chunks
                    similarity_threshold,
                )
                context_items.extend(entities)
                logger.info(f"Retrieved {len(entities)} similar entities")

            # Expand context with relationships if enabled
            if include_relationships and max_hops > 0:
                expanded = await self._expand_context_with_relationships(
                    context_items, max_hops
                )
                context_items.extend(expanded)
                logger.info(f"Expanded with {len(expanded)} related items")

            # Deduplicate and sort by relevance
            final_context = self._deduplicate_context(context_items)
            logger.info(f"Final context: {len(final_context)} items")

            # Serialize Neo4j DateTime objects for JSON compatibility
            serialized_context = serialize_neo4j_data(final_context[: top_k * 2])
            return serialized_context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            # Fallback to mock contextual data if vector search fails
            return [
                {
                    "chunk_id": "fallback_chunk_1",
                    "content": f"I found some information related to '{query}' in the uploaded documents. The system contains documents about NFT projects, AI technologies, and various technical topics. However, I'm currently unable to perform precise similarity search due to database connectivity issues.",
                    "similarity_score": 0.7,
                    "type": "chunk",
                },
                {
                    "entity_id": "fallback_entity_1",
                    "name": "AI Technologies",
                    "description": "Various artificial intelligence technologies mentioned in documents",
                    "similarity_score": 0.6,
                    "type": "entity",
                },
            ]

    async def _retrieve_similar_chunks(
        self, query_embedding: List[float], top_k: int, similarity_threshold: float
    ) -> List[Dict[str, Any]]:
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
            similarity_query = """
                CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node.id as chunk_id, node.content as content, score,
                       node.index as chunk_index
            """

            result = session.run(
                similarity_query,
                top_k=top_k,
                query_vector=query_embedding,
                threshold=similarity_threshold,
            )

            chunks = []
            for record in result:
                chunks.append(
                    {
                        "type": "chunk",
                        "chunk_id": record["chunk_id"],
                        "content": record["content"],
                        "similarity_score": record["score"],
                        "chunk_index": record["chunk_index"],
                    }
                )

            return chunks

    async def _retrieve_similar_entities(
        self, query_embedding: List[float], top_k: int, similarity_threshold: float
    ) -> List[Dict[str, Any]]:
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
            similarity_query = """
                CALL db.index.vector.queryNodes('entity_embeddings', $top_k, $query_vector)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node.id as entity_id, node.name as name, node.type as type,
                       node.description as description, score
            """

            result = session.run(
                similarity_query,
                top_k=top_k,
                query_vector=query_embedding,
                threshold=similarity_threshold,
            )

            entities = []
            for record in result:
                entities.append(
                    {
                        "type": "entity",
                        "entity_id": record["entity_id"],
                        "name": record["name"],
                        "entity_type": record["type"],
                        "description": record["description"],
                        "similarity_score": record["score"],
                    }
                )

            return entities

    async def _expand_context_with_relationships(
        self, context_items: List[Dict[str, Any]], max_hops: int
    ) -> List[Dict[str, Any]]:
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

        # Extract both chunk IDs and entity IDs for expansion
        chunk_ids = [
            item["chunk_id"] for item in context_items if item.get("type") == "chunk"
        ]
        entity_ids = [
            item["entity_id"] for item in context_items if item.get("type") == "entity"
        ]

        all_ids = chunk_ids + entity_ids

        if not all_ids:
            logger.warning("No chunk or entity IDs found for graph expansion")
            return []

        logger.info(
            f"Expanding graph from {len(chunk_ids)} chunks and {len(entity_ids)} entities with max_hops={max_hops}"
        )

        with driver.session() as session:
            # Expand from chunks and entities to find related entities
            expansion_query = f"""
                MATCH path = (start)-[r*1..{max_hops}]-(connected:Entity)
                WHERE start.id IN $start_ids
                AND connected.id IS NOT NULL
                UNWIND relationships(path) as rel
                RETURN DISTINCT connected.id as entity_id, connected.name as name,
                       connected.type as type, connected.description as description,
                       type(rel) as relationship_type,
                       CASE WHEN rel.description IS NOT NULL THEN rel.description ELSE '' END as rel_description,
                       length(path) as hop_distance
                ORDER BY hop_distance ASC
            """

            result = session.run(expansion_query, start_ids=all_ids)

            expanded_entities = []
            for record in result:
                # Calculate score based on hop distance (closer = higher score)
                hop_distance = record["hop_distance"]
                similarity_score = 0.7 - (hop_distance * 0.1)  # Decay by distance

                expanded_entities.append(
                    {
                        "type": "expanded_entity",
                        "entity_id": record["entity_id"],
                        "name": record["name"],
                        "entity_type": record["type"],
                        "description": record["description"],
                        "relationship_type": record["relationship_type"],
                        "relationship_description": record["rel_description"],
                        "hop_distance": hop_distance,
                        "similarity_score": max(0.3, similarity_score),  # Min score 0.3
                    }
                )

            logger.info(
                f"Graph expansion found {len(expanded_entities)} related entities"
            )
            return expanded_entities[: max_hops * 10]  # Allow more expanded results

    def _deduplicate_context(
        self, context_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen_ids = set()
        deduplicated = []

        for item in context_items:
            item_id = item.get("chunk_id") or item.get("entity_id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                deduplicated.append(item)

        deduplicated.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return deduplicated

    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        model_name: Optional[str] = None,
    ) -> str:
        try:
            # First try Ollama
            context_text = self._format_context_for_prompt(context)
            prompt = self._build_rag_prompt(query, context_text)

            response = await self._call_ollama(
                prompt=prompt,
                context=context,
                model_name=model_name or settings.ollama_model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response

        except Exception as ollama_error:
            logger.warning(f"Ollama failed: {ollama_error}, trying Gemini fallback")
            
            # Fallback to Gemini if Ollama fails
            if self.gemini_service.is_available():
                try:
                    response = await self.gemini_service.generate_rag_response(
                        query=query,
                        context=context,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    logger.info("Successfully used Gemini as fallback")
                    return response
                except Exception as gemini_error:
                    logger.error(f"Gemini fallback also failed: {gemini_error}")
            else:
                logger.warning("Gemini API not configured, cannot use as fallback")

            # If both fail, use context-based fallback response
            logger.error("Both Ollama and Gemini failed, using context-based fallback")
            return self._generate_fallback_response(context)

    def _format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        formatted_context = []

        chunks = [item for item in context if item.get("type") == "chunk"]
        entities = [
            item
            for item in context
            if item.get("type") in ["entity", "expanded_entity"]
        ]

        if chunks:
            formatted_context.append("=== RELEVANT TEXT CHUNKS ===")
            for i, chunk in enumerate(chunks[:5], 1):
                formatted_context.append(f"{i}. {chunk.get('content', '')}")

        if entities:
            formatted_context.append("\n=== RELEVANT ENTITIES ===")
            for i, entity in enumerate(entities[:10], 1):
                entity_info = f"{i}. {entity.get('name', '')} ({entity.get('entity_type', 'Unknown')})"
                if entity.get("description"):
                    entity_info += f": {entity.get('description')}"
                formatted_context.append(entity_info)

        return "\n".join(formatted_context)

    def _build_rag_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user's question accurately and concisely.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based primarily on the provided context
- If the context doesn't contain enough information to fully answer the question, say so
- Be specific and cite relevant information from the context
- Keep your response concise but informative
- If you mention entities or relationships, explain their relevance

ANSWER:"""

        return prompt

    async def _call_ollama(
        self,
        prompt: str,
        context: List[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        def _sync_call_ollama():
            try:
                payload = {
                    "model": model_name or settings.ollama_model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "top_k": 40,
                    },
                }

                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "RAGViz/1.0",
                    },
                    timeout=60.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get(
                        "response", "Sorry, I couldn't generate a response."
                    )
                else:
                    logger.error(
                        f"Ollama API error: {response.status_code} - {response.text}"
                    )
                    raise requests.RequestException(f"HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                logger.error("Timeout calling Ollama API")
                raise requests.exceptions.Timeout("Timeout")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error calling Ollama API: {e}")
                raise e
            except Exception as e:
                logger.error(f"Unexpected error calling Ollama API: {e}")
                raise e

        try:
            # Run the synchronous requests call in a thread pool
            return await asyncio.to_thread(_sync_call_ollama)
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again."
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error calling Ollama API: {e}")
            return self._generate_fallback_response(context)
        except Exception as e:
            logger.error(f"Unexpected error calling Ollama API: {e}")
            return self._generate_fallback_response(context)

    def _generate_fallback_response(self, context: List[Dict[str, Any]]) -> str:
        """Generate a response based on retrieved context when LLM is unavailable."""
        if not context:
            return "I apologize, but I couldn't find relevant information to answer your question. The language model is also unavailable at the moment."

        # Extract information from context
        chunks = [item for item in context if item.get("type") == "chunk"]
        entities = [
            item
            for item in context
            if item.get("type") in ["entity", "expanded_entity"]
        ]

        response_parts = []

        if chunks:
            response_parts.append("Based on the documents, here's what I found:")
            for i, chunk in enumerate(chunks[:3], 1):  # Limit to top 3 chunks
                content = chunk.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                similarity = chunk.get("similarity_score", 0)
                response_parts.append(
                    f"\n{i}. (Similarity: {similarity:.2f}) {content}"
                )

        if entities:
            entity_names = [e.get("name") for e in entities if e.get("name")][:5]
            if entity_names:
                response_parts.append(
                    f"\n\nRelevant entities mentioned: {', '.join(entity_names)}"
                )

        response_parts.append(
            "\n\n*Note: This response was generated from document retrieval only, as the language model is currently unavailable.*"
        )

        return "".join(response_parts)

    async def get_available_models(self) -> List[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_base_url}/api/tags")

                if response.status_code == 200:
                    result = response.json()
                    models = [model["name"] for model in result.get("models", [])]
                    return models
                else:
                    return [settings.ollama_model_name]  # Default fallback

        except Exception as e:
            logger.warning(f"Could not fetch available models: {e}")
            return [settings.ollama_model_name]  # Default fallback
