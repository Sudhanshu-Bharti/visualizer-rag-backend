from typing import List, Dict, Any, Optional
import logging
import asyncio
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google's Gemini API"""

    def __init__(self):
        self.api_key = settings.gemini_api_key
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            logger.warning("Gemini API key not configured")

    async def generate_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        """Generate a response using Gemini API"""
        if not self.model:
            raise ValueError("Gemini API key not configured")

        try:
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.9,
                top_k=40,
            )

            # Generate response asynchronously
            def _sync_generate():
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text

            # Run in thread pool to avoid blocking
            response_text = await asyncio.to_thread(_sync_generate)
            return response_text

        except Exception as e:
            logger.error(f"Error generating Gemini response: {e}")
            raise

    async def generate_rag_response(
        self,
        query: str,
        context: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """Generate a RAG response using Gemini with formatted context"""
        try:
            context_text = self._format_context_for_prompt(context)
            prompt = self._build_rag_prompt(query, context_text)

            return await self.generate_response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

        except Exception as e:
            logger.error(f"Error generating RAG response with Gemini: {e}")
            raise

    def _format_context_for_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Format retrieved context for the prompt"""
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
        """Build a RAG prompt for Gemini"""
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

    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.model is not None and bool(self.api_key)

    async def test_connection(self) -> bool:
        """Test the Gemini API connection"""
        if not self.is_available():
            return False

        try:
            await self.generate_response("Hello, this is a test.", max_tokens=10)
            return True
        except Exception as e:
            logger.error(f"Gemini connection test failed: {e}")
            return False