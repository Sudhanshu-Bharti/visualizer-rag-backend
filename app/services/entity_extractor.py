import re
from typing import List, Dict, Any, Tuple, Optional
import logging
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(self):
        self.ollama_base_url = settings.ollama_base_url

    async def extract_entities_and_relations(
        self, text: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        try:
            entities = await self._extract_entities_with_llm(text)

            if not entities:
                entities = self._extract_entities_rule_based(text)

            relationships = await self._extract_relationships_with_llm(text, entities)

            return entities, relationships

        except Exception as e:
            logger.warning(f"LLM extraction failed, falling back to rule-based: {e}")
            entities = self._extract_entities_rule_based(text)
            relationships = self._extract_relationships_rule_based(text, entities)
            return entities, relationships

    async def _extract_entities_with_llm(self, text: str) -> List[Dict[str, Any]]:
        try:
            prompt = f"""Extract entities from the following text. For each entity, provide:
1. Name (the exact text)
2. Type (PERSON, ORGANIZATION, LOCATION, CONCEPT, etc.)
3. Description (brief context from the text)

Text: {text[:2000]}

Return in this JSON format:
[{{"name": "Entity Name", "type": "TYPE", "description": "Context description"}}]

Only return the JSON array, no other text."""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 1000},
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    entities_text = result.get("response", "")

                    try:
                        import json

                        entities_data = json.loads(entities_text)

                        if isinstance(entities_data, list):
                            return [
                                {
                                    "name": entity.get("name", ""),
                                    "type": entity.get("type", "UNKNOWN"),
                                    "description": entity.get("description", ""),
                                }
                                for entity in entities_data
                                if entity.get("name")
                            ]
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse LLM entity extraction response")

        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")

        return []

    async def _extract_relationships_with_llm(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if len(entities) < 2:
            return []

        try:
            entity_names = [e["name"] for e in entities]

            prompt = f"""Given these entities: {", ".join(entity_names[:10])}

Identify relationships between them based on this text: {text[:1500]}

Return relationships in this JSON format:
[{{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE", "description": "How they are related"}}]

Common relationship types: WORKS_FOR, LOCATED_IN, PART_OF, RELATED_TO, MENTIONS, etc.

Only return the JSON array, no other text."""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": settings.ollama_model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 800},
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    relationships_text = result.get("response", "")

                    try:
                        import json

                        relationships_data = json.loads(relationships_text)

                        if isinstance(relationships_data, list):
                            valid_relationships = []
                            for rel in relationships_data:
                                source = rel.get("source", "")
                                target = rel.get("target", "")

                                if (
                                    source in entity_names
                                    and target in entity_names
                                    and source != target
                                ):
                                    valid_relationships.append(
                                        {
                                            "source": source,
                                            "target": target,
                                            "type": rel.get("type", "RELATED_TO"),
                                            "description": rel.get("description", ""),
                                        }
                                    )

                            return valid_relationships

                    except json.JSONDecodeError:
                        logger.warning(
                            "Failed to parse LLM relationship extraction response"
                        )

        except Exception as e:
            logger.warning(f"LLM relationship extraction failed: {e}")

        return []

    def _extract_entities_rule_based(self, text: str) -> List[Dict[str, Any]]:
        entities = []

        # Technical and concept patterns for AI/tech documents
        technology_patterns = [
            # Database and storage
            (
                r"\b(Neo4j|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch|Pinecone|Weaviate|Milvus|Chroma|ArangoDB|Amazon Neptune)\b",
                "TECHNOLOGY",
            ),
            # AI/ML frameworks and models
            (
                r"\b(GPT-?[0-9](?:\.[0-9])?|Llama ?[0-9](?:\.[0-9])?|Qwen ?[0-9](?:\.[0-9])?|Mistral|BERT|Claude|ChatGPT|Code ?Llama|CLIP|DALL-?E|Stable Diffusion|Midjourney|Whisper)\b",
                "MODEL",
            ),
            # Programming and frameworks
            (
                r"\b(Python|JavaScript|TypeScript|PyTorch|TensorFlow|JAX|FastAPI|Next\.js|React|Vue|Transformers|Cytoscape\.js|D3\.js)\b",
                "TECHNOLOGY",
            ),
            # AI/ML concepts
            (
                r"\b(RAG|Graph RAG|Retrieval[- ]Augmented Generation|vector (?:embedding|database|search|index)|knowledge graph|HNSW|similarity search|semantic search)\b",
                "CONCEPT",
            ),
            (
                r"\b(NER|Named Entity Recognition|entity (?:extraction|recognition)|relationship extraction|knowledge graph construction)\b",
                "CONCEPT",
            ),
            # Technical terms
            (
                r"\b(API|REST|GraphQL|microservice|containerization|Docker|Kubernetes)\b",
                "TECHNOLOGY",
            ),
        ]

        # Extract technical entities
        for pattern, entity_type in technology_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(1) if match.groups() else match.group()
                entities.append(
                    {
                        "name": entity_name,
                        "type": entity_type,
                        "description": f"{entity_type.capitalize()} mentioned in text",
                    }
                )

        # Traditional NER patterns
        person_patterns = [
            r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
            r"\b(?:Mr|Ms|Mrs|Dr|Prof)\. [A-Z][a-z]+\b",
        ]

        organization_patterns = [
            r"\b(?:OpenAI|Google|Meta|Microsoft|Amazon|Anthropic|Hugging Face|Nvidia)\b",
            r"\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization|University|Institute)\b",
            r"\b(?:University|Institute|Corporation|Company) of [A-Z][a-z]+\b",
        ]

        location_patterns = [
            r"\b[A-Z][a-z]+(?:, [A-Z][a-z]+)*\b(?= (?:City|State|Country|Street|Avenue|Road))",
            r"\b(?:New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose)\b",
        ]

        for pattern in person_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    {
                        "name": match.group(),
                        "type": "PERSON",
                        "description": "Person mentioned in text",
                    }
                )

        for pattern in organization_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    {
                        "name": match.group(),
                        "type": "ORGANIZATION",
                        "description": "Organization mentioned in text",
                    }
                )

        for pattern in location_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(
                    {
                        "name": match.group(),
                        "type": "LOCATION",
                        "description": "Location mentioned in text",
                    }
                )

        # Extract capitalized noun phrases as potential concepts
        noun_phrase_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b"
        matches = re.finditer(noun_phrase_pattern, text)
        for match in matches:
            phrase = match.group(1)
            # Only add if it's not already captured and looks like a concept
            if len(phrase.split()) >= 2 and not any(
                phrase == e["name"] for e in entities
            ):
                entities.append(
                    {
                        "name": phrase,
                        "type": "CONCEPT",
                        "description": "Concept or topic mentioned in text",
                    }
                )

        # Deduplicate by normalizing names (case-insensitive)
        seen_names = {}
        unique_entities = []
        for entity in entities:
            normalized_name = entity["name"].lower()
            if normalized_name not in seen_names:
                seen_names[normalized_name] = entity["name"]
                unique_entities.append(entity)

        logger.info(
            f"Extracted {len(unique_entities)} unique entities using rule-based approach"
        )
        return unique_entities[:100]  # Increased limit for better coverage

    def _extract_relationships_rule_based(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        relationships = []
        entity_names = [e["name"] for e in entities]

        relationship_patterns = [
            (r"(.+?) works for (.+?)", "WORKS_FOR"),
            (r"(.+?) is located in (.+?)", "LOCATED_IN"),
            (r"(.+?) and (.+?) collaborated", "COLLABORATED_WITH"),
            (r"(.+?) founded (.+?)", "FOUNDED"),
            (r"(.+?) is part of (.+?)", "PART_OF"),
        ]

        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source = match.group(1).strip()
                target = match.group(2).strip()

                source_entity = self._find_entity_match(source, entity_names)
                target_entity = self._find_entity_match(target, entity_names)

                if source_entity and target_entity and source_entity != target_entity:
                    relationships.append(
                        {
                            "source": source_entity,
                            "target": target_entity,
                            "type": rel_type,
                            "description": f"Extracted from: {match.group()}",
                        }
                    )

        for i, entity1 in enumerate(entity_names):
            for entity2 in entity_names[i + 1 :]:
                distance = text.find(entity2) - text.find(entity1)
                if 0 < abs(distance) < 200:
                    relationships.append(
                        {
                            "source": entity1,
                            "target": entity2,
                            "type": "MENTIONED_TOGETHER",
                            "description": "Entities mentioned close together in text",
                        }
                    )

        return relationships[:30]

    def _find_entity_match(self, text: str, entity_names: List[str]) -> Optional[str]:
        text_lower = text.lower()
        for entity_name in entity_names:
            if entity_name.lower() in text_lower:
                return entity_name
        return None
