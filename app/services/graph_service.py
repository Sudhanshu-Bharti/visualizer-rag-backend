from typing import List, Dict, Any, Optional
import logging
import hashlib
import json
import time
from functools import wraps
from app.core.database import neo4j_driver
from app.models.schemas import GraphData, GraphNode, GraphEdge
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


# Simple in-memory cache with TTL
class TTLCache:
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.ttl = ttl_seconds

    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        key_data = json.dumps(
            {"args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = (value, time.time())

    def clear(self) -> None:
        self.cache.clear()


def cache_graph_query(ttl_seconds: int = 300):
    """Decorator for caching graph queries"""
    cache = TTLCache(ttl_seconds)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = cache._get_cache_key(*args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache.set(cache_key, result)
            logger.info(f"Cache miss for {func.__name__} - cached result")
            return result

        # Add cache management methods
        wrapper.cache = cache
        wrapper.clear_cache = cache.clear
        return wrapper

    return decorator


class GraphService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    # @cache_graph_query(ttl_seconds=300)  # Cache disabled - using frontend React Query cache instead
    async def get_graph(
        self,
        node_limit: int = 100,
        include_embeddings: bool = False,
        node_types: Optional[List[str]] = None,
        document_id: Optional[str] = None,
        include_chunks: bool = False,
    ) -> GraphData:
        logger.info(
            f"get_graph called with node_limit={node_limit}, node_types={node_types}, include_chunks={include_chunks}"
        )

        # Ensure connection is healthy
        if not neo4j_driver.driver:
            logger.info("Initializing Neo4j driver")
            neo4j_driver.initialize()
        else:
            # Test existing connection, reinitialize if needed
            try:
                with neo4j_driver.driver.session() as test_session:
                    test_session.run("RETURN 1").single()
                logger.info("Neo4j connection test successful")
            except Exception as e:
                logger.warning(f"Neo4j connection test failed: {e}, reinitializing...")
                neo4j_driver.close()
                neo4j_driver.initialize()

        driver = neo4j_driver.get_driver()
        logger.info("Got Neo4j driver, creating session...")

        with driver.session() as session:
            node_type_filter = ""
            if node_types:
                labels = " OR ".join([f"n:{node_type}" for node_type in node_types])
                node_type_filter = f"WHERE {labels}"

            document_filter = ""
            document_param = None
            if document_id:
                document_param = document_id
                if node_type_filter:
                    document_filter = " AND EXISTS((n)<-[:HAS_CHUNK|HAS_ENTITY]-(:Document {id: $document_id}))"
                else:
                    document_filter = "WHERE EXISTS((n)<-[:HAS_CHUNK|HAS_ENTITY]-(:Document {id: $document_id}))"

            embedding_field = ", n.embedding as embedding" if include_embeddings else ""

            # Exclude Chunk nodes from visualization unless explicitly included
            chunk_exclusion = ""
            if not include_chunks:
                if not node_type_filter:
                    chunk_exclusion = "WHERE NOT n:Chunk"
                elif not any("Chunk" in t for t in (node_types or [])):
                    node_type_filter = node_type_filter + " AND NOT n:Chunk"

            node_query = f"""
                MATCH (n)
                {node_type_filter or chunk_exclusion}
                {document_filter}
                RETURN n.id as id, labels(n)[0] as type, n as properties{embedding_field}
                LIMIT {node_limit}
            """

            logger.info(f"Executing node query: {node_query}")
            query_params = {}
            if document_param:
                query_params["document_id"] = document_param
            nodes_result = session.run(node_query, **query_params)
            nodes = []
            node_ids = set()
            record_count = 0

            for record in nodes_result:
                record_count += 1
                node_id = record["id"]
                node_ids.add(node_id)

                properties = dict(record["properties"])
                if not include_embeddings and "embedding" in properties:
                    del properties["embedding"]

                # Convert Neo4j DateTime objects to native Python datetime
                for key, value in list(properties.items()):
                    if hasattr(value, "to_native"):
                        properties[key] = value.to_native()

                node = GraphNode(
                    id=node_id,
                    label=properties.get("name", properties.get("filename", node_id)),
                    type=record["type"],
                    properties=properties,
                )
                nodes.append(node)

            if node_ids:
                edge_query = """
                    MATCH (s)-[r]->(t)
                    WHERE s.id IN $node_ids AND t.id IN $node_ids
                      AND s.id <> t.id
                    RETURN s.id as source, t.id as target, type(r) as type, r as properties
                """

                edges_result = session.run(edge_query, node_ids=list(node_ids))
                edges = []

                for record in edges_result:
                    edge_properties = (
                        dict(record["properties"]) if record["properties"] else {}
                    )
                    # Convert Neo4j DateTime objects to ISO format strings
                    for key, value in list(edge_properties.items()):
                        if hasattr(value, "to_native"):
                            edge_properties[key] = value.to_native().isoformat()
                        elif hasattr(value, "isoformat"):  # datetime objects
                            edge_properties[key] = value.isoformat()

                    edge = GraphEdge(
                        id=f"{record['source']}_{record['target']}_{record['type']}",
                        source=record["source"],
                        target=record["target"],
                        type=record["type"],
                        properties=edge_properties,
                    )
                    edges.append(edge)
            else:
                edges = []

            logger.info(
                f"Found {record_count} records, returning {len(nodes)} nodes and {len(edges)} edges"
            )
            result = GraphData(nodes=nodes, edges=edges)
            logger.info(
                f"Returning GraphData with first node: {nodes[0].id if nodes else 'no nodes'}"
            )
            return result

    async def query_graph(
        self, query: str, node_limit: int = 100, include_embeddings: bool = False
    ) -> GraphData:
        driver = neo4j_driver.get_driver()

        try:
            with driver.session() as session:
                result = session.run(query)

                nodes = []
                edges = []
                node_ids = set()

                for record in result:
                    for key, value in record.items():
                        if hasattr(value, "labels") and hasattr(value, "id"):
                            node_id = value.get("id", str(value.id))
                            node_ids.add(node_id)

                            properties = dict(value)
                            if not include_embeddings and "embedding" in properties:
                                del properties["embedding"]

                            node = GraphNode(
                                id=node_id,
                                label=properties.get(
                                    "name", properties.get("filename", node_id)
                                ),
                                type=list(value.labels)[0]
                                if value.labels
                                else "Unknown",
                                properties=properties,
                            )
                            nodes.append(node)

                if len(nodes) > node_limit:
                    nodes = nodes[:node_limit]
                    node_ids = {n.id for n in nodes}

                if node_ids:
                    edge_query = """
                        MATCH (s)-[r]->(t)
                        WHERE s.id IN $node_ids AND t.id IN $node_ids
                          AND s.id <> t.id
                        RETURN s.id as source, t.id as target, type(r) as type, r as properties
                    """

                    edges_result = session.run(edge_query, node_ids=list(node_ids))

                    for record in edges_result:
                        edge = GraphEdge(
                            id=f"{record['source']}_{record['target']}_{record['type']}",
                            source=record["source"],
                            target=record["target"],
                            type=record["type"],
                            properties=dict(record["properties"])
                            if record["properties"]
                            else {},
                        )
                        edges.append(edge)

                return GraphData(nodes=nodes, edges=edges)

        except Exception as e:
            logger.error(f"Error executing graph query: {e}")
            return await self.get_graph(node_limit, include_embeddings)

    @cache_graph_query(ttl_seconds=600)  # Cache search for 10 minutes
    async def search_nodes(
        self, query: str, node_types: Optional[List[str]] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            type_filter = ""
            if node_types:
                labels = " OR ".join([f"n:{node_type}" for node_type in node_types])
                type_filter = f"AND ({labels})"

            search_query = f"""
                MATCH (n)
                WHERE (n.name CONTAINS $query OR n.description CONTAINS $query OR n.filename CONTAINS $query)
                {type_filter}
                RETURN n.id as id, n.name as name, labels(n)[0] as type, n.description as description
                LIMIT {limit}
            """

            result = session.run(search_query, query=query.lower())

            nodes = []
            for record in result:
                nodes.append(
                    {
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "description": record["description"],
                    }
                )

            return nodes

    async def get_node_neighbors(
        self,
        node_id: str,
        max_hops: int = 2,
        relationship_types: Optional[List[str]] = None,
    ) -> GraphData:
        driver = neo4j_driver.get_driver()

        with driver.session() as session:
            rel_filter = ""
            if relationship_types:
                rel_types = "|".join(relationship_types)
                rel_filter = f":{rel_types}"

            neighbor_query = f"""
                MATCH path = (start {{id: $node_id}})-[r{rel_filter}*1..{max_hops}]-(neighbor)
                UNWIND nodes(path) as n
                UNWIND relationships(path) as rel
                RETURN DISTINCT n.id as node_id, labels(n)[0] as node_type, n as node_props,
                       startNode(rel).id as rel_source, endNode(rel).id as rel_target, 
                       type(rel) as rel_type, rel as rel_props
            """

            result = session.run(neighbor_query, node_id=node_id)

            nodes = []
            edges = []
            seen_nodes = set()
            seen_edges = set()

            for record in result:
                node_id = record["node_id"]
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)

                    properties = dict(record["node_props"])
                    if "embedding" in properties:
                        del properties["embedding"]

                    node = GraphNode(
                        id=node_id,
                        label=str(
                            properties.get("name")
                            or properties.get("filename")
                            or node_id
                        ),
                        type=record["node_type"],
                        properties=properties,
                    )
                    nodes.append(node)

                if record["rel_source"] and record["rel_target"]:
                    edge_id = f"{record['rel_source']}_{record['rel_target']}_{record['rel_type']}"
                    if edge_id not in seen_edges:
                        seen_edges.add(edge_id)

                        edge = GraphEdge(
                            id=edge_id,
                            source=record["rel_source"],
                            target=record["rel_target"],
                            type=record["rel_type"],
                            properties=dict(record["rel_props"])
                            if record["rel_props"]
                            else {},
                        )
                        edges.append(edge)

            return GraphData(nodes=nodes, edges=edges)

    async def get_subgraph_for_context(
        self, context_items: List[Dict[str, Any]], max_nodes: int = 50
    ) -> Optional[Dict[str, Any]]:
        if not context_items:
            return None

        try:
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
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            # Return mock graph data for demonstration
            return {
                "nodes": [
                    {
                        "id": "demo_document",
                        "label": "Demo Document",
                        "type": "Document",
                        "highlighted": True,
                        "properties": {"filename": "Sample Document", "type": "demo"},
                    },
                    {
                        "id": "demo_entity",
                        "label": "AI Technology",
                        "type": "Entity",
                        "highlighted": False,
                        "properties": {"name": "AI Technology", "type": "CONCEPT"},
                    },
                ],
                "edges": [
                    {
                        "id": "demo_document_demo_entity_HAS_ENTITY",
                        "source": "demo_document",
                        "target": "demo_entity",
                        "type": "HAS_ENTITY",
                        "properties": {},
                    }
                ],
            }

        with driver.session() as session:
            context_node_ids = []
            for item in context_items:
                if "node_id" in item:
                    context_node_ids.append(item["node_id"])
                elif "chunk_id" in item:
                    context_node_ids.append(item["chunk_id"])

            if not context_node_ids:
                return None

            subgraph_query = f"""
                MATCH (n)-[r*0..2]-(m)
                WHERE n.id IN $context_node_ids
                RETURN DISTINCT n.id as node_id, labels(n)[0] as node_type, n as node_props,
                       m.id as connected_id, labels(m)[0] as connected_type, m as connected_props
                LIMIT {max_nodes * 2}
            """

            result = session.run(subgraph_query, context_node_ids=context_node_ids)

            nodes = []
            seen_nodes = set()

            for record in result:
                for node_key, type_key, props_key in [
                    ("node_id", "node_type", "node_props"),
                    ("connected_id", "connected_type", "connected_props"),
                ]:
                    node_id = record[node_key]
                    if node_id and node_id not in seen_nodes:
                        seen_nodes.add(node_id)

                        properties = dict(record[props_key])
                        if "embedding" in properties:
                            del properties["embedding"]

                        # Convert Neo4j DateTime objects to ISO format strings
                        for key, value in list(properties.items()):
                            if hasattr(value, "to_native"):
                                properties[key] = value.to_native().isoformat()
                            elif hasattr(value, "isoformat"):  # datetime objects
                                properties[key] = value.isoformat()

                        is_highlighted = node_id in context_node_ids

                        nodes.append(
                            {
                                "id": node_id,
                                "label": properties.get(
                                    "name", properties.get("filename", node_id)
                                ),
                                "type": record[type_key],
                                "highlighted": is_highlighted,
                                "properties": properties,
                            }
                        )

                        if len(nodes) >= max_nodes:
                            break

                if len(nodes) >= max_nodes:
                    break

            node_ids = [n["id"] for n in nodes]
            if node_ids:
                edge_query = """
                    MATCH (s)-[r]->(t)
                    WHERE s.id IN $node_ids AND t.id IN $node_ids
                      AND s.id <> t.id
                    RETURN s.id as source, t.id as target, type(r) as type, r as properties
                """

                edges_result = session.run(edge_query, node_ids=node_ids)
                edges = []

                for record in edges_result:
                    edge_properties = (
                        dict(record["properties"]) if record["properties"] else {}
                    )

                    # Convert Neo4j DateTime objects to ISO format strings
                    for key, value in list(edge_properties.items()):
                        if hasattr(value, "to_native"):
                            edge_properties[key] = value.to_native().isoformat()
                        elif hasattr(value, "isoformat"):  # datetime objects
                            edge_properties[key] = value.isoformat()

                    edges.append(
                        {
                            "id": f"{record['source']}_{record['target']}_{record['type']}",
                            "source": record["source"],
                            "target": record["target"],
                            "type": record["type"],
                            "properties": edge_properties,
                        }
                    )
            else:
                edges = []

            return {"nodes": nodes, "edges": edges}

    @cache_graph_query(ttl_seconds=900)  # Cache stats for 15 minutes
    async def get_graph_statistics(self) -> Dict[str, Any]:
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
            stats_query = """
                MATCH (n)
                WITH labels(n)[0] as label, count(n) as node_count
                RETURN label, node_count
                ORDER BY node_count DESC
            """

            node_stats = session.run(stats_query)

            relationship_stats_query = """
                MATCH ()-[r]->()
                WITH type(r) as rel_type, count(r) as rel_count
                RETURN rel_type, rel_count
                ORDER BY rel_count DESC
            """

            rel_stats = session.run(relationship_stats_query)

            total_nodes_query = "MATCH (n) RETURN count(n) as total"
            total_edges_query = "MATCH ()-[r]->() RETURN count(r) as total"

            total_nodes = session.run(total_nodes_query).single()["total"]
            total_edges = session.run(total_edges_query).single()["total"]

            node_type_counts = {}
            for record in node_stats:
                node_type_counts[record["label"]] = record["node_count"]

            rel_type_counts = {}
            for record in rel_stats:
                rel_type_counts[record["rel_type"]] = record["rel_count"]

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_type_counts": node_type_counts,
                "relationship_type_counts": rel_type_counts,
            }
