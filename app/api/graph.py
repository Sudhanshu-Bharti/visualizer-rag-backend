from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.schemas import GraphData, GraphQueryRequest
from app.services.graph_service import GraphService

router = APIRouter()
graph_service = GraphService()


@router.get("/", response_model=GraphData)
async def get_graph(
    node_limit: int = Query(100, ge=10, le=1000),
    include_embeddings: bool = Query(False),
    node_types: Optional[str] = Query(None, description="Comma-separated node types"),
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    include_chunks: bool = Query(
        False, description="Include chunk nodes in visualization"
    ),
):
    try:
        types_list = node_types.split(",") if node_types else None

        graph_data = await graph_service.get_graph(
            node_limit=node_limit,
            include_embeddings=include_embeddings,
            node_types=types_list,
            document_id=document_id,
            include_chunks=include_chunks,
        )

        return graph_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=GraphData)
async def query_graph(request: GraphQueryRequest):
    try:
        if request.query:
            graph_data = await graph_service.query_graph(
                query=request.query,
                node_limit=request.node_limit,
                include_embeddings=request.include_embeddings,
            )
        else:
            graph_data = await graph_service.get_graph(
                node_limit=request.node_limit,
                include_embeddings=request.include_embeddings,
            )

        return graph_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_graph_stats():
    try:
        stats = await graph_service.get_graph_statistics()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_nodes(
    q: str = Query(..., description="Search query"),
    node_types: Optional[str] = Query(None, description="Comma-separated node types"),
    limit: int = Query(20, ge=1, le=100),
):
    try:
        types_list = node_types.split(",") if node_types else None

        results = await graph_service.search_nodes(
            query=q, node_types=types_list, limit=limit
        )

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neighbors/{node_id}")
async def get_node_neighbors(
    node_id: str,
    max_hops: int = Query(2, ge=1, le=5),
    relationship_types: Optional[str] = Query(
        None, description="Comma-separated relationship types"
    ),
):
    try:
        rel_types_list = relationship_types.split(",") if relationship_types else None

        neighbors = await graph_service.get_node_neighbors(
            node_id=node_id, max_hops=max_hops, relationship_types=rel_types_list
        )

        return neighbors

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_graph_cache():
    """Clear all graph query caches"""
    try:
        # Clear caches for methods that have them
        if hasattr(graph_service.get_graph, "clear_cache"):
            graph_service.get_graph.clear_cache()
        if hasattr(graph_service.search_nodes, "clear_cache"):
            graph_service.search_nodes.clear_cache()
        if hasattr(graph_service.get_graph_statistics, "clear_cache"):
            graph_service.get_graph_statistics.clear_cache()

        return {"message": "Graph cache cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
