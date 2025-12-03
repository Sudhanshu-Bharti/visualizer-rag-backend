from typing import AsyncGenerator, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

from app.api import documents, chat, graph
from app.core.config import settings
from app.core.database import neo4j_driver
from app.utils.neo4j_serialization import serialize_neo4j_data

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    try:
        neo4j_driver.initialize()
        print("Neo4j initialized successfully")
    except Exception as e:
        print(f"Warning: Neo4j initialization failed: {e}")
    yield
    # Shutdown
    neo4j_driver.close()


class CustomJSONResponse(JSONResponse):
    """Custom JSON response that handles Neo4j DateTime objects"""

    def render(self, content: Any) -> bytes:
        content = serialize_neo4j_data(content)
        return super().render(content)


app = FastAPI(
    title="RAG Visualization API",
    description="Graph RAG visualization and parameter tuning platform",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=CustomJSONResponse,
)

# CORS configuration - use environment variable for allowed origins
allowed_origins_list = [
    origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()
]

# For production deployment, allow all origins if not specified
if os.getenv("DEPLOYMENT_ENV") in ["vercel", "render", "production"]:
    if not allowed_origins_list or allowed_origins_list == [""]:
        allowed_origins_list = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_list,
    allow_credentials=True if "*" not in allowed_origins_list else False,  # Can't use credentials with wildcard
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Graph RAG Visualization API"}


@app.get("/test")
async def test() -> Dict[str, str]:
    return {"message": "Test endpoint working"}


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    try:
        # Use improved connectivity verification
        if not neo4j_driver.verify_connectivity():
            # Try to reinitialize connection
            try:
                neo4j_driver.close()
                neo4j_driver.initialize()
            except Exception as reinit_error:
                return {
                    "status": "degraded",
                    "neo4j": "disconnected",
                    "error": f"Reinitialize failed: {str(reinit_error)}",
                }

        # Get connection pool statistics
        connection_stats = neo4j_driver.get_connection_stats()

        return {
            "status": "healthy",
            "neo4j": "connected",
            "database": settings.neo4j_database,
            "connection_pool": connection_stats,
        }
    except Exception as e:
        return {"status": "degraded", "neo4j": "disconnected", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
