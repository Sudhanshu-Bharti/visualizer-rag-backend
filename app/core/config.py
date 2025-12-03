from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    aura_instanceid: str = ""
    aura_instancename: str = ""

    # LLM Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "qwen2.5:7b-instruct"
    gemini_api_key: str = ""

    # Embedding Configuration (Memory Optimized)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model (80MB)
    embedding_dimension: int = 384

    # Upload Configuration (Memory Limits)
    upload_max_size_mb: int = 10  # Reduced from 50MB to 10MB
    max_chunk_size: int = 500     # Reduced from 1000 to 500
    chunk_overlap: int = 100      # Reduced from 200 to 100
    
    # Memory Management
    max_batch_size: int = 5       # Maximum batch size for processing
    enable_memory_optimization: bool = True

    # CORS configuration
    allowed_origins: str = "http://localhost:3000,http://localhost:3006,visualizer-ivory-beta.vercel.app"

    class Config:
        env_file = ".env"


settings = Settings()
