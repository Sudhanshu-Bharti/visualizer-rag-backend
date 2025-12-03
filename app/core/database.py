from typing import Optional, Dict, Any
from neo4j import GraphDatabase, Driver
from app.core.config import settings
import logging
import ssl

logger = logging.getLogger(__name__)


class Neo4jDriver:
    def __init__(self) -> None:
        self.driver: Optional[Driver] = None

    def initialize(self) -> None:
        try:
            # Configuration for Neo4j Aura
            # Note: neo4j+s:// URI scheme handles SSL automatically
            config = {
                "max_connection_lifetime": 30 * 60,  # 30 minutes
                "max_connection_pool_size": 10,  # Suitable for Aura Free
                "connection_acquisition_timeout": 30,  # 30 second timeout
            }

            logger.info(f"Attempting to connect to Neo4j at: {settings.neo4j_uri}")
            logger.info(f"Using username: {settings.neo4j_username}")

            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
                **config,
            )

            with self.driver.session() as session:
                session.run("RETURN 1")

            self._create_indexes()
            logger.info("Neo4j driver initialized successfully")

        except Exception as e:
            error_msg = str(e)
            if "Unable to retrieve routing information" in error_msg:
                logger.error("Neo4j Aura connection failed. Possible causes:")
                logger.error("1. Instance is paused - check Neo4j Aura Console")
                logger.error("2. Incorrect password - verify credentials")
                logger.error("3. Network/firewall issues")
                logger.error("4. Instance is not running")
            elif "CERTIFICATE_VERIFY_FAILED" in error_msg:
                logger.error("SSL certificate verification failed - this is common on Windows")
                logger.error("Your Neo4j Aura instance might be using self-signed certificates")
            else:
                logger.error(f"Neo4j connection failed with unexpected error: {e}")
            
            logger.error("Please check your Neo4j Aura instance status at https://console.neo4j.io/")
            raise

    def _create_indexes(self) -> None:
        with self.driver.session() as session:
            try:
                session.run(
                    """
                    CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                    FOR (d:Document) ON (d.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dimension,
                        `vector.similarity_function`: 'cosine'
                    }}
                """,
                    dimension=settings.embedding_dimension,
                )

                session.run(
                    """
                    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dimension,
                        `vector.similarity_function`: 'cosine'
                    }}
                """,
                    dimension=settings.embedding_dimension,
                )

                session.run(
                    """
                    CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dimension,
                        `vector.similarity_function`: 'cosine'
                    }}
                """,
                    dimension=settings.embedding_dimension,
                )

                session.run(
                    "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)"
                )
                session.run(
                    "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)"
                )
                session.run(
                    "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)"
                )

                logger.info("Neo4j indexes created successfully")

            except Exception as e:
                logger.warning(f"Some indexes may already exist: {e}")

    def get_driver(self) -> Driver:
        if not self.driver:
            self.initialize()
        return self.driver

    def verify_connectivity(self) -> bool:
        """Verify Neo4j connection health"""
        try:
            if not self.driver:
                return False

            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                return record["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        try:
            if self.driver and hasattr(self.driver, "_pool"):
                pool = self.driver._pool
                return {
                    "active_connections": getattr(pool, "size", 0),
                    "max_pool_size": getattr(pool, "max_size", 50),
                }
            return {"status": "no_pool_info"}
        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed")


neo4j_driver = Neo4jDriver()
