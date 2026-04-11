# src/healthcare_rag_llm/graph_builder/neo4j_loader.py
import os
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable
from dotenv import load_dotenv

# Load Docker .env
load_dotenv("docker/.env")

class Neo4jConnector:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")

        # NEO4J_AUTH
        auth = os.getenv("NEO4J_AUTH")
        if auth and "/" in auth:
            user_env, password_env = auth.split("/", 1)
        else:
            user_env = os.getenv("NEO4J_USERNAME", "neo4j")
            password_env = os.getenv("NEO4J_PASSWORD")

        self.user = user or user_env
        self.password = password or password_env

        if not self.password:
            raise ValueError("ERROR: Neo4j password not set. Check your .env file.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._verify_connection()

    def _verify_connection(self):
        try:
            self.driver.verify_connectivity()
        except AuthError as exc:
            self.driver.close()
            raise RuntimeError(
                "Failed to authenticate with Neo4j. Check NEO4J_AUTH or the "
                "NEO4J_USERNAME/NEO4J_PASSWORD environment variables."
            ) from exc
        except ServiceUnavailable as exc:
            self.driver.close()
            raise RuntimeError(
                "Neo4j is not reachable at "
                f"{self.uri}. This project defaults to a local Docker Neo4j instance. "
                "Start it from the repo's docker directory with `docker compose up -d`, "
                "or set NEO4J_URI to a running Bolt endpoint before rerunning."
            ) from exc

    def close(self):
        self.driver.close()

    def init_schema(self):
        """Initialize Neo4j constraints and vector index"""
        with self.driver.session() as session:
            # Unique constraints
            session.run("CREATE CONSTRAINT authority_name_unique IF NOT EXISTS "
                        "FOR (a:Authority) REQUIRE a.name IS UNIQUE")
            session.run("CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
                        "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            session.run("CREATE CONSTRAINT page_uid_unique IF NOT EXISTS "
                        "FOR (p:Page) REQUIRE p.uid IS UNIQUE")
            session.run("CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")

            # Vector index for dense embedding (BGE-M3 → 1024 dims)
            session.run("""
            CREATE VECTOR INDEX chunk_vec IF NOT EXISTS
            FOR (c:Chunk) ON (c.denseEmbedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 1024,
              `vector.similarity_function`: 'cosine'
            }}
            """)

            # Create Separate Vector Index for different doc_class

            # session.run("""
            # CREATE VECTOR INDEX policy_chunk_vec IF NOT EXISTS
            # FOR (c:PolicyChunk) ON (c.denseEmbedding)
            # OPTIONS {indexConfig: {
            #   `vector.dimensions`: 1024,
            #   `vector.similarity_function`: 'cosine'
            # }}
            # """)
            # session.run("""
            # CREATE VECTOR INDEX manual_chunk_vec IF NOT EXISTS
            # FOR (c:ManualChunk) ON (c.denseEmbedding)
            # OPTIONS {indexConfig: {
            #   `vector.dimensions`: 1024,
            #   `vector.similarity_function`: 'cosine'
            # }}
            # """)
            
            print("✅ Neo4j schema initialized (Authority, Document, Page, Chunk, vector index).")
