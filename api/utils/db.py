"""
Database utilities for face recognition system.

Handles SQLAlchemy setup, pgvector extension, and database initialization.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    poolclass=NullPool,  # Use NullPool for better performance with pgvector
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def init_db():
    """
    Initialize database: enable pgvector extension and create tables.
    
    This function is called on application startup to ensure:
    1. pgvector extension is enabled
    2. Users table exists with proper schema
    3. Vector index is created for fast similarity search
    """
    logger.info("Initializing database...")
    
    try:
        with engine.connect() as conn:
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("✓ pgvector extension enabled")
            
            # Create users table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    embedding vector(128) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            logger.info("✓ Users table created/verified")
            
            # Create vector index for fast similarity search
            # Using ivfflat index with cosine distance for optimal performance
            # lists parameter: ~sqrt(rows) for good balance (100 for ~5000-10000 users)
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS users_embedding_idx
                    ON users USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = :lists)
                """), {"lists": settings.VECTOR_INDEX_LISTS})
                conn.commit()
                logger.info(f"✓ Vector index created with {settings.VECTOR_INDEX_LISTS} lists")
            except Exception as e:
                # Index might already exist or need to be recreated
                logger.warning(f"Index creation note: {e}")
                # Try to create without specifying lists (uses default)
                try:
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS users_embedding_idx
                        ON users USING ivfflat (embedding vector_cosine_ops)
                    """))
                    conn.commit()
                    logger.info("✓ Vector index created with default settings")
                except Exception as e2:
                    logger.warning(f"Could not create index: {e2}")
                    logger.info("Continuing without index (slower queries)")
            
            # Get current user count
            result = conn.execute(text("SELECT COUNT(*) FROM users"))
            count = result.scalar()
            logger.info(f"✓ Database initialized. Current users: {count}")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def get_db():
    """
    Dependency function for FastAPI to get database session.
    
    Yields a database session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

