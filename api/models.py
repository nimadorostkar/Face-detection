"""
SQLAlchemy models for the face recognition database.

Uses pgvector for efficient vector similarity search.
"""

from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from pgvector.sqlalchemy import Vector
from utils.db import Base


class User(Base):
    """
    User model storing face embeddings.
    
    Uses pgvector's Vector type for efficient similarity search.
    Embedding dimension is 128 (face_recognition) but can be extended to 512 (ArcFace).
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    embedding = Column(Vector(128), nullable=False)  # 128D for face_recognition
    # TODO: For ArcFace/InsightFace, change to Vector(512)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}')>"

