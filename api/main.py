"""
FastAPI application for real-time face recognition.

Provides REST API endpoints for:
- POST /register: Register a new person with face image
- POST /recognize: Recognize a face from an image

Designed to handle 5000+ users efficiently using PostgreSQL with pgvector.
"""

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np
from typing import Optional

from utils.db import init_db, get_db, engine
from utils.face_utils import extract_face_embedding, validate_image
from models import User
from utils.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Real-time face recognition system with PostgreSQL and pgvector",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on application startup."""
    logger.info("Starting Face Recognition API...")
    try:
        init_db()
        logger.info("✓ API ready")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /register",
            "recognize": "POST /recognize",
            "health": "GET /health",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint."""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "threshold": settings.FACE_MATCH_THRESHOLD,
            "embedding_dimension": settings.EMBEDDING_DIMENSION
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get database statistics."""
    try:
        user_count = db.query(User).count()
        return {
            "total_users": user_count,
            "threshold": settings.FACE_MATCH_THRESHOLD,
            "embedding_dimension": settings.EMBEDDING_DIMENSION
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting stats: {str(e)}"
        )


@app.post("/register")
async def register_person(
    name: str,
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Register a new person with their face image.
    
    Args:
        name: Person's name (must be unique)
        image: Face image file (JPEG, PNG, etc.)
        
    Returns:
        JSON response with registration status and user ID
    """
    try:
        # Validate name
        if not name or not name.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Name is required"
            )
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.name == name).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User '{name}' already exists"
            )
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Validate image
        if not validate_image(image_bytes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        # Extract face embedding
        embedding = extract_face_embedding(image_bytes)
        if embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in image. Please provide an image with a clear face."
            )
        
        # Convert numpy array to list for storage
        embedding_list = embedding.tolist()
        
        # Create new user
        new_user = User(
            name=name.strip(),
            embedding=embedding_list  # pgvector will handle the conversion
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"Registered new user: {name} (ID: {new_user.id})")
        
        return {
            "status": "success",
            "message": f"User '{name}' registered successfully",
            "user_id": new_user.id,
            "name": new_user.name,
            "embedding_dimension": len(embedding)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering user: {str(e)}"
        )


@app.post("/recognize")
async def recognize_face(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    threshold: Optional[float] = None
):
    """
    Recognize a face from an image.
    
    Uses pgvector's cosine distance operator (<->) for efficient similarity search.
    Returns the best match if distance is below threshold.
    
    Args:
        image: Face image file (JPEG, PNG, etc.)
        threshold: Optional custom threshold (defaults to configured value)
        
    Returns:
        JSON response with recognition result, distance, and match status
        
    TODO: For real-time webcam inference:
        - Add WebSocket endpoint for streaming video
        - Process frames in batches
        - Return results in real-time
        
    TODO: Add liveness detection:
        - Blink detection using facial landmarks
        - Depth/IR camera input
        - Challenge-response mechanisms
    """
    try:
        # Use provided threshold or default
        match_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD
        
        # Read image bytes
        image_bytes = await image.read()
        
        # Validate image
        if not validate_image(image_bytes):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        # Extract face embedding
        embedding = extract_face_embedding(image_bytes)
        if embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in image. Please provide an image with a clear face."
            )
        
        # Convert to list for pgvector query
        embedding_list = embedding.tolist()
        
        # Query database for best match using pgvector cosine distance
        # The <-> operator computes cosine distance (0 = identical, 1 = completely different)
        # We use ORDER BY ... LIMIT 1 for efficiency with the ivfflat index
        query = text("""
            SELECT id, name, embedding <-> :embedding::vector AS distance
            FROM users
            ORDER BY embedding <-> :embedding::vector
            LIMIT 1
        """)
        
        result = db.execute(query, {"embedding": str(embedding_list)})
        row = result.fetchone()
        
        if row is None:
            # No users in database
            return {
                "status": "no_match",
                "message": "No users registered in database",
                "distance": None,
                "matched_user": None
            }
        
        user_id, name, distance = row
        
        # Check if distance is below threshold
        if distance <= match_threshold:
            logger.info(f"Face recognized: {name} (distance: {distance:.4f})")
            return {
                "status": "match",
                "message": f"Face recognized as '{name}'",
                "distance": float(distance),
                "confidence": float(1.0 - distance),  # Convert distance to confidence
                "matched_user": {
                    "id": user_id,
                    "name": name
                },
                "threshold": match_threshold
            }
        else:
            logger.info(f"No match found (best distance: {distance:.4f}, threshold: {match_threshold})")
            return {
                "status": "no_match",
                "message": "No matching face found",
                "distance": float(distance),
                "confidence": float(1.0 - distance),
                "matched_user": None,
                "threshold": match_threshold
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recognizing face: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recognizing face: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )

