"""
Face recognition utilities.

Handles face detection, embedding extraction, and matching.
Currently uses face_recognition (dlib-based) for 128D embeddings.

For production upgrades:
- RetinaFace/MediaPipe: Faster, more accurate detection
- ArcFace/InsightFace: 512D embeddings with better accuracy
- ONNX/TensorRT: GPU acceleration
"""

import cv2
import numpy as np
import face_recognition
from typing import Optional, Tuple
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)


def extract_face_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Extract face embedding from an image.
    
    Args:
        image_bytes: Image file as bytes (JPEG, PNG, etc.)
        
    Returns:
        128-dimensional face embedding as numpy array, or None if no face found
        
    TODO: Upgrade to ArcFace/InsightFace for 512D embeddings:
        import insightface
        app = insightface.app.FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))
        # Returns 512D embeddings with better accuracy
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB (face_recognition expects RGB)
        rgb_image = np.array(image.convert('RGB'))
        
        # Detect faces
        # TODO: Upgrade to RetinaFace for better detection:
        # from retinaface import RetinaFace
        # faces = RetinaFace.detect_faces(rgb_image)
        # Faster and more accurate, especially for small faces
        
        # TODO: Upgrade to MediaPipe for mobile/edge devices:
        # import mediapipe as mp
        # mp_face_detection = mp.solutions.face_detection
        # Very fast, optimized for mobile devices
        
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        
        if len(face_locations) == 0:
            logger.warning("No face detected in image")
            return None
        
        # Use the first face found
        # For multiple faces, you might want to return all or the largest
        face_location = face_locations[0]
        
        # Extract 128-dimensional embedding
        # TODO: Upgrade to ArcFace for 512D embeddings:
        # embeddings = face_recognition.face_encodings(rgb_image, [face_location])
        # For now, using standard face_recognition (128D)
        embeddings = face_recognition.face_encodings(rgb_image, [face_location])
        
        if len(embeddings) == 0:
            logger.warning("Could not extract embedding from detected face")
            return None
        
        embedding = embeddings[0]
        
        # Ensure it's a numpy array and has the right shape
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Verify dimension (should be 128 for face_recognition)
        if len(embedding) != 128:
            logger.error(f"Unexpected embedding dimension: {len(embedding)}")
            return None
        
        logger.info(f"Successfully extracted {len(embedding)}D embedding")
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting face embedding: {e}")
        return None


def validate_image(image_bytes: bytes) -> bool:
    """
    Validate that the image can be loaded and processed.
    
    Args:
        image_bytes: Image file as bytes
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify it's a valid image
        return True
    except Exception as e:
        logger.error(f"Invalid image: {e}")
        return False


def embedding_to_list(embedding: np.ndarray) -> list:
    """
    Convert numpy array embedding to Python list for JSON serialization.
    
    Args:
        embedding: Numpy array embedding
        
    Returns:
        List of floats
    """
    return embedding.tolist()


def list_to_embedding(embedding_list: list) -> np.ndarray:
    """
    Convert Python list to numpy array embedding.
    
    Args:
        embedding_list: List of floats
        
    Returns:
        Numpy array embedding
    """
    return np.array(embedding_list, dtype=np.float32)

