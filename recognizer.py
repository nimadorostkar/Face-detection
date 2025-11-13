"""
Face Recognition Module

This module handles loading known faces, encoding extraction, and matching.
Default implementation uses face_recognition library (dlib-based).

For production use, consider replacing with:
- ArcFace / InsightFace: State-of-the-art accuracy, better for large-scale recognition
- FaceNet: Good balance of speed and accuracy
- ONNX Runtime: For optimized inference with GPU support
- TensorRT: For NVIDIA GPU acceleration
"""

import os
import cv2
import face_recognition
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Optional database import
try:
    from database import FaceDatabase
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    FaceDatabase = None


class FaceRecognizer:
    """
    Face recognizer using face_recognition library (dlib-based).
    
    For production alternatives:
    - ArcFace: Use insightface library (pip install insightface)
    - FaceNet: Use facenet-pytorch or tensorflow
    - ONNX: Convert models to ONNX for optimized inference
    """
    
    def __init__(self, 
                 known_faces_dir: str = "known_faces", 
                 tolerance: float = 0.6,
                 use_database: bool = False,
                 db: FaceDatabase = None):
        """
        Initialize face recognizer.
        
        Args:
            known_faces_dir: Directory containing subdirectories with person names
                           Each subdirectory should contain images of that person
            tolerance: Distance threshold for face matching (lower = more strict)
                     Typical values: 0.4-0.6 for face_recognition
            use_database: If True, use database for storing/loading faces
            db: FaceDatabase instance (if None and use_database=True, will create one)
        """
        self.known_faces_dir = known_faces_dir
        self.tolerance = tolerance
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.use_database = use_database and DB_AVAILABLE
        self.db = db
        
        # Initialize database if needed
        if self.use_database and self.db is None:
            try:
                self.db = FaceDatabase()
                print("✓ Using database for face storage")
            except Exception as e:
                print(f"Warning: Could not connect to database: {e}")
                print("Falling back to file-based storage")
                self.use_database = False
        
        # TODO: Production - Initialize ArcFace/InsightFace model
        # import insightface
        # self.arcface_model = insightface.app.FaceAnalysis()
        # self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))
        
        # TODO: Production - Initialize ONNX model
        # import onnxruntime as ort
        # self.ort_session = ort.InferenceSession("face_recognition.onnx")
        
        # Load known faces
        self.load_known_faces()
    
    def load_known_faces(self):
        """
        Load all known faces from database or file system.
        
        If use_database is True, loads from database.
        Otherwise, loads from known_faces directory.
        """
        if self.use_database and self.db:
            self._load_from_database()
        else:
            self._load_from_files()
    
    def _load_from_database(self):
        """Load known faces from database."""
        try:
            encodings_data = self.db.get_all_encodings()
            
            # Group encodings by person
            person_encodings_dict = {}
            for encoding, name, person_id in encodings_data:
                if name not in person_encodings_dict:
                    person_encodings_dict[name] = []
                person_encodings_dict[name].append(encoding)
            
            # Average encodings per person
            for name, encodings in person_encodings_dict.items():
                avg_encoding = np.mean(encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(name)
                print(f"Loaded {len(encodings)} face encoding(s) for {name} from database")
            
            print(f"\nLoaded {len(self.known_face_names)} known person(s) from database")
            print(f"Known persons: {', '.join(self.known_face_names)}")
        except Exception as e:
            print(f"Error loading from database: {e}")
            print("Falling back to file-based storage")
            self._load_from_files()
    
    def _load_from_files(self):
        """
        Load all known faces from the known_faces directory.
        
        Directory structure should be:
        known_faces/
            person1/
                image1.jpg
                image2.jpg
            person2/
                image1.jpg
        """
        if not os.path.exists(self.known_faces_dir):
            print(f"Warning: {self.known_faces_dir} directory not found. Creating it...")
            os.makedirs(self.known_faces_dir, exist_ok=True)
            return
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Iterate through each person's folder
        for person_name in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_name)
            
            if not os.path.isdir(person_path):
                continue
            
            # Load all images for this person
            person_encodings = []
            for image_file in os.listdir(person_path):
                if not any(image_file.lower().endswith(ext) for ext in image_extensions):
                    continue
                
                image_path = os.path.join(person_path, image_file)
                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings (128-dimensional vector)
                    encodings = face_recognition.face_encodings(image)
                    
                    if len(encodings) > 0:
                        # Use the first face found in the image
                        encoding = encodings[0]
                        person_encodings.append(encoding)
                        
                        # Optionally save to database if enabled
                        if self.use_database and self.db:
                            try:
                                person_id = self.db.get_person_by_name(person_name)
                                if person_id is None:
                                    person_id = self.db.add_person(person_name)
                                self.db.add_face_encoding(person_id, encoding, image_path)
                            except Exception as e:
                                print(f"Warning: Could not save to database: {e}")
                        
                        print(f"Loaded face encoding for {person_name} from {image_file}")
                    else:
                        print(f"Warning: No face found in {image_path}")
                        
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
            
            # Add all encodings for this person
            if person_encodings:
                # Average multiple encodings for the same person (optional)
                avg_encoding = np.mean(person_encodings, axis=0)
                self.known_face_encodings.append(avg_encoding)
                self.known_face_names.append(person_name)
                print(f"Added {len(person_encodings)} face(s) for {person_name}")
        
        print(f"\nLoaded {len(self.known_face_names)} known person(s)")
        print(f"Known persons: {', '.join(self.known_face_names)}")
    
    def encode_face(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face encoding from a detected face location.
        
        Args:
            frame: BGR frame from OpenCV
            face_location: Face bounding box as (top, right, bottom, left)
            
        Returns:
            128-dimensional face encoding or None if encoding fails
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract encoding
        encodings = face_recognition.face_encodings(
            rgb_frame, 
            [face_location]
        )
        
        if len(encodings) > 0:
            return encodings[0]
        return None
    
    def encode_face_arcface(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Alternative encoding using ArcFace/InsightFace (for production).
        
        TODO: Implement ArcFace encoding
        # Crop face from frame
        top, right, bottom, left = face_location
        face_crop = frame[top:bottom, left:right]
        
        # Get ArcFace embedding (512-dimensional)
        face_embedding = self.arcface_model.get(face_crop)
        return face_embedding[0].embedding
        """
        pass
    
    def match_face(self, face_encoding: np.ndarray, log_to_db: bool = True) -> Tuple[Optional[str], float]:
        """
        Match a face encoding against known faces.
        
        Args:
            face_encoding: 128-dimensional face encoding
            log_to_db: If True and database is enabled, log recognition event
            
        Returns:
            Tuple of (matched_name, distance) or (None, distance) if no match
        """
        if len(self.known_face_encodings) == 0:
            return None, float('inf')
        
        # Calculate distances to all known faces
        distances = face_recognition.face_distance(
            self.known_face_encodings, 
            face_encoding
        )
        
        # Find the best match
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]
        
        # Check if distance is below threshold
        matched_name = None
        if best_distance <= self.tolerance:
            matched_name = self.known_face_names[best_match_index]
        
        # Log to database if enabled
        if log_to_db and self.use_database and self.db:
            try:
                person_id = None
                if matched_name:
                    person_id = self.db.get_person_by_name(matched_name)
                confidence = 1.0 - best_distance  # Convert distance to confidence
                self.db.log_recognition(
                    person_id=person_id,
                    recognized_name=matched_name,
                    confidence=confidence,
                    distance=best_distance
                )
            except Exception as e:
                print(f"Warning: Could not log to database: {e}")
        
        return matched_name, best_distance
    
    def match_face_cosine(self, face_encoding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Alternative matching using cosine similarity (for production with ArcFace).
        
        TODO: Implement cosine similarity matching
        # Normalize encodings
        face_encoding_norm = face_encoding / np.linalg.norm(face_encoding)
        known_encodings_norm = self.known_face_encodings / np.linalg.norm(
            self.known_face_encodings, axis=1, keepdims=True
        )
        
        # Cosine similarity (higher is better)
        similarities = np.dot(known_encodings_norm, face_encoding_norm)
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[best_match_index]
        
        # Threshold check (e.g., > 0.6 for cosine similarity)
        if best_similarity > self.tolerance:
            return self.known_face_names[best_match_index], best_similarity
        """
        pass
    
    def add_known_face(self, name: str, encoding: np.ndarray):
        """
        Add a new known face to the database.
        
        Args:
            name: Person's name
            encoding: Face encoding
        """
        self.known_face_encodings.append(encoding)
        self.known_face_names.append(name)
        print(f"Added new known face: {name}")

