"""
Face Detection Module

This module handles face detection in images/video frames.
Default implementation uses face_recognition (dlib-based HOG detector).

For production use, consider replacing with:
- RetinaFace: Faster and more accurate, especially for small faces
- MediaPipe/BlazeFace: Mobile-friendly, very fast, good for edge devices
- MTCNN: Good balance of speed and accuracy
"""

import cv2
import face_recognition
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Face detector using face_recognition library (dlib HOG/SVM).
    
    For production alternatives:
    - RetinaFace: Use retinaface library (pip install retinaface)
    - MediaPipe: Use mediapipe library (pip install mediapipe)
    - MTCNN: Use mtcnn library (pip install mtcnn)
    """
    
    def __init__(self, model: str = 'hog', scale_factor: float = 0.25):
        """
        Initialize face detector.
        
        Args:
            model: Detection model - 'hog' (faster, CPU) or 'cnn' (more accurate, GPU)
            scale_factor: Factor to resize frames for faster processing (0.25 = 1/4 size)
        """
        self.model = model
        self.scale_factor = scale_factor
        
        # TODO: Production - Initialize RetinaFace detector
        # from retinaface import RetinaFace
        # self.retina_detector = RetinaFace.build_model()
        
        # TODO: Production - Initialize MediaPipe detector
        # import mediapipe as mp
        # self.mp_face_detection = mp.solutions.face_detection
        # self.face_detection = self.mp_face_detection.FaceDetection(
        #     model_selection=0, min_detection_confidence=0.5
        # )
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame.
        
        Args:
            frame: Input BGR frame from OpenCV
            
        Returns:
            List of face bounding boxes as (top, right, bottom, left) tuples
            Note: face_recognition uses (top, right, bottom, left) format
        """
        # Convert BGR to RGB (face_recognition expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for faster processing
        small_frame = cv2.resize(
            rgb_frame, 
            (0, 0), 
            fx=self.scale_factor, 
            fy=self.scale_factor
        )
        
        # Detect faces
        face_locations = face_recognition.face_locations(
            small_frame, 
            model=self.model
        )
        
        # Scale locations back to original frame size
        scaled_locations = []
        for top, right, bottom, left in face_locations:
            scaled_locations.append((
                int(top / self.scale_factor),
                int(right / self.scale_factor),
                int(bottom / self.scale_factor),
                int(left / self.scale_factor)
            ))
        
        return scaled_locations
    
    def detect_faces_retinaface(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Alternative detection using RetinaFace (for production).
        
        TODO: Implement RetinaFace detection
        from retinaface import RetinaFace
        
        detections = RetinaFace.detect_faces(frame)
        # Convert RetinaFace format to (top, right, bottom, left)
        """
        pass
    
    def detect_faces_mediapipe(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Alternative detection using MediaPipe (for production).
        
        TODO: Implement MediaPipe detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        # Convert MediaPipe format to (top, right, bottom, left)
        """
        pass

