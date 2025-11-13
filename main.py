"""
Real-time Face Recognition System

Main entry point for the face recognition application.
Handles video capture, real-time processing, and display.

Performance optimizations:
- Frame resizing for faster processing
- Efficient OpenCV video capture
- Optional frame skipping for lower-end hardware
"""

import cv2
import numpy as np
import time
from detector import FaceDetector
from recognizer import FaceRecognizer


class FaceRecognitionSystem:
    """
    Main face recognition system that orchestrates detection and recognition.
    """
    
    def __init__(
        self,
        known_faces_dir: str = "known_faces",
        camera_index: int = 0,
        display_scale: float = 1.0,
        process_scale: float = 0.25,
        frame_skip: int = 1,
        tolerance: float = 0.6
    ):
        """
        Initialize the face recognition system.
        
        Args:
            known_faces_dir: Directory containing known faces
            camera_index: Camera device index (0 for default webcam)
            display_scale: Scale factor for display window (1.0 = original size)
            process_scale: Scale factor for processing (0.25 = 1/4 size, faster)
            frame_skip: Process every Nth frame (1 = all frames, 2 = every other)
            tolerance: Face matching tolerance
        """
        self.camera_index = camera_index
        self.display_scale = display_scale
        self.frame_skip = frame_skip
        
        # Initialize detector and recognizer
        self.detector = FaceDetector(model='hog', scale_factor=process_scale)
        self.recognizer = FaceRecognizer(
            known_faces_dir=known_faces_dir,
            tolerance=tolerance
        )
        
        # Performance tracking
        self.fps_history = []
        self.frame_count = 0
        
        # Video capture (will be initialized in run)
        self.cap = None
    
    def run(self):
        """
        Main loop for real-time face recognition.
        """
        # Initialize video capture
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "="*50)
        print("Face Recognition System Started")
        print("="*50)
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("="*50 + "\n")
        
        # Variables for frame skipping
        last_recognition_time = time.time()
        recognition_cache = {}  # Cache results for skipped frames
        
        try:
            while True:
                start_time = time.time()
                
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect (optional)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                self.frame_count += 1
                should_process = (self.frame_count % self.frame_skip == 0)
                
                if should_process:
                    # Detect and recognize faces
                    face_locations = self.detector.detect_faces(frame)
                    recognition_cache = {}
                    
                    for face_location in face_locations:
                        # Extract encoding
                        face_encoding = self.recognizer.encode_face(frame, face_location)
                        
                        if face_encoding is not None:
                            # Match against known faces
                            name, distance = self.recognizer.match_face(face_encoding)
                            recognition_cache[face_location] = (name, distance)
                
                # Draw results on frame
                self._draw_results(frame, recognition_cache)
                
                # Calculate and display FPS
                fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                # Display FPS
                cv2.putText(
                    frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Resize for display if needed
                if self.display_scale != 1.0:
                    display_frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=self.display_scale,
                        fy=self.display_scale
                    )
                else:
                    display_frame = frame
                
                # Show frame
                cv2.imshow('Face Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to {filename}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def _draw_results(self, frame: np.ndarray, recognition_cache: dict):
        """
        Draw bounding boxes and names on the frame.
        
        Args:
            frame: Frame to draw on
            recognition_cache: Dictionary mapping face_locations to (name, distance)
        """
        for face_location, (name, distance) in recognition_cache.items():
            top, right, bottom, left = face_location
            
            # Draw bounding box
            color = (0, 255, 0) if name else (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Draw label background
            label = name if name else "Unknown"
            label_text = f"{label} ({distance:.2f})"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
            
            # Draw label background rectangle
            cv2.rectangle(
                frame,
                (left, top - text_height - baseline - 10),
                (left + text_width, top),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label_text,
                (left, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    def cleanup(self):
        """
        Clean up resources.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"\nAverage FPS: {avg_fps:.2f}")
        print("System stopped.")


def main():
    """
    Main entry point.
    """
    # Configuration
    config = {
        'known_faces_dir': 'known_faces',
        'camera_index': 0,
        'display_scale': 1.0,      # Display at original size
        'process_scale': 0.25,    # Process at 1/4 size for speed
        'frame_skip': 1,           # Process every frame
        'tolerance': 0.6           # Face matching tolerance
    }
    
    # Adjust for lower-end hardware
    # config['process_scale'] = 0.5  # Process at 1/2 size
    # config['frame_skip'] = 2       # Process every other frame
    
    # Create and run system
    system = FaceRecognitionSystem(**config)
    system.run()


if __name__ == "__main__":
    main()

