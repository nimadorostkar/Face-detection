"""
Face Capture Utility

Helper script to capture face images from webcam and save them to known_faces directory.
This makes it easy to build your face database.
"""

import cv2
import os
from pathlib import Path


def capture_faces(person_name: str, num_images: int = 5, output_dir: str = "known_faces"):
    """
    Capture face images from webcam and save them to known_faces directory.
    
    Args:
        person_name: Name of the person (will create a folder with this name)
        num_images: Number of images to capture
        output_dir: Directory to save images (default: known_faces)
    """
    # Create output directory structure
    person_dir = os.path.join(output_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print(f"Capturing {num_images} images for: {person_name}")
    print("="*60)
    print("Instructions:")
    print("  - Position your face in the center of the frame")
    print("  - Look directly at the camera")
    print("  - Press SPACE to capture an image")
    print("  - Press 'q' to quit early")
    print("="*60 + "\n")
    
    captured_count = 0
    frame_count = 0
    
    try:
        while captured_count < num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw instructions on frame
            cv2.putText(
                frame,
                f"Captured: {captured_count}/{num_images}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Press SPACE to capture | 'q' to quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Draw a guide rectangle in the center
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            rect_size = 200
            cv2.rectangle(
                frame,
                (center_x - rect_size, center_y - rect_size),
                (center_x + rect_size, center_y + rect_size),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Position face here",
                (center_x - 80, center_y - rect_size - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            
            # Show frame
            cv2.imshow('Face Capture - Press SPACE to capture', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nCapture cancelled by user")
                break
            elif key == ord(' '):  # Spacebar
                # Save image
                filename = f"{person_name}_{captured_count + 1:03d}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, frame)
                captured_count += 1
                print(f"✓ Captured image {captured_count}/{num_images}: {filename}")
                
                # Show confirmation
                frame_count = 0
                while frame_count < 30:  # Show green flash for ~1 second
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        # Flash green overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
                        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                        cv2.putText(
                            frame,
                            "CAPTURED!",
                            (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 255, 0),
                            3
                        )
                        cv2.imshow('Face Capture - Press SPACE to capture', frame)
                        frame_count += 1
                        cv2.waitKey(33)
    
    except KeyboardInterrupt:
        print("\nCapture interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_count > 0:
            print(f"\n✓ Successfully captured {captured_count} image(s) for {person_name}")
            print(f"  Saved to: {person_dir}")
            print(f"\nYou can now run 'python main.py' to test recognition!")
        else:
            print("\nNo images captured.")


def main():
    """
    Interactive main function.
    """
    print("\n" + "="*60)
    print("Face Capture Utility")
    print("="*60)
    
    # Get person name
    person_name = input("\nEnter person's name: ").strip()
    if not person_name:
        print("Error: Person name cannot be empty")
        return
    
    # Get number of images
    try:
        num_images = input("Number of images to capture (default: 5): ").strip()
        num_images = int(num_images) if num_images else 5
        if num_images < 1:
            print("Error: Must capture at least 1 image")
            return
    except ValueError:
        print("Error: Invalid number, using default: 5")
        num_images = 5
    
    # Capture faces
    capture_faces(person_name, num_images)


if __name__ == "__main__":
    main()

