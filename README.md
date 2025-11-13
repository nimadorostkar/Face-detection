# Real-time Face Recognition System

A real-time face recognition system built with Python, OpenCV, and face_recognition (dlib-based). The system captures live video from a webcam, detects faces, and matches them against a database of known faces.

## Features

- ✅ Real-time face detection using face_recognition (dlib HOG/SVM)
- ✅ Face encoding extraction (128-dimensional embeddings)
- ✅ Real-time matching against known faces database
- ✅ Optimized for speed with frame resizing and processing optimizations
- ✅ Clean, modular architecture ready for production enhancements

## Project Structure

```
face/
├── detector.py          # Face detection module (dlib-based, ready for RetinaFace/MediaPipe)
├── recognizer.py        # Face recognition module (ready for ArcFace/InsightFace)
├── main.py              # Main application with video loop
├── requirements.txt     # Python dependencies
├── known_faces/         # Directory for known face images
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       └── image1.jpg
└── README.md
```

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
brew install cmake
brew install dlib
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake
sudo apt-get install libdlib-dev
```

**Windows:**
Install CMake from https://cmake.org/download/

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note:** Installing `dlib` can be tricky. If you encounter issues:

- **macOS:** `pip install dlib` should work after installing cmake
- **Linux:** May need to compile from source or use conda
- **Windows:** Consider using conda: `conda install -c conda-forge dlib`

### 3. Prepare Known Faces

Create a `known_faces` directory and add subdirectories for each person:

```bash
mkdir -p known_faces/person1
mkdir -p known_faces/person2
```

Add multiple images of each person to their respective folders. The system will automatically load and encode all faces.

## Usage

### Basic Usage

```bash
python main.py
```

### Configuration

Edit `main.py` to adjust settings:

```python
config = {
    'known_faces_dir': 'known_faces',
    'camera_index': 0,           # Camera device index
    'display_scale': 1.0,        # Display window scale
    'process_scale': 0.25,       # Processing scale (0.25 = 1/4 size, faster)
    'frame_skip': 1,             # Process every Nth frame
    'tolerance': 0.6             # Matching threshold (lower = stricter)
}
```

### Controls

- **'q'**: Quit the application
- **'s'**: Save current frame to disk

## Performance Optimization

The system includes several optimizations:

1. **Frame Resizing**: Processes frames at 1/4 size by default (configurable)
2. **Frame Skipping**: Can process every Nth frame for lower-end hardware
3. **Efficient Detection**: Uses HOG model (faster) instead of CNN (more accurate)

For lower-end hardware, adjust in `main.py`:
```python
'process_scale': 0.5,    # Process at 1/2 size
'frame_skip': 2,         # Process every other frame
```

## Production Enhancements

The code includes TODO comments for production-level improvements:

### 1. Faster Detection

Replace dlib with RetinaFace or MediaPipe:

**RetinaFace:**
```python
# In detector.py
from retinaface import RetinaFace
# Implement detect_faces_retinaface()
```

**MediaPipe:**
```python
# In detector.py
import mediapipe as mp
# Implement detect_faces_mediapipe()
```

### 2. Better Accuracy

Replace face_recognition embeddings with ArcFace/InsightFace:

```python
# In recognizer.py
import insightface
# Initialize ArcFace model
# Use encode_face_arcface() for 512-dimensional embeddings
```

### 3. Liveness Detection

Add anti-spoofing measures:

- **Blink Detection**: Track eye blinks using facial landmarks
- **Challenge-Response**: Ask user to perform actions (turn head, smile)
- **Depth/IR Input**: Use depth cameras for 3D face verification

### 4. GPU Acceleration

- **ONNX Runtime**: Convert models to ONNX format for optimized inference
- **TensorRT**: For NVIDIA GPUs, use TensorRT for maximum performance
- **CUDA**: Enable CUDA support in dlib/OpenCV

## Troubleshooting

### Camera Not Opening

- Check camera permissions
- Try different `camera_index` values (0, 1, 2, etc.)
- Verify camera is not being used by another application

### Poor Recognition Accuracy

- Add more images per person (different angles, lighting)
- Adjust `tolerance` value (try 0.5-0.7)
- Ensure faces are clearly visible in known face images
- Use better quality images (higher resolution)

### Low FPS

- Reduce `process_scale` (e.g., 0.5 or 0.25)
- Increase `frame_skip` (e.g., 2 or 3)
- Use HOG model instead of CNN
- Consider production alternatives (RetinaFace, MediaPipe)

### dlib Installation Issues

- Use conda: `conda install -c conda-forge dlib`
- Or compile from source with proper CMake configuration
- Consider using Docker with pre-installed dependencies

## License

This project is provided as-is for educational and development purposes.

## Acknowledgments

- face_recognition library by Adam Geitgey
- dlib library by Davis King
- OpenCV community

