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
face-detection/
├── detector.py              # Face detection module (dlib-based, ready for RetinaFace/MediaPipe)
├── recognizer.py            # Face recognition module (ready for ArcFace/InsightFace)
├── main.py                  # Main application with video loop
├── capture_faces.py         # Utility script to capture faces from webcam
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image for face recognition (with GUI support)
├── Dockerfile.headless      # Docker image for headless servers
├── docker-compose.yml       # Docker Compose configuration
├── .dockerignore            # Files to exclude from Docker build
├── .gitignore               # Git ignore rules
├── known_faces/             # Directory for known face images (gitignored)
│   ├── person1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── person2/
│       └── image1.jpg
└── README.md
```

## Installation

### Option A: Docker (Recommended for Easy Setup)

Docker eliminates the need to install system dependencies manually.

#### Prerequisites
- [Docker](https://www.docker.com/get-started) installed
- [Docker Compose](https://docs.docker.com/compose/install/) (usually included with Docker Desktop)

#### Quick Start with Docker

1. **Build and run with docker-compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually:**
   ```bash
   # Build the image
   docker build -t face-recognition .
   
   # Run the container (Linux)
   docker run -it --rm \
     --device=/dev/video0 \
     -v $(pwd)/known_faces:/app/known_faces \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     face-recognition
   
   # Run on macOS/Windows (camera access may vary)
   docker run -it --rm \
     -v $(pwd)/known_faces:/app/known_faces \
     face-recognition
   ```

3. **For headless servers (no display):**
   ```bash
   docker build -f Dockerfile.headless -t face-recognition-headless .
   docker run -it --rm \
     --device=/dev/video0 \
     -v $(pwd)/known_faces:/app/known_faces \
     face-recognition-headless
   ```

**Note:** Camera access on macOS/Windows may require additional configuration. On Linux, ensure your user is in the `video` group.

#### Docker Commands

- **Capture faces:** Modify `docker-compose.yml` to use `capture_faces.py` or run:
  ```bash
  docker-compose run --rm face-recognition python capture_faces.py
  ```

- **Stop container:** Press `Ctrl+C` or run `docker-compose down`

- **View logs:** `docker-compose logs -f`

### Option B: Local Installation

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

You have two options to add face images:

#### Option A: Use the Capture Script (Recommended)

Use the included `capture_faces.py` script to capture images directly from your webcam:

```bash
python capture_faces.py
```

The script will:
- Prompt you for the person's name
- Open your webcam
- Let you capture multiple images (press SPACE to capture)
- Automatically save images to the correct folder structure

#### Option B: Manually Add Images

Create a `known_faces` directory and add subdirectories for each person:

```bash
mkdir -p known_faces/person1
mkdir -p known_faces/person2
```

Then add image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`) to each person's folder:

```bash
known_faces/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── person2/
    ├── photo1.jpg
    └── photo2.jpg
```

**What Images Work Best?**

For best recognition accuracy, use images with these characteristics:

✅ **Good Images:**
- Clear, front-facing photos (person looking at camera)
- Good lighting (face is well-lit, not too dark or overexposed)
- Face takes up a reasonable portion of the image (not too small)
- Multiple images per person (3-10 images recommended)
- Different angles slightly (straight, slight left, slight right)
- Different expressions (neutral, smiling)
- Different lighting conditions if possible

❌ **Avoid:**
- Blurry or low-resolution images
- Faces that are too small in the frame
- Extreme angles (profile views, looking up/down)
- Heavy shadows or backlighting
- Images with multiple faces (only first face is used)
- Sunglasses or face coverings
- Very old photos if appearance has changed significantly

**Tips:**
- **5-10 images per person** is ideal for good recognition
- The system averages all encodings from a person's folder, so more variety helps
- Test with the same lighting conditions you'll use during recognition

## Usage

### Step 1: Capture Known Faces

First, capture face images for people you want to recognize:

```bash
python capture_faces.py
```

Follow the prompts to:
1. Enter the person's name
2. Position face in the camera frame
3. Press SPACE to capture each image
4. Capture 5-10 images per person for best results

### Step 2: Run Face Recognition

Once you have known faces loaded, run the main application:

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

### Docker Issues

#### Camera Not Accessible in Docker

**Linux:**
```bash
# Add your user to video group
sudo usermod -aG video $USER
# Log out and back in, then try again

# Or run with privileged mode (less secure)
docker run --privileged ...
```

**macOS/Windows:**
- Docker Desktop on macOS/Windows has limited camera access
- Consider using a USB camera with specific device mapping
- Alternative: Use the local installation instead of Docker

#### Display Not Working (Linux)

```bash
# Allow X11 forwarding
xhost +local:docker

# Then run docker with display settings
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
```

#### Build Fails (dlib compilation)

The Dockerfile includes all necessary build dependencies. If build fails:
- Ensure you have enough disk space (dlib compilation needs ~2GB)
- Try building with more memory: `docker build --memory=4g ...`
- Check Docker logs: `docker-compose logs`

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

