# FastAPI Face Recognition System

A production-ready, Dockerized face recognition system built with FastAPI and PostgreSQL (pgvector) that can efficiently handle 5000+ users.

## 🚀 Features

- **FastAPI REST API** for face registration and recognition
- **PostgreSQL with pgvector** for efficient vector similarity search
- **Optimized for scale**: Handles 5000+ users with <100ms query time
- **Docker Compose** orchestration for easy deployment
- **128D embeddings** (face_recognition) with easy upgrade path to 512D (ArcFace)
- **Vector indexing** (ivfflat) for fast similarity search
- **Production-ready** with health checks and error handling

## 📁 Project Structure

```
face-detection/
├── docker-compose.fastapi.yml  # Docker Compose for FastAPI setup
├── db/
│   └── Dockerfile              # PostgreSQL with pgvector
└── api/
    ├── Dockerfile              # FastAPI application
    ├── main.py                 # FastAPI app and endpoints
    ├── models.py               # SQLAlchemy models
    ├── requirements.txt        # Python dependencies
    └── utils/
        ├── config.py           # Configuration management
        ├── db.py              # Database setup and utilities
        └── face_utils.py      # Face detection and embedding extraction
```

## 🛠️ Installation

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM recommended
- 10GB free disk space

### Quick Start

1. **Clone and navigate to the project:**
   ```bash
   cd face-detection
   ```

2. **Start the services:**
   ```bash
   docker-compose -f docker-compose.fastapi.yml up --build
   ```

   This will:
   - Build PostgreSQL container with pgvector extension
   - Build FastAPI application container
   - Initialize database schema and indexes
   - Start API server on port 8000

3. **Verify the API is running:**
   ```bash
   curl http://localhost:8000/health
   ```

## 📡 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `GET /`
Get API information and available endpoints.

#### `GET /health`
Health check endpoint. Returns API and database status.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "threshold": 0.45,
  "embedding_dimension": 128
}
```

#### `GET /stats`
Get database statistics.

**Response:**
```json
{
  "total_users": 42,
  "threshold": 0.45,
  "embedding_dimension": 128
}
```

#### `POST /register`
Register a new person with their face image.

**Request:**
- `name` (form-data): Person's name (must be unique)
- `image` (file): Face image (JPEG, PNG, etc.)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/register" \
  -F "name=John Doe" \
  -F "image=@/path/to/photo.jpg"
```

**Response:**
```json
{
  "status": "success",
  "message": "User 'John Doe' registered successfully",
  "user_id": 1,
  "name": "John Doe",
  "embedding_dimension": 128
}
```

#### `POST /recognize`
Recognize a face from an image.

**Request:**
- `image` (file): Face image to recognize
- `threshold` (optional, query param): Custom match threshold (default: 0.45)

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "image=@/path/to/photo.jpg"
```

**Response (match found):**
```json
{
  "status": "match",
  "message": "Face recognized as 'John Doe'",
  "distance": 0.32,
  "confidence": 0.68,
  "matched_user": {
    "id": 1,
    "name": "John Doe"
  },
  "threshold": 0.45
}
```

**Response (no match):**
```json
{
  "status": "no_match",
  "message": "No matching face found",
  "distance": 0.67,
  "confidence": 0.33,
  "matched_user": null,
  "threshold": 0.45
}
```

## 🔧 Configuration

### Environment Variables

Set these in `docker-compose.fastapi.yml` or via `.env` file:

**Database:**
- `DB_HOST`: PostgreSQL host (default: `postgres`)
- `DB_PORT`: PostgreSQL port (default: `5432`)
- `DB_NAME`: Database name (default: `face_recognition`)
- `DB_USER`: Database user (default: `postgres`)
- `DB_PASSWORD`: Database password (default: `postgres`)

**Face Recognition:**
- `FACE_MATCH_THRESHOLD`: Distance threshold for matching (default: `0.45`)
  - Lower = stricter matching
  - Recommended: 0.4-0.6 for face_recognition
- `EMBEDDING_DIMENSION`: Embedding dimension (default: `128`)

**Performance:**
- `VECTOR_INDEX_LISTS`: Number of lists for ivfflat index (default: `100`)
  - Recommended: ~sqrt(number of users)
  - For 5000 users: 70-100
  - For 10000 users: 100-150

## 🎯 Performance

### Optimizations

1. **Vector Indexing**: Uses PostgreSQL's ivfflat index for fast similarity search
2. **Efficient Queries**: Single query with `ORDER BY ... LIMIT 1` for best match
3. **Connection Pooling**: SQLAlchemy connection management
4. **Docker Optimization**: Multi-stage builds and slim base images

### Expected Performance

- **Registration**: <500ms (face detection + embedding + DB insert)
- **Recognition**: <100ms for 5000+ users (with index)
- **Database Queries**: <50ms with proper indexing

### Scaling

For 10,000+ users:
- Increase `VECTOR_INDEX_LISTS` to 150-200
- Consider read replicas for recognition queries
- Use connection pooling for high concurrency

## 🔄 Upgrading to Production Features

The code includes TODO comments for production upgrades:

### 1. Better Detection (RetinaFace/MediaPipe)

**Location:** `api/utils/face_utils.py`

Replace `face_recognition.face_locations()` with:
```python
# RetinaFace (faster, more accurate)
from retinaface import RetinaFace
faces = RetinaFace.detect_faces(image)

# Or MediaPipe (mobile-friendly)
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
```

### 2. Better Embeddings (ArcFace/InsightFace)

**Location:** `api/utils/face_utils.py` and `api/models.py`

1. Change embedding dimension to 512:
   ```python
   # In models.py
   embedding = Column(Vector(512), nullable=False)
   ```

2. Update embedding extraction:
   ```python
   # In face_utils.py
   import insightface
   app = insightface.app.FaceAnalysis()
   app.prepare(ctx_id=0, det_size=(640, 640))
   # Returns 512D embeddings
   ```

3. Update database index:
   ```sql
   DROP INDEX users_embedding_idx;
   CREATE INDEX users_embedding_idx
   ON users USING ivfflat (embedding vector_cosine_ops)
   WITH (lists = 150);
   ```

### 3. GPU Acceleration

**Location:** `api/Dockerfile` and `api/utils/face_utils.py`

1. Use GPU-enabled base image
2. Install CUDA dependencies
3. Use ONNX Runtime with GPU:
   ```python
   import onnxruntime as ort
   providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
   session = ort.InferenceSession("model.onnx", providers=providers)
   ```

### 4. Real-time Webcam Inference

**Location:** `api/main.py`

Add WebSocket endpoint:
```python
from fastapi import WebSocket

@app.websocket("/ws/recognize")
async def recognize_stream(websocket: WebSocket):
    # Process video frames in real-time
    pass
```

### 5. Liveness Detection

**Location:** `api/utils/face_utils.py`

Add liveness checks:
- Blink detection using facial landmarks
- Depth/IR camera input
- Challenge-response mechanisms

## 🧪 Testing

### Test Registration

```bash
# Register a person
curl -X POST "http://localhost:8000/register" \
  -F "name=Test User" \
  -F "image=@test_photo.jpg"
```

### Test Recognition

```bash
# Recognize a face
curl -X POST "http://localhost:8000/recognize" \
  -F "image=@test_photo.jpg"
```

### Using Python

```python
import requests

# Register
with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/register",
        files={"image": f},
        data={"name": "John Doe"}
    )
    print(response.json())

# Recognize
with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/recognize",
        files={"image": f}
    )
    print(response.json())
```

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker-compose -f docker-compose.fastapi.yml ps

# View database logs
docker-compose -f docker-compose.fastapi.yml logs postgres

# Connect to database
docker-compose -f docker-compose.fastapi.yml exec postgres psql -U postgres -d face_recognition
```

### API Issues

```bash
# View API logs
docker-compose -f docker-compose.fastapi.yml logs api

# Restart API
docker-compose -f docker-compose.fastapi.yml restart api
```

### Performance Issues

1. **Slow recognition queries:**
   - Check if index exists: `\d users` in psql
   - Recreate index with appropriate lists count
   - Verify pgvector extension is enabled

2. **High memory usage:**
   - Reduce `VECTOR_INDEX_LISTS` if memory constrained
   - Use connection pooling limits

## 📊 Database Schema

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    embedding vector(128) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX users_embedding_idx
ON users USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

## 🔐 Security Considerations

- Change default database password in production
- Use environment variables for sensitive data
- Implement API authentication (JWT, OAuth)
- Add rate limiting for API endpoints
- Use HTTPS in production
- Validate and sanitize all inputs

## 📝 License

This project is provided as-is for educational and development purposes.

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- pgvector for PostgreSQL vector similarity
- face_recognition library by Adam Geitgey
- dlib library by Davis King

