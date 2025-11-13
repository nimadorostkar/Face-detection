"""
Database Module for Face Recognition System

Handles PostgreSQL database operations for storing:
- Person information
- Face encodings
- Recognition logs
"""

import os
import psycopg2
import psycopg2.extras
import numpy as np
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import json


class FaceDatabase:
    """
    Database interface for face recognition system.
    """
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 database: str = None,
                 user: str = None,
                 password: str = None):
        """
        Initialize database connection.
        
        Args:
            host: Database host (defaults to DB_HOST env var or 'localhost')
            port: Database port (defaults to DB_PORT env var or 5432)
            database: Database name (defaults to DB_NAME env var or 'face_recognition')
            user: Database user (defaults to DB_USER env var or 'postgres')
            password: Database password (defaults to DB_PASSWORD env var or 'postgres')
        """
        self.host = host or os.getenv('DB_HOST', 'localhost')
        self.port = port or int(os.getenv('DB_PORT', '5432'))
        self.database = database or os.getenv('DB_NAME', 'face_recognition')
        self.user = user or os.getenv('DB_USER', 'postgres')
        self.password = password or os.getenv('DB_PASSWORD', 'postgres')
        
        self.conn = None
        self._connect()
        self._initialize_schema()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            print(f"✓ Connected to PostgreSQL database: {self.database}")
        except psycopg2.Error as e:
            print(f"Error connecting to database: {e}")
            raise
    
    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        with self.conn.cursor() as cur:
            # Create persons table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create face_encodings table
            # Store encoding as JSON array (128 floats for face_recognition)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                    encoding JSONB NOT NULL,
                    image_path VARCHAR(500),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create recognition_logs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
                    recognized_name VARCHAR(255),
                    confidence FLOAT,
                    distance FLOAT,
                    frame_path VARCHAR(500),
                    recognized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_face_encodings_person_id 
                ON face_encodings(person_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_recognition_logs_person_id 
                ON recognition_logs(person_id)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_recognition_logs_recognized_at 
                ON recognition_logs(recognized_at)
            """)
            
            self.conn.commit()
            print("✓ Database schema initialized")
    
    def add_person(self, name: str) -> int:
        """
        Add a new person to the database.
        
        Args:
            name: Person's name
            
        Returns:
            Person ID
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO persons (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET updated_at = CURRENT_TIMESTAMP RETURNING id",
                    (name,)
                )
                person_id = cur.fetchone()[0]
                self.conn.commit()
                print(f"✓ Added person: {name} (ID: {person_id})")
                return person_id
            except psycopg2.Error as e:
                self.conn.rollback()
                print(f"Error adding person: {e}")
                raise
    
    def add_face_encoding(self, person_id: int, encoding: np.ndarray, image_path: str = None) -> int:
        """
        Add a face encoding for a person.
        
        Args:
            person_id: Person ID
            encoding: Face encoding as numpy array
            image_path: Optional path to source image
            
        Returns:
            Encoding ID
        """
        # Convert numpy array to list for JSON storage
        encoding_list = encoding.tolist()
        
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO face_encodings (person_id, encoding, image_path) VALUES (%s, %s, %s) RETURNING id",
                    (person_id, json.dumps(encoding_list), image_path)
                )
                encoding_id = cur.fetchone()[0]
                self.conn.commit()
                return encoding_id
            except psycopg2.Error as e:
                self.conn.rollback()
                print(f"Error adding face encoding: {e}")
                raise
    
    def get_all_encodings(self) -> List[Tuple[np.ndarray, str, int]]:
        """
        Get all face encodings with person names.
        
        Returns:
            List of tuples: (encoding, person_name, person_id)
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT fe.encoding, p.name, p.id
                FROM face_encodings fe
                JOIN persons p ON fe.person_id = p.id
                ORDER BY p.name, fe.created_at
            """)
            
            results = []
            for row in cur.fetchall():
                encoding_json = row[0]
                encoding = np.array(encoding_json)
                results.append((encoding, row[1], row[2]))
            
            return results
    
    def get_person_encodings(self, person_id: int) -> List[np.ndarray]:
        """
        Get all encodings for a specific person.
        
        Args:
            person_id: Person ID
            
        Returns:
            List of encodings
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT encoding FROM face_encodings WHERE person_id = %s",
                (person_id,)
            )
            
            encodings = []
            for row in cur.fetchall():
                encoding_json = row[0]
                encoding = np.array(encoding_json)
                encodings.append(encoding)
            
            return encodings
    
    def log_recognition(self, 
                       person_id: Optional[int],
                       recognized_name: Optional[str],
                       confidence: float,
                       distance: float,
                       frame_path: str = None):
        """
        Log a recognition event.
        
        Args:
            person_id: Person ID if recognized, None if unknown
            recognized_name: Person name if recognized, None if unknown
            confidence: Recognition confidence (1 - distance)
            distance: Face distance
            frame_path: Optional path to frame image
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO recognition_logs 
                    (person_id, recognized_name, confidence, distance, frame_path)
                    VALUES (%s, %s, %s, %s, %s)
                """, (person_id, recognized_name, confidence, distance, frame_path))
                self.conn.commit()
            except psycopg2.Error as e:
                self.conn.rollback()
                print(f"Error logging recognition: {e}")
    
    def get_person_by_name(self, name: str) -> Optional[int]:
        """
        Get person ID by name.
        
        Args:
            name: Person's name
            
        Returns:
            Person ID or None if not found
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT id FROM persons WHERE name = %s", (name,))
            result = cur.fetchone()
            return result[0] if result else None
    
    def get_recognition_stats(self, limit: int = 100) -> List[Dict]:
        """
        Get recent recognition statistics.
        
        Args:
            limit: Number of recent records to retrieve
            
        Returns:
            List of recognition records
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    rl.id,
                    rl.recognized_name,
                    rl.confidence,
                    rl.distance,
                    rl.recognized_at,
                    p.id as person_id
                FROM recognition_logs rl
                LEFT JOIN persons p ON rl.person_id = p.id
                ORDER BY rl.recognized_at DESC
                LIMIT %s
            """, (limit,))
            
            return [dict(row) for row in cur.fetchall()]
    
    def delete_person(self, person_id: int):
        """
        Delete a person and all their encodings.
        
        Args:
            person_id: Person ID to delete
        """
        with self.conn.cursor() as cur:
            try:
                cur.execute("DELETE FROM persons WHERE id = %s", (person_id,))
                self.conn.commit()
                print(f"✓ Deleted person ID: {person_id}")
            except psycopg2.Error as e:
                self.conn.rollback()
                print(f"Error deleting person: {e}")
                raise
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

