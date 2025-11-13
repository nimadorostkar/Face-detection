#!/usr/bin/env python3
"""
Simple test script for the Face Recognition API.

Usage:
    python test_api.py register <name> <image_path>
    python test_api.py recognize <image_path>
    python test_api.py stats
"""

import sys
import requests
import json

API_BASE_URL = "http://localhost:8000"


def register(name: str, image_path: str):
    """Register a new person."""
    print(f"Registering {name} with image: {image_path}")
    
    with open(image_path, "rb") as f:
        files = {"image": f}
        data = {"name": name}
        
        response = requests.post(f"{API_BASE_URL}/register", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success: {result['message']}")
            print(f"  User ID: {result['user_id']}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  {response.text}")


def recognize(image_path: str):
    """Recognize a face from an image."""
    print(f"Recognizing face from: {image_path}")
    
    with open(image_path, "rb") as f:
        files = {"image": f}
        
        response = requests.post(f"{API_BASE_URL}/recognize", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Message: {result['message']}")
            
            if result['status'] == 'match':
                print(f"✓ Match found!")
                print(f"  Name: {result['matched_user']['name']}")
                print(f"  Distance: {result['distance']:.4f}")
                print(f"  Confidence: {result['confidence']:.4f}")
            else:
                print(f"✗ No match found")
                print(f"  Best distance: {result['distance']:.4f}")
                print(f"  Threshold: {result['threshold']}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  {response.text}")


def stats():
    """Get API statistics."""
    print("Fetching API statistics...")
    
    response = requests.get(f"{API_BASE_URL}/stats")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total users: {result['total_users']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Embedding dimension: {result['embedding_dimension']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def health():
    """Check API health."""
    print("Checking API health...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Database: {result['database']}")
        print(f"Threshold: {result['threshold']}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_api.py register <name> <image_path>")
        print("  python test_api.py recognize <image_path>")
        print("  python test_api.py stats")
        print("  python test_api.py health")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "register":
        if len(sys.argv) < 4:
            print("Error: register requires name and image_path")
            sys.exit(1)
        register(sys.argv[2], sys.argv[3])
    elif command == "recognize":
        if len(sys.argv) < 3:
            print("Error: recognize requires image_path")
            sys.exit(1)
        recognize(sys.argv[2])
    elif command == "stats":
        stats()
    elif command == "health":
        health()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

