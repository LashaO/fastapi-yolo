"""Pytest configuration and fixtures for testing the FastAPI YOLO application."""
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.yolo_handler import YOLOHandler
import json

# Sample test image URL (a small image with a clear object for detection)
TEST_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"

@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def yolo_handler():
    """Create a YOLO handler with a test model."""
    handler = YOLOHandler()
    # Use a small YOLOv8n model for testing
    model_path = "yolov8n.pt"  # This will be downloaded automatically
    handler.load_model(
        model_id="test_model",
        model_path=model_path,
        device="cpu",
        imgsz=640,
        conf=0.25
    )
    return handler

@pytest.fixture(scope="module")
def expected_classes():
    """Return the expected class names for the test image."""
    # These are the classes we expect to find in the test image
    return ["person", "tie"]

@pytest.fixture(scope="module")
def sample_image_bytes():
    """Download and return the sample image as bytes."""
    import requests
    response = requests.get(TEST_IMAGE_URL)
    response.raise_for_status()
    return response.content
