"""Tests for the prediction functionality of the FastAPI YOLO application."""
import pytest
import numpy as np
from app.schemas.model_response import ModelResponse

# Test the YOLO handler directly
def test_yolo_handler_predict(yolo_handler, sample_image_bytes, expected_classes):
    """Test that the YOLO handler can make predictions on an image."""
    # Make prediction
    result = yolo_handler.predict("test_model", sample_image_bytes)
    
    # Verify the result is a ModelResponse
    assert isinstance(result, ModelResponse)
    
    # Verify we have detections
    assert len(result.bboxes) > 0
    assert len(result.scores) == len(result.bboxes)
    
    # Verify bbox format: [x, y, w, h] with values between 0 and 1
    for bbox in result.bboxes:
        assert len(bbox) == 4  # x, y, w, h
        assert all(0 <= coord <= 1 for coord in bbox)  # Normalized coordinates
    
    # Verify scores are between 0 and 1
    assert all(0 <= score <= 1 for score in result.scores)
    
    # If we have class names, verify they match expected classes
    if result.class_names:
        assert any(cls in result.class_names for cls in expected_classes)

# Test the FastAPI endpoint
def test_predict_endpoint(test_client, sample_image_bytes, expected_classes):
    """Test the /predict/ endpoint with a URL and verify the response format."""
    # Test with image URL
    response = test_client.post(
        "/predict/",
        json={
            "model_id": "test_model",
            "image_url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
        }
    )
    
    # Verify successful response
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "bboxes" in data
    assert "scores" in data
    assert len(data["bboxes"]) == len(data["scores"])
    
    # If class names are included, verify they match expected classes
    if "class_names" in data and data["class_names"]:
        assert any(cls in data["class_names"] for cls in expected_classes)

def test_predict_with_invalid_url(test_client):
    """Test the /predict/ endpoint with an invalid URL."""
    response = test_client.post(
        "/predict/",
        json={
            "model_id": "test_model",
            "image_url": "https://example.com/nonexistent.jpg"
        }
    )
    assert response.status_code == 400
    assert "Failed to download image" in response.text

def test_predict_with_invalid_model(test_client, sample_image_bytes):
    """Test the /predict/ endpoint with an invalid model ID."""
    response = test_client.post(
        "/predict/",
        json={
            "model_id": "nonexistent_model",
            "image_url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
        }
    )
    assert response.status_code == 400
    assert "not loaded" in response.text

# Test ModelResponse model
def test_model_response_validation():
    """Test that ModelResponse validates its inputs correctly."""
    # Valid data
    valid_data = {
        "bboxes": [[0.1, 0.2, 0.3, 0.4]],
        "scores": [0.95],
        "thetas": [0.0],
        "class_names": ["person"],
        "class_ids": [0]
    }
    response = ModelResponse(**valid_data)
    assert response is not None
    
    # Test with missing optional fields
    minimal_data = {
        "bboxes": [[0.1, 0.2, 0.3, 0.4]],
        "scores": [0.95]
    }
    response = ModelResponse(**minimal_data)
    assert response is not None
    assert response.thetas is None
    assert response.class_names is None
    assert response.class_ids is None
    
    # Test with invalid bbox format
    with pytest.raises(ValueError):
        invalid_data = valid_data.copy()
        invalid_data["bboxes"] = [[0.1, 0.2, 0.3]]  # Missing one coordinate
        ModelResponse(**invalid_data)
