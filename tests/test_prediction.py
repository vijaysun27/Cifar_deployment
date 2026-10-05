import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from app.services.prediction_service import PredictionService, CLASS_NAMES

@patch('os.path.exists')
@patch('app.services.prediction_service.load_model')
def test_prediction_service(mock_load_model, mock_exists):
    # Setup mock file exists
    mock_exists.return_value = True
    
    # Setup mock model
    mock_model = MagicMock()
    # Mock predict to return probabilities where index 3 (cat) is highest
    mock_probs = np.zeros((1, 10))
    mock_probs[0, 3] = 0.85
    mock_probs[0, 5] = 0.10
    mock_probs[0, 0] = 0.05
    mock_model.predict.return_value = mock_probs
    mock_load_model.return_value = mock_model
    
    # Initialize service
    service = PredictionService("dummy_path.keras")
    
    # Create dummy image
    from PIL import Image
    dummy_image = Image.new('RGB', (32, 32))
    
    # Perform prediction
    result = service.predict(dummy_image)
    
    # Verify results
    assert result["predicted_class"] == "cat"
    assert result["confidence"] == 85.0
    assert len(result["top_predictions"]) == 3
    
    top = result["top_predictions"]
    assert top[0]["class"] == "cat"
    assert top[0]["confidence"] == 85.0
    assert top[1]["class"] == "dog"
    assert top[1]["confidence"] == 10.0
    
    # Verify valid classes and confidence bounds
    assert result["predicted_class"] in CLASS_NAMES
    assert 0 <= result["confidence"] <= 100
