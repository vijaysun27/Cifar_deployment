import pytest
from io import BytesIO
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config.update({
        "TESTING": True,
    })
    
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200

def test_predict_no_image(client):
    response = client.post('/predict')
    assert response.status_code == 400
    assert response.json['success'] is False

def test_predict_empty_file(client):
    data = {
        'image': (BytesIO(b""), '')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json['success'] is False

def test_predict_unsupported_file(client):
    data = {
        'image': (BytesIO(b"test data"), 'test.txt')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json['success'] is False

def test_predict_invalid_image(client):
    data = {
        'image': (BytesIO(b"not a real image data"), 'test.png')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400
    assert response.json['success'] is False

def test_predict_valid_image(client):
    from PIL import Image
    img = Image.new('RGB', (32, 32))
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    data = {
        'image': (img_byte_arr, 'test.png')
    }
    
    mock_result = {
        "predicted_class": "cat",
        "confidence": 85.0,
        "top_predictions": [
            {"class": "cat", "confidence": 85.0},
            {"class": "dog", "confidence": 10.0},
            {"class": "frog", "confidence": 5.0}
        ]
    }
    
    app = client.application
    
    class DummyService:
        def predict(self, image):
            return mock_result.copy()
            
    app.prediction_service = DummyService()
    
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.json['success'] is True
    assert response.json['predicted_class'] == 'cat'
    assert response.json['confidence'] == 85.0
    assert len(response.json['top_predictions']) == 3

def test_predict_oversized_file(client):
    app = client.application
    app.config['MAX_CONTENT_LENGTH'] = 1  # 1 byte limit
    
    data = {
        'image': (BytesIO(b"a" * 10), 'test.png')
    }
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 413
    assert response.json['success'] is False
    assert response.json['error'] == "File size exceeds the configured limit."
