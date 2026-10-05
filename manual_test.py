import io
import json
from PIL import Image
from app import create_app

def run_test():
    app = create_app()
    app.config['TESTING'] = True
    client = app.test_client()

    print("=== Creating Test Image ===")
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    print("=== Sending POST /predict Request ===")
    data = {
        'image': (img_byte_arr, 'test_red.png')
    }
    
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.get_json(), indent=2))

if __name__ == '__main__':
    run_test()
