import numpy as np
from PIL import Image
from app.utils.image_utils import preprocess_image

def test_preprocess_image():
    # Create a random RGB image of size 100x100
    test_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), "RGB")
    
    # Process it
    processed = preprocess_image(test_image)
    
    # Verify shape
    assert processed.shape == (1, 32, 32, 3)
    
    # Verify datatype
    assert processed.dtype == np.float32
    
    # Verify normalization
    assert np.min(processed) >= 0.0
    assert np.max(processed) <= 1.0
