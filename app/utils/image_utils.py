import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses a PIL Image for the CIFAR-10 model."""
    image = image.convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image).astype("float32")
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
