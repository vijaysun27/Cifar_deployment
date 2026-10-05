import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from app.utils.image_utils import preprocess_image

# Restrict TensorFlow thread usage to save memory on free tier
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

class PredictionService:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"CIFAR-10 model file not found: {model_path}")
            raise FileNotFoundError(f"CIFAR-10 model file not found: {model_path}")
        
        try:
            self.model = load_model(model_path)
            logger.info("CIFAR-10 model loaded successfully")
            
            # Warm up the model to allocate TF buffers before the first user request
            logger.info("Running warmup prediction...")
            dummy_image = np.zeros((1, 32, 32, 3), dtype=np.float32)
            self.model.predict(dummy_image, verbose=0)
            logger.info("Warmup prediction complete.")
        except Exception as e:
            logger.error(f"Failed to load CIFAR-10 model from {model_path}: {e}")
            raise

    def predict(self, image):
        try:
            processed_image = preprocess_image(image)
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = predictions[0]
            
            predicted_index = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(probabilities[predicted_index]) * 100
            
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_predictions = [
                {
                    "class": CLASS_NAMES[int(i)],
                    "confidence": round(float(probabilities[i]) * 100, 2)
                }
                for i in top_indices
            ]
            
            logger.info(f"Prediction completed: {predicted_class} ({confidence:.2f}%)")
            
            return {
                "predicted_class": predicted_class,
                "confidence": round(confidence, 2),
                "top_predictions": top_predictions
            }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
