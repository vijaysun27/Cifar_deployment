import logging
from flask import Blueprint, render_template, request, jsonify, current_app
from PIL import Image, UnidentifiedImageError

bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": hasattr(current_app, 'prediction_service') and current_app.prediction_service is not None
    }), 200

@bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image was uploaded."}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "The uploaded file is empty."}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported image format."}), 400
        
    try:
        image = Image.open(file)
        image.verify() # Verify it's an image
        file.seek(0)
        image = Image.open(file)
        
        if not hasattr(current_app, 'prediction_service') or current_app.prediction_service is None:
            return jsonify({"success": False, "error": "Prediction service is unavailable."}), 500
            
        result = current_app.prediction_service.predict(image)
        result["success"] = True
        return jsonify(result), 200
        
    except UnidentifiedImageError:
        logger.error("Uploaded file is not a valid image.")
        return jsonify({"success": False, "error": "The uploaded file is not a valid image."}), 400
    except Exception as e:
        logger.error(f"Unable to process uploaded image: {e}")
        return jsonify({"success": False, "error": "Unable to process the image."}), 500
