import logging
from flask import Flask
from config import Config
from app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    try:
        app.prediction_service = PredictionService(app.config['MODEL_PATH'])
    except Exception as e:
        app.logger.error(f"Initialization error: {e}")
        app.prediction_service = None

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return {"success": False, "error": "File size exceeds the configured limit."}, 413


    from app.routes import bp as main_bp
    app.register_blueprint(main_bp)

    return app
