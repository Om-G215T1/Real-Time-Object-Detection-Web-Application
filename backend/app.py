# backend/app.py
# Flask application entry point

from flask import Flask, send_from_directory
from flask_cors import CORS
from backend.routes import api
from utils.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG
import os


def create_app() -> Flask:
    """
    Application factory — creates and configures Flask app.
    """
    app = Flask(
        __name__,
        static_folder='../frontend'
    )

    # Allow cross-origin requests (needed for dev)
    CORS(app)

    # Register API routes under /api prefix
    app.register_blueprint(api, url_prefix='/api')

    # Serve frontend
    @app.route('/')
    def index():
        return send_from_directory('../frontend', 'index.html')

    @app.route('/<path:path>')
    def static_files(path):
        return send_from_directory('../frontend', path)

    # Global error handlers
    @app.errorhandler(404)
    def not_found(e):
        return {'error': 'Route not found'}, 404

    @app.errorhandler(500)
    def server_error(e):
        return {'error': 'Internal server error'}, 500

    @app.errorhandler(413)
    def too_large(e):
        return {'error': 'File too large'}, 413

    return app


if __name__ == '__main__':
    app = create_app()
    print(f"\n🚀 Starting Flask server...")
    print(f"   URL     : http://localhost:{FLASK_PORT}")
    print(f"   Debug   : {FLASK_DEBUG}")
    print(f"   API     : http://localhost:{FLASK_PORT}/api/health\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True       # Handle multiple requests concurrently
    )