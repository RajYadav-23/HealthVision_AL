import os
from flask import send_from_directory
from config import Config


def register_upload_route(app):
    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(Config.UPLOAD_FOLDER, filename)
