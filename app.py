import os
from flask import Flask, redirect, url_for, render_template
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from config import Config
from database.models import db, User

csrf = CSRFProtect()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    db.init_app(app)
    csrf.init_app(app)
    login_manager = LoginManager(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'warning'

    @login_manager.user_loader
    def load_user(user_id: int):
        return User.query.get(int(user_id))

    # Register blueprints
    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.explorer import explorer_bp
    from routes.visualization import viz_bp
    from routes.preprocessing import preprocess_bp
    from routes.ann import ann_bp
    from routes.cnn import cnn_bp
    from routes.history import history_bp
    from routes.analytics import analytics_bp
    from routes.about import about_bp
    from routes.profile import profile_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(explorer_bp)
    app.register_blueprint(viz_bp)
    app.register_blueprint(preprocess_bp)
    app.register_blueprint(ann_bp)
    app.register_blueprint(cnn_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(about_bp)
    app.register_blueprint(profile_bp)

    from utils.upload_helper import register_upload_route
    register_upload_route(app)

    # Exempt JSON API endpoints from CSRF
    csrf.exempt('ann.train_status')

    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500

    @app.route('/')
    def index():
        return redirect(url_for('dashboard.index'))

    # Create tables
    with app.app_context():
        db.create_all()

    return app


app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
