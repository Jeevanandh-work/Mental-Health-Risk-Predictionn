from api.main import app as application

# Keep both names so Azure/Gunicorn and local tooling can detect the app.
app = application
