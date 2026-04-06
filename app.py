from __future__ import annotations

import os

import uvicorn

from api.main import app as application

# Keep both names so Azure/Gunicorn and local tooling can detect the app.
app = application


if __name__ == "__main__":
	port = int(os.getenv("PORT", "8000"))
	uvicorn.run("api.main:app", host="0.0.0.0", port=port)
