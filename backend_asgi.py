"""ASGI entry point for a separately deployed Vite frontend.

Run from the repository root with:
    uvicorn backend_asgi:app --host 0.0.0.0 --port 5004

Set FRONTEND_ORIGINS to a comma-separated list of deployed frontend origins.
"""

import os

from fastapi.middleware.cors import CORSMiddleware

from main import app


origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
