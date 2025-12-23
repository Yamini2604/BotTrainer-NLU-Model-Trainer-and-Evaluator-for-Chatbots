# routes/__init__.py

from fastapi import FastAPI

from .auth import router as auth_router
from .workspace import router as workspace_router
from .dataset import router as dataset_router
from .annotation import router as annotate_router

def setup_routes(app: FastAPI):
    app.include_router(auth_router)
    app.include_router(workspace_router)
    app.include_router(dataset_router)
    app.include_router(annotate_router)

