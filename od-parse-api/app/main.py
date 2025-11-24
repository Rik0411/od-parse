"""
FastAPI application entry point.
"""

import sys
from pathlib import Path

# Add parent directory to Python path to import from od_parse module
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass  # dotenv is optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings

# Get settings to configure app
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Intelligent file parser API supporting PDF, images, Excel, and vector files"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api", tags=["parsing"])


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Temp directory: {settings.temp_dir}")
    print(f"Supported extensions: {', '.join(settings.allowed_extensions)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    print(f"Shutting down {settings.app_name}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

