"""
Dwell.ai API v2.0

FastAPI application for the Generative 3D Interior Design Agent.

Features:
- AI-powered layout generation with multiple variations
- Photorealistic perspective rendering
- Conversational editing interface
- LangSmith tracing for observability
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings, setup_langsmith
from app.models.api import HealthResponse, ErrorResponse
from app.routes import analyze, optimize, chat, render, shop
from app.core.exceptions import (
    PocketPlannerError,
    VisionExtractionError,
    ConstraintViolationError,
    RenderingError,
    InvalidImageError,
)


# Get settings
settings = get_settings()

# Setup LangSmith tracing
langsmith_enabled = setup_langsmith()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    **Dwell.ai API v2.0** - AI-Powered Generative 3D Interior Design Agent.
    
    ## Features
    - **Analyze**: Extract 3D floor plan understanding from room photos
    - **Optimize**: Generate multiple AI-powered layout variations
    - **Render**: Create photorealistic perspective renders
    - **Edit**: Conversational image editing interface
    
    ## AI Models Used
    - `gemini-2.5-pro`: Vision analysis and layout generation
    - `gemini-2.5-flash-preview-05-20`: Photorealistic rendering
    
    ## Workflow
    1. Upload a room photo → `/api/v1/analyze`
    2. Lock objects you want to keep in place
    3. Request AI optimization → `/api/v1/optimize`
    4. Select from 3 layout variations
    5. Render the result → `/api/v1/render`
    """,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analyze.router, prefix=settings.api_prefix)
app.include_router(optimize.router, prefix=settings.api_prefix)
app.include_router(render.router, prefix=settings.api_prefix)
app.include_router(chat.router, prefix=settings.api_prefix)
app.include_router(shop.router, prefix=settings.api_prefix)


# ============ Exception Handlers ============

@app.exception_handler(VisionExtractionError)
async def vision_extraction_error_handler(request: Request, exc: VisionExtractionError):
    """Handle vision extraction failures."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.message, "error_code": exc.error_code}
    )


@app.exception_handler(ConstraintViolationError)
async def constraint_violation_error_handler(request: Request, exc: ConstraintViolationError):
    """Handle constraint violations."""
    return JSONResponse(
        status_code=400,
        content={
            "detail": exc.message,
            "error_code": exc.error_code,
            "violations": exc.violations
        }
    )


@app.exception_handler(RenderingError)
async def rendering_error_handler(request: Request, exc: RenderingError):
    """Handle rendering failures."""
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message, "error_code": exc.error_code}
    )


@app.exception_handler(InvalidImageError)
async def invalid_image_error_handler(request: Request, exc: InvalidImageError):
    """Handle invalid image data."""
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message, "error_code": exc.error_code}
    )


@app.exception_handler(PocketPlannerError)
async def pocket_planner_error_handler(request: Request, exc: PocketPlannerError):
    """Handle generic Dwell.ai errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message, "error_code": exc.error_code}
    )


# ============ Health Check ============

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        message="Dwell.ai API is running. Visit /docs for API documentation."
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=settings.app_version
    )


# ============ Run with Uvicorn ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
