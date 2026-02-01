"""
Pocket Planner API

FastAPI application for the Small Space Optimization Agent.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.api import HealthResponse
from app.routes import analyze, optimize, render


# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    **Pocket Planner API** - An agentic spatial planner that optimizes small rooms.
    
    ## Features
    - **Analyze**: Extract furniture and room layout from photos
    - **Optimize**: Intelligently reposition furniture while respecting locked objects
    - **Render**: Generate edited images showing the optimized layout
    
    ## Workflow
    1. Upload a room photo → `/api/v1/analyze`
    2. Lock objects you want to keep in place
    3. Request optimization → `/api/v1/optimize`
    4. Render the result → `/api/v1/render`
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


# ============ Health Check ============

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        message="Pocket Planner API is running. Visit /docs for API documentation."
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
