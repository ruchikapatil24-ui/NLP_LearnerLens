import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api.endpoints import router as api_router
from .services.classifier_service import model_registry

app = FastAPI(
    title="LearnLens NLP: Early Identification of Learning Difficulties",
    description="NLP-based Educational Diagnostic Platform: Weak Learner vs Advanced Learner Analysis",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes mounted under /api
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
def startup_event():
    print("[LearnLens NLP] Starting up backend server...")
    try:
        model_registry.ensure_trained()
        print("[LearnLens NLP] Machine learning models trained and initialized successfully.")
    except Exception as e:
        print(f"[LearnLens NLP] Warning during model initialization: {e}")

# Resolve static frontend directory for single-container deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    # Try sibling frontend/dist
    STATIC_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")

if os.path.exists(STATIC_DIR):
    assets_dir = os.path.join(STATIC_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Don't intercept API routes
        if full_path.startswith("api/") or full_path == "api":
            return {"error": "API route not found"}
        file_path = os.path.join(STATIC_DIR, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
else:
    @app.get("/")
    def root():
        return {
            "message": "LearnLens NLP Engine is running.",
            "documentation": "/docs",
            "health": "/api/health"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
