import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.affirmation.affirmation_route import router as affirmation_router
from app.vectordb.vectordb_route import router as alignment_router
from app.services.quiz_evaluation.quiz_evaluation_route import router as quiz_evaluation_router

app = FastAPI(
    title="Sui Amor",
    description="An AI-powered perfume buying app that helps users find their perfect scent match based on personal preferences and characteristics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(affirmation_router)
app.include_router(alignment_router)
app.include_router(quiz_evaluation_router)


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint for health checks"""
    return {
        "message": "Welcome to Sui Amor!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Docker/monitoring"""
    return {
        "status": "healthy",
        "service": "Sui Amor"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )