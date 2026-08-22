from fastapi import FastAPI
from app.api import routes

app = FastAPI(
    title="SIH26001 EWS API",
    description="Backend API for Landslide Early Warning System",
    version="1.0.0"
)

app.include_router(routes.router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {"status": "healthy"}
