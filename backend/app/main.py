from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Prepare the Assam / Arunachal Pradesh / Meghalaya pilot terrain rasters ONCE, at
    startup, instead of on every /api/v1/validation/status request.

    A freshly deployed instance has never run scripts/prepare_<state>_terrain.py, so
    the five pilot rasters those dashboards read are absent and DEM status honestly
    reports missing. This hook closes that gap by reusing the existing
    acquire_state_dem() + process_dem_in_chunks() logic.

    It is idempotent (a pilot whose artifacts already exist is skipped with zero
    network I/O), it never fabricates availability (on failure the real error is
    logged and the rasters stay absent, so DEM status stays unavailable), it leaves
    Sikkim alone, and by default it runs on a background daemon thread so the app
    binds its port and serves /health immediately. Set
    SIH_PILOT_TERRAIN_BOOTSTRAP=0 to disable it entirely.
    """
    from app.services.pilot_terrain_bootstrap import start_pilot_terrain_bootstrap
    start_pilot_terrain_bootstrap()
    yield


app = FastAPI(
    title="SIH 2026 EWS API",
    description="Backend API for Landslide Early Warning System",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://sih26001-landslide-ews.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "healthy"}
