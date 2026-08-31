from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Make the Assam / Arunachal Pradesh / Meghalaya pilot terrain rasters available ONCE,
    at startup, instead of on every /api/v1/validation/status request.

    A freshly deployed instance has never run scripts/prepare_<state>_terrain.py, so the
    five pilot rasters those dashboards read are absent and DEM status honestly reports
    missing. Two paths can close that gap, both idempotent, both on a background daemon
    thread so the app binds its port and serves /health immediately, and neither ever
    fabricating availability -- on failure the real error is logged and the rasters stay
    absent:

      1. DOWNLOAD the prebuilt artifacts from object storage (the production path).
         Off unless SIH_PILOT_ARTIFACT_BASE_URL is configured. Memory-flat.
      2. REGENERATE them from Copernicus GLO-30. Off unless
         SIH_PILOT_TERRAIN_BOOTSTRAP=1, because the uncapped mosaic merge OOM-killed the
         Render instance (exit status 137).

    Both leave Sikkim alone, and with neither configured this hook is a no-op.
    """
    from app.services.pilot_artifact_store import start_pilot_artifact_fetch
    from app.services.pilot_terrain_bootstrap import start_pilot_terrain_bootstrap
    start_pilot_artifact_fetch()
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
    "https://sih26001-landslide-ftmr1p1m4-lazy-coders2.vercel.app",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api/v1")


@app.get("/health")
def health_check():
    return {"status": "healthy"}
