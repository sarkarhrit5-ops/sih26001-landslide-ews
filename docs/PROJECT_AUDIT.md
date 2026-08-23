# SIH26001 — Landslide Early Warning System · Technical Project Audit

**Audit date:** 2026-08-23 · **Auditor role:** incoming developer (read-only inspection) · **Repo:** `D:\landslide\sih26001-landslide-ews`
**No files were modified. Nothing was installed or run destructively. This is an inspection only.**

Legend for state tags used throughout:
- **IMPLEMENTED** — code exists and appears functional by reading it
- **TESTED** — exercised by an automated test in `backend/tests`
- **VALIDATED** — backed by reproducible scientific/ML evidence *in the repo*
- **PLANNED** — referenced/declared but not built
- **BLOCKED** — cannot run/complete without a missing external resource

> Headline: This is a **well-structured research prototype** with genuinely careful geospatial/ML *code*, wrapped around a **demo runtime** that currently serves mostly hardcoded/mock values. The single "validated" claim (East Sikkim pilot) is **asserted via hardcoded numbers, not reproducible from the repo** (no DEM, no model artifact, metrics never persisted). Treat all performance figures as unverified until re-run.

---

## 1. What the project is

An AI-driven **dynamic landslide risk monitoring and early-warning system for Northeast India**, built for Smart India Hackathon 2026 (problem statement: *AI-Based Early Warning and Landslide Risk Monitoring*).

- **"NER" = North Eastern Region — the 8 NE India states**, not Named Entity Recognition. States: Sikkim, Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Tripura (`backend/app/core/config_states.py`).
- **Pilot area:** East Sikkim (27.0–28.1 N, 88.0–88.9 E).
- **Stated philosophy** (root `README.md`): scientific validity, accurate data, spatial/temporal validation, explainability, reproducibility, real-world feasibility. README self-describes status as *"Research and architecture phase."*
- **Chosen methodology = "Option C"**: static terrain **susceptibility** × empirical **rainfall-trigger** thresholds × **forecast** escalation × **exposure**, fused into a risk score + warning level. ("Option A" = full temporal ML, deliberately rejected due to sparse events / high location uncertainty — see §8.)

---

## 2. Current architecture

```
Frontend (React 19 + Vite + TS + Tailwind v4 + Leaflet)
   src/services/api.ts  ──HTTP──►  FastAPI backend (backend/app)
                                     ├─ api/routes.py         (6 endpoints)
                                     ├─ models/ml_pipeline.py (risk fusion, baselines, SHAP)
                                     ├─ models/thresholds.py  (rainfall power-law)
                                     └─ services/
                                          terrain_processing.py (DEM → slope/aspect/roughness/TPI)
                                          weather_ingestion.py  (Open-Meteo + NASA IMERG)
                                          exposure.py           (OSM/Overpass + spatial join)
                                          state_validation.py   (per-state pipeline)
                                          label_gate.py         (Option A vs C decision)
Data (flat files, no DB):
   backend/data/raw/       glc_legacy.csv, {sikkim,arunachal_pradesh,assam}_osm.geojson
   backend/data/processed/ state_validation.json, training_matrix.parquet
Batch scripts: scripts/{train_real_models, run_state_validation, run_validation, run_imerg_smoke_test, download_real_glc}.py
```

- **Backend:** FastAPI, Python (compiled bytecode indicates 3.14). Layered app/service structure, clean separation.
- **Frontend:** SPA, no router (a `useState` toggles landing⇄dashboard), Leaflet used via its imperative API (not react-leaflet, though it's installed).
- **Persistence:** **flat files only.** `sqlalchemy` + `psycopg2-binary` are in `requirements.txt` but there is **zero** database code anywhere (`grep` for create_engine/sessionmaker/declarative_base → nothing). → **DB is PLANNED, not implemented.**
- **8 GB RAM constraint** is explicitly engineered for: DEM processing is chunked (512-px windows), IMERG uses server-side spatial subsetting, parquet uses float32.

---

## 3. What is actually implemented

**IMPLEMENTED (verified by reading code):**
- **FastAPI app + 6 endpoints** (`main.py`, `api/routes.py`): `/health`, `/api/v1/risk/current`, `/risk/forecast`, `/cell/{id}/explain`, `/exposure/alerts`, `/validation/status`. *(caveat: inputs are hardcoded — see §6/§11).*
- **Risk-fusion engine** (`ml_pipeline.dynamic_risk_module`): susceptibility × trigger-multiplier (rainfall + slope escalation), warning-level bucketing (EXTREME/HIGH/MEDIUM/LOW), confidence flagging.
- **Empirical rainfall thresholds** (`thresholds.py`): power-law `I = 14.2·D^(-0.62)`, critical intensity/accumulation, trigger evaluation.
- **Terrain/DEM processing** (`terrain_processing.py`): Horn's-method slope/aspect, roughness, TPI; chunked windowed processing with 1-px overlap; boundary-artifact verification. Real, careful code.
- **Weather ingestion** (`weather_ingestion.py`): real Open-Meteo forecast (no auth); real NASA IMERG via Earthdata OPeNDAP subsetting with proper auth handling; NetCDF parse via xarray/h5netcdf.
- **OSM exposure** (`exposure.py`): real Overpass query + GeoDataFrame + disk cache (`get_osm_assets`); spatial-join exposure analysis (`analyze_exposure`).
- **Per-state validation pipeline** (`state_validation.py` + `scripts/run_state_validation.py`): inventory quality, DEM acquisition (Copernicus 30 m from AWS S3), OSM acquisition, rainfall-auth check, blocker/status determination; writes `state_validation.json`.
- **Real modeling pipeline** (`scripts/train_real_models.py`): downloads Copernicus DEM → terrain features → real GLC positives → spatial buffered negatives → **real ERA5 antecedent rainfall** (Open-Meteo archive, leakage-safe T-14…T-1) → LightGBM/RF/LogReg on spatial & temporal holdouts → prints metrics.
- **Frontend UI**: landing page (+ animated canvas hero), dashboard with Leaflet map (CartoDB dark tiles), 6 toggleable layers, intelligence panel (regional/state/cell views), warning badges, legends.

**PLANNED (declared, not built):** database/persistence layer; alerting/notification delivery; user auth; real land-cover data; per-state model training beyond the pilot.

---

## 4. What is tested

**TESTED — 30 automated tests across 6 files** (`backend/tests`). All are designed to pass **without** network, credentials, DEM, or a model, because every real dependency is mocked or guarded.

| File | Tests | Covers | Real or mocked |
|---|---|---|---|
| test_api.py | 7 | all endpoints (shape/plumbing) | in-process app; internals hardcoded/mock |
| test_risk_fusion.py | 3 | fusion formula, warning levels, metadata | deterministic literals |
| test_thresholds.py | 6 | rainfall power-law math | pure numpy (only bulletproof file) |
| test_weather_and_dem.py | 2 | Earthdata-missing guard; terrain shapes | env-pop + synthetic 5×5 DEM |
| test_state_validation.py | 6 | inventory dedup, status/blocker tree | fake DataFrame fixture |
| test_weather_ingestion_imerg.py | 6 | grid index, cred guards, aggregation, NetCDF parse | unittest.mock + synthetic NetCDF |

**Not covered by any test:** the entire ML training/holdout/metrics path (`generate_spatial_negative_samples`, `run_*_holdout_validation`, `train_and_evaluate_baselines`, `evaluate_model_decision`), real DEM processing (`process_dem_in_chunks`, `verify_*`), real network services (Overpass, Open-Meteo, real IMERG fetch), spatial-join exposure, the `scripts/`, and any persistence.

**Config risks:** no `pytest.ini`/`conftest.py`/`__init__.py` (tests use `sys.path.append` shims). `httpx` (needed by `TestClient`) and `scikit-learn` (imported in `ml_pipeline.py`) are **not in `requirements.txt`** → collection could error before tests even run.

**No test asserts a real scientific metric.** At most, tests check that hardcoded metric dicts contain the key `"PR-AUC"`.

---

## 5. What is scientifically validated

**VALIDATED (reproducibly, in the repo): essentially nothing.** This is the most important finding.

- The pilot performance numbers — **PR-AUC 0.7762, ROC-AUC 0.9190, FAR 0.0317, Precision 0.7778, Recall 0.3684, F1 0.5000** — appear **identically and hardcoded** in three places: `ml_pipeline.STATIC_MODEL_METADATA`, `state_validation.determine_overall_status` (pilot branch), and `data/processed/state_validation.json`.
- The only code that could generate them (`train_real_models.py`) **prints metrics to stdout and persists nothing** — no metrics file, **no saved model artifact** (`grep` for joblib/pickle/`.pkl` → none), and **no `.tif` DEM exists in the repo**. So the figures were hand-copied from a presumed one-off run and **cannot currently be reproduced or verified from the repository**.
- `state_validation.json` marks Sikkim `"VALIDATED_PILOT"`, but that string is **assigned unconditionally for the pilot** in `determine_overall_status` regardless of whether a model ran.

**Methodologically sound *by design*** (the code implements these correctly, which is a real positive): leakage-aware negative sampling (≥5 km spatial buffer), spatial holdout (latitude split) and temporal holdout (≤2014 train / ≥2015 test), strictly-antecedent rainfall features, and honest self-documented uncertainty (threshold metadata notes 78% of catalog events have ≥5 km location error; calibration note says the score is a *relative* index, not an absolute probability). But **design ≠ validated results present in the repo.**

Bottom line: treat the pilot as **"claimed / not reproducibly validated"** until `train_real_models.py` is re-run and its outputs (model + metrics) are persisted and checked in.

---

## 6. What is only mocked / test data

- **Live API risk is fake:** `routes.py` hardcodes `base_susceptibility = 0.65`, `current_rain = 55.0`, `slope = 35.0`, `exposure_score = 0.5`, and passes `has_real_dem=True, has_real_rainfall=True` regardless. So `/risk/current` returns a computed-but-meaningless number.
- **Exposure endpoint uses `mock_get_osm_assets()`** — 2 hardcoded points ("NH-10", "STNM Hospital") — **not** the real cached OSM geojson that exists on disk.
- **SHAP explanation is a hardcoded fallback:** `/cell/{id}/explain` calls `explain_risk(None, None)`, which always returns the static list (slope 0.42, rain_3d 0.28, roughness 0.18). The real SHAP branch is never reached at runtime.
- **Frontend is mock-driven:** all map polygons, risk numbers, exposure points, and historical events come from `src/data/mockCells.ts` and in-component arrays. "Forecast risk" = `finalRisk × 1.15` computed client-side. The honesty banner (`DataStateBanner`) is pinned to `"UNAVAILABLE"`. Only 2 of 6 API calls are wired (`getValidationStatus`, `getCellExplanation`), both with silent fallbacks to mock content.
- **`scripts/run_validation.py` is a synthetic-plumbing script** (random DEM, 45 synthetic labels, random slope/aspect) — and it's **stale/broken**: it calls `process_dem_in_chunks(..., overlap=1)` but the current signature has no `overlap` param, so it would crash. If it did run, it would **overwrite `training_matrix.parquet` with random data.** Do not run it.
- **`land_cover_class` is an elevation-based proxy**, not real land-cover data (`train_real_models.assign_land_cover_proxy`).
- **Rainfall fallback labeling:** `state_validation.json` shows every state's rainfall as `"Open-Meteo / Fallback Synthetic"` — i.e., no authenticated IMERG was used even for the pilot.

---

## 7. Current datasets

| Dataset | Location | Real? | Notes |
|---|---|---|---|
| NASA Global Landslide Catalog | `data/raw/glc_legacy.csv` | **Real** | ~11,059 rows, global; source data.nasa.gov. Filtered per-state by bbox. |
| OSM exposure — Sikkim | `data/raw/(osm/)sikkim_osm.geojson` | **Real** | 62 features (real hospitals/roads w/ coords). |
| OSM exposure — Arunachal | `…/arunachal_pradesh_osm.geojson` | **Real** | 24 features. |
| OSM exposure — Assam | `…/assam_osm.geojson` | **Real** | 221 features. |
| Training matrix | `data/processed/training_matrix.parquet` | **Unverified** | 81 KB; consistent with a real `train_real_models.py` run (~328 rows) but could also be a synthetic `run_validation.py` output. Not readable here (no pyarrow); **verify contents before trusting.** |
| State validation report | `data/processed/state_validation.json` | Report | 8 states; partially **stale** (says Arunachal/Assam OSM "Missing" though geojson exist). |
| **DEM (any `.tif`)** | — | **ABSENT** | No `east_sikkim_dem.tif`, no Copernicus tiles, no derived slope/aspect rasters anywhere. |
| **Trained model** | — | **ABSENT** | No `.pkl`/`.joblib`/`.onnx`. |
| IMERG cache / `.nc4` | — | **ABSENT** | Requires Earthdata creds to fetch. |

Only **3 of 8** states have OSM data; **0 of 8** have DEM committed.

---

## 8. Current model methodology

**Option C — structured risk fusion** (`dynamic_risk_module`):
1. `susceptibility_score` ∈ [0,1] (intended from a static terrain model).
2. Current & forecast **rainfall triggers** via power-law `I_crit = 14.2·D^(-0.62)` (East-Sikkim-derived, 82 events, ERA5-paired, compared to Caine 1980 `14.82·D^(-0.39)`; explicitly **not universal**, MEDIUM confidence).
3. **Escalation multiplier**: +0.4 if current trigger exceeded, +0.3 if forecast exceeded, +0.2 if slope ≥ 35° and any trigger fires.
4. `final_risk = clamp(susceptibility × multiplier, 0, 1)` → warning level (≥0.85 EXTREME, ≥0.65 HIGH, ≥0.40 MEDIUM, else LOW). Confidence from data-availability flags.

**Susceptibility model (baselines):** LightGBM + RandomForest + LogisticRegression on 6 static features (elevation, slope, aspect, roughness, TPI, land_cover proxy) ± 5 rainfall features (1d/3d/7d/14d-antecedent/3d-max). Leakage controls: 5 km buffered negatives (3:1), spatial (latitude) and temporal (2014 cutoff) holdouts. **Option A vs C gate** (`label_gate.py`, `evaluate_model_decision`): if <100 independent events, >50% high spatial uncertainty, or unstable holdout FAR/PR-AUC → fall back to Option C. Given only 82 pilot events with 78% ≥5 km error, Option C is the justified choice.

**Explainability:** SHAP `TreeExplainer` intended; at runtime only the **hardcoded fallback** importances are served.

---

## 9. Current NER state coverage

From real GLC filtering (`state_validation.json`); DEM absent for all, OSM present only where noted:

| State | GLC events | OSM data | Status in report | Blockers |
|---|---|---|---|---|
| **Sikkim** (pilot) | 82 | ✅ (62) | `VALIDATED_PILOT` *(hardcoded)* | none listed |
| Arunachal Pradesh | 88 | ✅ (24) | DATA UNAVAILABLE | Missing DEM, OSM* |
| Assam | 401 | ✅ (221) | DATA UNAVAILABLE | Missing DEM, OSM* |
| Manipur | 128 | ❌ | DATA UNAVAILABLE | Missing DEM, OSM |
| Nagaland | 110 | ❌ | DATA UNAVAILABLE | Missing DEM, OSM |
| Meghalaya | 47 | ❌ | DATA UNAVAILABLE | + insufficient events (<50) |
| Mizoram | 48 | ❌ | DATA UNAVAILABLE | + insufficient events (<50) |
| Tripura | 11 | ❌ | DATA UNAVAILABLE | + insufficient events (<50) |

\*Report says OSM "Missing" for Arunachal/Assam although geojson files now exist → **stale JSON**. Only **Sikkim** has any frontend map cells (6, hardcoded). Effective validated coverage = **1 of 8 states, and even that is a claim not reproducible in-repo.**

---

## 10. Current frontend functionality

- **Works standalone (no backend), from mock data:** landing page + animated hero; dashboard layout; Leaflet map (CartoDB dark tiles, NE-India view); 6 layers — risk, susceptibility, forecast(72h only), rainfall(shows "UNAVAILABLE"), exposure(6 hardcoded points), events(4 hardcoded); layer toggles; fly-to on selection; 8 clickable state labels; intelligence panel (regional list / state checklist / cell detail with progress meters); warning badges; bottom info strip (hardcoded rainfall windows).
- **Wired to backend (optional, silent fallback):** `getValidationStatus()` → state badges/checklists; `getCellExplanation()` → SHAP-style attributions (the one genuinely backend-driven widget when the API is up).
- **Defined but never called:** `checkHealth`, `getCurrentRisk`, `getForecastRisk`, `getExposureAlerts`.
- **Honesty signal:** `DataStateBanner` supports LIVE/HISTORICAL/FORECAST/UNAVAILABLE but is rendered once, pinned to **UNAVAILABLE** — no dynamic real-vs-synthetic detection.
- **Cruft:** unused deps (`clsx`, `tailwind-merge`, `react-leaflet`), orphan `App.css`, default `index.html` title.
- **Assessment:** high-quality **demo UI**; not yet a live operational dashboard.

---

## 11. Current backend APIs

Base: `/` and `/api/v1` (dev proxy → `127.0.0.1:8000`).

| Endpoint | Method | Real data? | Notes |
|---|---|---|---|
| `/health` | GET | n/a | returns `{status: healthy}`. **IMPLEMENTED/TESTED** |
| `/api/v1/risk/current` | GET | **No** | hardcoded susceptibility/rain/slope. **IMPLEMENTED (demo)/TESTED** |
| `/api/v1/risk/forecast` | GET | Partial | real Open-Meteo forecast *if reachable* (exception→0); rest hardcoded. **IMPLEMENTED/TESTED** |
| `/api/v1/cell/{id}/explain` | GET | **No** | always hardcoded SHAP fallback. **IMPLEMENTED/TESTED** |
| `/api/v1/exposure/alerts` | GET | **No** | mock 2-point OSM, not the real geojson. **IMPLEMENTED/TESTED** |
| `/api/v1/validation/status` | GET | Static file | serves `state_validation.json` (partly stale). **IMPLEMENTED/TESTED** |

No auth, no rate-limiting, no DB-backed endpoints, no write/ingestion endpoints, no alerting endpoint.

---

## 12. Current blockers

- **BLOCKED — NASA Earthdata credentials** (`EARTHDATA_TOKEN` or `EARTHDATA_USERNAME/PASSWORD`) absent → no real IMERG rainfall. Code correctly refuses to substitute synthetic IMERG. No `.env`/`.env.example` present.
- **BLOCKED — No DEM in repo** → terrain features can't be produced; the entire susceptibility path can't run without first downloading Copernicus tiles (needs network + disk).
- **BLOCKED/RISK — Pilot metrics unreproducible**: no persisted model, no persisted metrics, no committed DEM. Cannot verify the headline numbers.
- **RISK — `requirements.txt` incomplete**: missing `scikit-learn` and `httpx` (and `psutil`, used by every script). Test collection and scripts may fail on a clean install.
- **BROKEN — `scripts/run_validation.py`**: stale `overlap=` arg → crashes; would overwrite the parquet with synthetic data if "fixed" naively.
- **STALE — `state_validation.json`** disagrees with on-disk OSM; regenerate.
- **No DB** despite dependencies — no persistence/history/audit trail.
- **Git hygiene** — latest commit named "error"; `data/` gitignored yet force-tracked; CRLF/LF churn (see §14).

---

## 13. Missing functionality

Persistence/database layer (or removal of unused SQL deps); model artifact **save/load** + metrics persistence (JSON/MLflow-lite) so validation is reproducible; committed or scripted-on-demand DEM acquisition with caching; real land-cover integration (replace elevation proxy); wiring the real endpoints into the frontend + a working DataStateBanner; alerting/notification delivery (SMS/email/push) — core to an "early warning system"; scheduled/near-real-time ingestion loop; authentication & basic rate-limiting; tests for the real ML/DEM/network paths (currently untested); an `.env.example` + setup/run docs; per-state susceptibility beyond the pilot.

---

## 14. Current Git status

- **Branch** `main`, tracking `origin/main` (**up to date** — commits are pushed). Remote: `github.com/sarkarhrit5-ops/sih26001-landslide-ews`.
- **12 commits**, ending `e9ad7b9 "error"` (Gaurav Singh, 2026-08-23 16:14 IST), preceded by `c7f42d9 "logo"`, `0a927ab feat: NASA IMERG INTEGRATION`, `e0c0b6a feat: add NER state validation pipeline`, etc.
- **Working tree shows 38 "modified" files but this is 100% line-ending (CRLF↔LF) noise** — the whitespace-ignoring diff is empty and insertions exactly equal deletions (18385/18385). **No real uncommitted content changes.** Likely a Windows `core.autocrlf` mismatch.
- **No untracked files.** `data/raw/*` and `data/processed/*` are in `.gitignore` yet tracked (force-added earlier), so data edits show as diffs. `.gitignore` references `!.env.example`, but no such file exists.
- Note: `D:\landslide` (the parent folder) is itself a separate git repo; the project is the `sih26001-landslide-ews` subfolder.

---

## 15. Recommended next steps

Ordered; each respects the 8 GB constraint and the "no overclaiming" principle. **Awaiting your go-ahead before changing anything.**

**A. Establish ground truth (do first, low risk)**
1. Inspect `training_matrix.parquet` contents (row count, columns, target balance) to confirm it's the real 328-row dataset vs synthetic — decide keep/regenerate.
2. Re-run `scripts/train_real_models.py` end-to-end once (downloads DEM + ERA5), then **persist** the trained model and a `metrics.json`; wire `STATIC_MODEL_METADATA` and the pilot block in `state_validation.json` to read those files instead of hardcoded literals. This converts "claimed" → **VALIDATED**.
3. Regenerate `state_validation.json` so it matches on-disk OSM/DEM.

**B. Fix correctness/repro hazards**
4. Add `scikit-learn`, `httpx`, `psutil` to `requirements.txt`; pin versions.
5. Quarantine or repair `scripts/run_validation.py` (remove `overlap=`, stop it clobbering the real parquet).
6. Add `.env.example` (Earthdata vars) + a short RUN/README with setup steps.
7. Normalize line endings (`.gitattributes` + renormalize) and untangle the gitignored-but-tracked `data/`.

**C. Close the demo→product gap (only after A/B, and only what you approve)**
8. Wire real endpoints into the frontend; make `DataStateBanner` reflect real vs mock; stop the client-side `×1.15` forecast.
9. Point `/exposure/alerts` at the real OSM geojson instead of the mock.
10. Add tests for the real ML/DEM/threshold-fit paths and a tiny fixture DEM.
11. Decide on persistence: implement a lightweight store (SQLite fits 8 GB) or drop the unused SQL deps.
12. Design the actual **warning-delivery** mechanism (the "EW" in EWS).

**D. Scientific credibility**
13. Document the empirical threshold derivation and, if feasible, re-fit `14.2 / -0.62` from data with a goodness-of-fit; keep the "relative index, not calibrated probability" caveat prominent.
14. Replace the elevation-based land-cover proxy with real land-cover (e.g., ESA WorldCover) or clearly label it everywhere.

---

*Prepared from direct inspection of source, data, tests, and git. Performance figures in this repo are hardcoded and were not reproduced during this read-only audit.*
