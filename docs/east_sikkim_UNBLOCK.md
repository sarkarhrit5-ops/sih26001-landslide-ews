# East Sikkim Pilot — Reproduction Unblock Guide

**Status: BLOCKED (environment), not a methodology failure.** The reproduction pipeline
exists and is sound; it cannot run in the current sandbox because that sandbox has **no
internet egress** and **cannot install the scientific Python stack**. Nothing here requires
changing the science — only granting access to the real data + libraries.

Generated 2026-08-23. Reproduction stops at Step 1 today (see bottom).

---

## What must be true to reproduce (any ONE of the three options below)

### Network egress (HTTPS / 443) — required hosts
| Host | Purpose | Auth |
|---|---|---|
| `copernicus-dem-30m.s3.amazonaws.com` | Copernicus GLO-30 DEM tiles `N27_00_E088` + `N28_00_E088` | none (public) |
| `archive-api.open-meteo.com` | **ERA5 historical daily precipitation** = the real training rainfall (antecedent, leakage-safe) | none |
| `pypi.org` + `files.pythonhosted.org` | install packages | none |
| *(optional)* `urs.earthdata.nasa.gov`, `gpm1.gesdisc.eosdis.nasa.gov` | live NASA IMERG stream **only** | Earthdata login |

> **Credentials are NOT required to reproduce the model.** Training rainfall comes from
> ERA5 via Open-Meteo (no auth). Earthdata/IMERG is only for the live real-time stream and
> is out of scope for reproduction. If IMERG creds are absent the code raises a clear
> blocker and does **not** substitute synthetic data — which is correct.

### Packages (pip, `--break-system-packages`)
```
rasterio geopandas scikit-learn lightgbm pyarrow xarray h5netcdf h5py \
shap joblib scipy requests fastapi uvicorn pydantic httpx psutil
```
Note: `backend/requirements.txt` is currently **missing** `scikit-learn`, `httpx`, and
`psutil` (all used by the code/tests). These should be added.

### Disk / RAM
- A few GB free (two DEM tiles + merged `east_sikkim_dem.tif` + derived rasters). Rasters
  are cached locally and **never committed to Git**.
- 8 GB RAM is sufficient — DEM processing is chunked (512-px windows, float32). *(This
  sandbox has only ~3.8 GB, another reason it can't exercise the full pipeline.)*

---

## Three ways to enable

**Option 1 — Run me where there IS egress (recommended; "reproduce + verify in one pass").**
Give this agent an environment with outbound access to the hosts above and the ability to
`pip install`. I then execute all 14 steps, persist real artifacts, run the tests, and
report actual metrics.

**Option 2 — Provide data locally.**
Drop `east_sikkim_dem.tif` (or the two Copernicus tiles) into `backend/data/raw/`, pre-install
the packages, and still allow `archive-api.open-meteo.com` for rainfall (or supply a
pre-fetched real ERA5 series). Without Open-Meteo egress, rainfall stays blocked — and I
will **not** fabricate it.

**Option 3 — Run it yourself on your 8 GB laptop (has internet).**
I hand you an upgraded driver that reproduces **and persists** artifacts (the current
`scripts/train_real_models.py` prints metrics but never saves them — the root cause the
pilot was unreproducible). You run one command and paste me the outputs; I then apply the
validation-status change + tests.

---

## What I will do the moment access exists (maps to your 14 steps)
1. Positives from real GLC — **already reproduced** (82 events, 72 independent dates, 78% ≥5 km uncertainty; 0 duplicates).
2. Download + verify Copernicus GLO-30 DEM; generate elevation/slope/aspect/roughness/TPI via existing chunked processing.
3. Fetch **real ERA5 antecedent rainfall** (T-14…T-1, zero future leakage) via Open-Meteo archive; document dataset explicitly.
4. Keep the elevation-binned land cover but label it a **proxy** in schema + provenance.
5. Rebuild the training matrix from real sources; save to a **new** file first; report rows/cols/balance/NaN/ranges.
6. Existing leakage-aware spatial negative sampling (≥0.05° buffer, 3:1); report class balance.
7. Train LightGBM / RandomForest / LogisticRegression on spatial + temporal holdouts; report PR-AUC, ROC-AUC, precision, recall, F1, FAR per model.
8. Run the existing Option A/C decision logic on the **real** results; document the outcome.
9. Persist `backend/data/processed/east_sikkim/{metrics.json, model_metadata.json, feature_schema.json, model.joblib}` — **no metrics hardcoded in source**.
10. Change validation-status logic so Sikkim no longer auto-claims `VALIDATED_PILOT`; require the persisted artifacts → `VALIDATED` on success, `VALIDATION_PENDING` when absent.
11. Measure peak RAM (chunked, float32, explicit cleanup).
12. Add/execute tests: matrix schema, class balance, artifact existence, metrics persistence, status logic, model load, prediction shape, **no hardcoded metrics**.
13. Write a machine-readable provenance record (datasets, URLs, timestamps, DEM resolution, rainfall dataset, AOI, event count, feature list, model, validation strategy, environment).

## Safety rules being honored
No fabricated rainfall/DEM/metrics. The old hardcoded `0.7762 / 0.9190 / …` values will **not**
be copied. `scripts/run_validation.py` will **not** be run. No synthetic substitution. The
frontend will **not** be touched.

---

### Today's actual result (this offline sandbox)
- **Step 1 (real GLC): COMPLETED** — 82 events / 72 independent dates / 78% ≥5 km / 0 dups.
- **Steps 2–13: BLOCKED** — no egress (DEM, rainfall, PyPI), no scientific packages, existing `training_matrix.parquet` unreadable (no parquet engine).
- **SCIENTIFIC STATUS: BLOCKED.** No repo files were modified.
