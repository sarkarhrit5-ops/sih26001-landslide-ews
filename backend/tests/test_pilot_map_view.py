"""
Offline tests for app.services.pilot_map_view -- the lightweight map projection of
an already-computed /predict/<state>/grid prediction.

Dependency budget: stdlib only. That is the point of the module under test: it
runs no model, fetches no rainfall, opens no raster and reads no file, so it needs
nothing beyond a dict.

What is pinned here:

  1. FIELD CONTRACT. Exactly the documented top-level keys, and per cell exactly
     cell_id / status / probability / risk_class / exceeds_decision_threshold.
     The 11-value feature vector, the cell bbox, the per-cell reasons and the
     row/col indices are ABSENT anywhere in the document.
  2. SIZE. The map payload is materially smaller than the grid payload it was
     projected from, for all four pilot states.
  3. HONESTY. Cells are copied, never filtered: an UNAVAILABLE cell survives with
     probability None. Rainfall provenance is copied verbatim, so a FALLBACK
     series is still labelled FALLBACK here, and `generated_from` is whatever the
     producer said -- this module never writes one.
  4. GEOMETRY. GeoJSON coordinates are [lon, lat], the opposite order to
     Leaflet's LatLng, at the cell CENTER the model was sampled at.
  5. It REFUSES rather than inventing a map when handed something that is not a
     pilot grid prediction.
  6. The four prediction services no longer hard-code "real IMERG" in
     `generated_from` (source introspection, no import needed).
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import pilot_map_view as pmv  # noqa: E402

REPO_SERVICES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'app', 'services')
)

# (state, pilot_area, canonical AOI) for the four pilots.
PILOTS = [
    ("Sikkim", "East Sikkim pilot AOI",
     {"min_lat": 27.0, "max_lat": 28.1, "min_lon": 88.0, "max_lon": 88.9}),
    ("Assam", "Guwahati-Karbi pilot AOI",
     {"min_lat": 25.6, "max_lat": 26.6, "min_lon": 91.3, "max_lon": 93.7}),
    ("Arunachal Pradesh", "Subansiri-Siang pilot AOI",
     {"min_lat": 26.5, "max_lat": 27.99, "min_lon": 92.0, "max_lon": 94.5}),
    ("Meghalaya", "East Khasi + Jaintia Hills pilot AOI",
     {"min_lat": 25.0, "max_lat": 25.99, "min_lon": 91.0, "max_lon": 92.8}),
]

_REAL_RAINFALL = {
    "source": "IMERG_Early",
    "source_kind": "IMERG",
    "run_type": "Early",
    "is_fallback": False,
    "data_quality_status": "REAL",
    "units": "mm",
    "requested_date": "2026-08-30",
    "rainfall_observation_date": "2026-08-29",
    "fetched_at_utc": "2026-08-30T06:00:00Z",
    "freshness": {"cache_hit": False, "probe_reach": 3},
    "window_days": 14,
    "aoi_uniform": True,
    "note": "Antecedent-only (T-1..T-14, event day excluded).",
    "daily_series_mm": [1.5] * 14,
    "features": {"rain_1d_mm": 1.5, "rain_3d_mm": 4.5, "rain_7d_mm": 10.5,
                 "rain_14d_mm": 21.0, "api_mm": 7.0},
    "coverage": {"state": "Sikkim", "aoi_uniform": True, "window_days": 14,
                 "window_semantics": "T-1..T-14"},
}

_FALLBACK_RAINFALL = dict(
    _REAL_RAINFALL,
    source="Open-Meteo ERA5 archive (FALLBACK)",
    source_kind="OPEN_METEO_FALLBACK",
    is_fallback=True,
    data_quality_status="FALLBACK",
    caveats=["Reanalysis, not a live satellite observation."],
)


def _cell(row, col, lat, lon, scored=True):
    """A full grid cell, exactly as the prediction services emit one."""
    cell = {
        "cell_id": "r%02dc%02d" % (row, col),
        "row": row,
        "col": col,
        "latitude": lat,
        "longitude": lon,
        "bbox": {"min_lat": lat - 0.05, "max_lat": lat + 0.05,
                 "min_lon": lon - 0.05, "max_lon": lon + 0.05},
    }
    if scored:
        cell.update({
            "status": "OK",
            "susceptibility_probability": 0.1 * (row + col + 1),
            "risk_class": "MODERATE",
            "exceeds_decision_threshold": False,
            "features": {
                "elevation": 1483.2734375, "slope": 24.117645263671875,
                "aspect": 181.40626525878906, "curvature": 0.011342163197696209,
                "twi": 6.204118728637695, "land_cover_class": 2,
                "rain_1d_mm": 1.5399999618530273,
                "rain_3d_mm": 4.610000133514404,
                "rain_7d_mm": 10.520000457763672,
                "rain_14d_mm": 21.049999237060547,
                "api_mm": 7.032187461853027,
            },
            "reasons": [],
        })
    else:
        cell.update({
            "status": "UNAVAILABLE",
            "susceptibility_probability": None,
            "risk_class": None,
            "exceeds_decision_threshold": None,
            "features": None,
            "reasons": ["terrain sample is nodata at this cell centre"],
        })
    return cell


def _prediction(state, pilot_area, aoi, rainfall=None, n_ok=6, n_unavailable=2):
    cells = []
    lat = aoi["min_lat"] + 0.05
    lon = aoi["min_lon"] + 0.05
    for i in range(n_ok):
        cells.append(_cell(i // 3, i % 3, lat + 0.1 * i, lon + 0.1 * i))
    for j in range(n_unavailable):
        cells.append(_cell(9, j, lat + 0.5 + 0.1 * j, lon + 0.5, scored=False))
    return {
        "state": state,
        "pilot_area": pilot_area,
        "generated_from": ("persisted LightGBM (static_plus_rainfall, 11 features) "
                           "+ real IMERG_Early antecedent rainfall"),
        "target_date": "2026-08-30",
        "aoi": dict(aoi),
        "grid": {"step_deg": 0.1, "rows": 3, "cols": 3, "cells": len(cells)},
        "decision_threshold": 0.4315,
        "model": {"artifact": "fixture", "features": 11},
        "rainfall": dict(_REAL_RAINFALL if rainfall is None else rainfall),
        "summary": {
            "cells_total": len(cells),
            "cells_scored": n_ok,
            "cells_unavailable": n_unavailable,
            "risk_class_counts": {"MODERATE": n_ok},
            "cells_exceeding_threshold": 0,
            "max_probability": 0.6,
            "mean_probability": 0.35,
        },
        "disclosures": ["fixture disclosure"],
        "cells": cells,
    }


# --- Field contract -------------------------------------------------------

@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_top_level_keys_are_exactly_the_documented_set(state, pilot_area, aoi):
    doc = pmv.to_map_geojson(_prediction(state, pilot_area, aoi))
    assert set(doc) == set(pmv.TOP_LEVEL_KEYS)
    assert doc["type"] == "FeatureCollection"
    assert doc["state"] == state
    assert doc["pilot_area"] == pilot_area
    assert doc["target_date"] == "2026-08-30"
    assert doc["decision_threshold"] == 0.4315
    assert doc["aoi"] == aoi
    assert doc["summary"]["cells_total"] == 8


@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_each_cell_carries_exactly_five_properties(state, pilot_area, aoi):
    doc = pmv.to_map_geojson(_prediction(state, pilot_area, aoi))
    assert len(doc["features"]) == 8
    for feature in doc["features"]:
        assert feature["type"] == "Feature"
        assert set(feature) == {"type", "id", "geometry", "properties"}
        assert set(feature["properties"]) == set(pmv.CELL_PROPERTY_KEYS)
        assert feature["id"] == feature["properties"]["cell_id"]


@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_the_heavy_per_cell_payload_is_gone_everywhere(state, pilot_area, aoi):
    doc = pmv.to_map_geojson(_prediction(state, pilot_area, aoi))
    # Serialise the CELLS only: the `view` block deliberately NAMES the omitted
    # keys as documentation, which is not the same as carrying their values.
    blob = json.dumps(doc["features"])
    for omitted in ("bbox", "reasons", "twi", "land_cover_class", "curvature",
                    "elevation", "slope", "aspect", "row", "col"):
        assert omitted not in blob, "%s leaked %r into the map view" % (state, omitted)
    for feature in doc["features"]:
        for key in pmv.OMITTED_CELL_KEYS:
            assert key not in feature
            assert key not in feature["properties"]
    # The 14-day series and the 5 rainfall features are not repeated anywhere.
    whole = json.dumps(doc)
    assert "daily_series_mm" not in whole
    assert "rain_14d_mm" not in whole


# --- Size ------------------------------------------------------------------

@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_the_map_payload_is_materially_smaller(state, pilot_area, aoi):
    prediction = _prediction(state, pilot_area, aoi, n_ok=60, n_unavailable=6)
    grid_bytes = len(json.dumps(prediction))
    map_bytes = len(json.dumps(pmv.to_map_geojson(prediction)))
    assert map_bytes < grid_bytes
    # The 11-value feature vector plus the bbox and reasons dominate the grid
    # payload. On this fixture (66 cells, realistic float precision) the map view
    # measures ~42% of the grid payload; the assertion leaves headroom but still
    # fails loudly if a heavy per-cell member ever creeps back in.
    assert map_bytes < 0.5 * grid_bytes, (
        "%s: map %d bytes vs grid %d bytes" % (state, map_bytes, grid_bytes)
    )


def test_per_cell_cost_grows_only_with_the_cell_count():
    """No hidden per-cell copy of an O(features) structure."""
    small = pmv.to_map_geojson(_prediction(*PILOTS[0][:2], PILOTS[0][2],
                                           n_ok=10, n_unavailable=0))
    large = pmv.to_map_geojson(_prediction(*PILOTS[0][:2], PILOTS[0][2],
                                           n_ok=100, n_unavailable=0))
    per_cell_small = len(json.dumps(small["features"])) / 10.0
    per_cell_large = len(json.dumps(large["features"])) / 100.0
    assert per_cell_large < per_cell_small * 1.2


# --- Honesty ---------------------------------------------------------------

@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_unavailable_cells_stay_visible_with_no_probability(state, pilot_area, aoi):
    doc = pmv.to_map_geojson(_prediction(state, pilot_area, aoi))
    unavailable = [f for f in doc["features"]
                   if f["properties"]["status"] == "UNAVAILABLE"]
    assert len(unavailable) == 2, "a dropped cell would read as safe"
    for feature in unavailable:
        props = feature["properties"]
        assert props["probability"] is None
        assert props["risk_class"] is None
        assert props["exceeds_decision_threshold"] is None


@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_probabilities_are_copied_never_recomputed(state, pilot_area, aoi):
    prediction = _prediction(state, pilot_area, aoi)
    doc = pmv.to_map_geojson(prediction)
    by_id = {f["properties"]["cell_id"]: f["properties"] for f in doc["features"]}
    for cell in prediction["cells"]:
        expected = cell["susceptibility_probability"]
        got = by_id[cell["cell_id"]]["probability"]
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected)
        assert by_id[cell["cell_id"]]["risk_class"] == cell["risk_class"]


@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_a_fallback_series_is_still_labelled_fallback(state, pilot_area, aoi):
    prediction = _prediction(state, pilot_area, aoi, rainfall=_FALLBACK_RAINFALL)
    doc = pmv.to_map_geojson(prediction)
    rain = doc["rainfall"]
    assert rain["is_fallback"] is True
    assert rain["data_quality_status"] == "FALLBACK"
    assert rain["source_kind"] == "OPEN_METEO_FALLBACK"
    assert "FALLBACK" in rain["source"]
    assert rain["caveats"] == ["Reanalysis, not a live satellite observation."]


@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_real_rainfall_provenance_survives_intact(state, pilot_area, aoi):
    doc = pmv.to_map_geojson(_prediction(state, pilot_area, aoi))
    rain = doc["rainfall"]
    for key in ("source", "source_kind", "run_type", "is_fallback",
                "data_quality_status", "units", "requested_date",
                "rainfall_observation_date", "fetched_at_utc", "freshness",
                "window_days"):
        assert rain[key] == _REAL_RAINFALL[key], "%s lost rainfall.%s" % (state, key)
    assert rain["coverage"]["window_semantics"] == "T-1..T-14"
    # The heavy members are dropped, not rewritten.
    assert "daily_series_mm" not in rain
    assert "features" not in rain


def test_generated_from_is_copied_and_never_authored_here():
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2])
    prediction["generated_from"] = "whatever the producer said"
    doc = pmv.to_map_geojson(prediction)
    assert doc["generated_from"] == "whatever the producer said"
    source = open(os.path.join(REPO_SERVICES, "pilot_map_view.py")).read()
    # No code path in the transform builds a rainfall or model claim of its own.
    assert "persisted LightGBM" not in source


def test_route_level_provenance_block_passes_through_when_present():
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2])
    prediction["rainfall_provenance"] = {"data_quality_status": "REAL",
                                         "is_fallback": False}
    doc = pmv.to_map_geojson(prediction)
    assert doc["rainfall_provenance"]["data_quality_status"] == "REAL"


def test_a_producer_without_a_rainfall_report_gets_no_invented_block():
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2])
    prediction["rainfall"] = None
    doc = pmv.to_map_geojson(prediction)
    assert doc["rainfall"] is None


# --- Geometry --------------------------------------------------------------

@pytest.mark.parametrize("state,pilot_area,aoi", PILOTS)
def test_coordinates_are_lon_lat_at_the_cell_centre(state, pilot_area, aoi):
    prediction = _prediction(state, pilot_area, aoi)
    doc = pmv.to_map_geojson(prediction)
    for cell, feature in zip(prediction["cells"], doc["features"]):
        lon, lat = feature["geometry"]["coordinates"]
        assert feature["geometry"]["type"] == "Point"
        assert lon == pytest.approx(cell["longitude"])
        assert lat == pytest.approx(cell["latitude"])
        # Longitude first: a lat/lon swap would put NER India in the ocean.
        assert lon > lat


# --- Refusals --------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, "prediction", 3, 0.0])
def test_a_non_prediction_is_refused(bad):
    with pytest.raises(pmv.MapViewError):
        pmv.to_map_geojson(bad)


def test_a_bare_cell_list_is_refused_not_treated_as_a_prediction():
    with pytest.raises(pmv.MapViewError):
        pmv.to_map_geojson([])
    with pytest.raises(pmv.MapViewError):
        pmv.to_map_geojson([{"cell_id": "r00c00", "latitude": 27.0,
                             "longitude": 88.0}])


def test_a_prediction_without_cells_is_refused():
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2])
    del prediction["cells"]
    with pytest.raises(pmv.MapViewError):
        pmv.to_map_geojson(prediction)


def test_non_list_cells_are_refused():
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2])
    prediction["cells"] = {"r00c00": {}}
    with pytest.raises(pmv.MapViewError):
        pmv.to_map_geojson(prediction)


@pytest.mark.parametrize("missing", ["latitude", "longitude"])
def test_a_cell_without_a_coordinate_is_refused_not_placed_at_zero(missing):
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2])
    prediction["cells"][2][missing] = None
    with pytest.raises(pmv.MapViewError) as excinfo:
        pmv.to_map_geojson(prediction)
    assert missing in str(excinfo.value)


def test_an_empty_grid_projects_to_an_empty_feature_collection():
    prediction = _prediction(*PILOTS[0][:2], PILOTS[0][2],
                             n_ok=0, n_unavailable=0)
    doc = pmv.to_map_geojson(prediction)
    assert doc["type"] == "FeatureCollection"
    assert doc["features"] == []


# --- Purity (source introspection) ----------------------------------------

def test_the_transform_touches_no_model_no_raster_and_no_network():
    source = open(os.path.join(REPO_SERVICES, "pilot_map_view.py")).read()
    for forbidden in ("rasterio", ".tif", "predict_proba", "rainfall_service",
                      "requests", "httpx", "open(", "np.", "import "):
        assert forbidden not in source, "pilot_map_view must not use %r" % forbidden


def test_the_four_services_no_longer_hard_code_a_real_imerg_claim():
    """
    `generated_from` must derive its rainfall clause from the series actually
    used, so a FALLBACK run cannot advertise a live IMERG observation.
    """
    for name in ("sikkim_prediction", "assam_prediction",
                 "arunachal_prediction", "meghalaya_prediction"):
        source = open(os.path.join(REPO_SERVICES, name + ".py")).read()
        assert '+ real IMERG antecedent rainfall"' not in source, name
        assert "rainfall_source_label" in source, name
