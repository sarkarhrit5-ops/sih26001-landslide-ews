"""
Lightweight map payload for the four pilot /predict/<state>/grid predictions.

WHY THIS EXISTS
    The full grid response carries, per cell, the complete 11-feature vector plus
    the cell bbox and reasons. That is the right payload for auditing a
    prediction, and its contract is FROZEN -- nothing here changes it. It is the
    wrong payload for drawing a map: the feature vectors dominate the bytes and
    the map only needs a coordinate, a probability and a class.

WHAT THIS IS
    A PURE TRANSFORM of an already-computed prediction dict. This module runs no
    model, fetches no rainfall, opens no raster and reads no file; it cannot
    change a probability because it only ever copies one. The caller runs the
    prediction ONCE and hands the result here, so the map endpoints cost exactly
    what the grid endpoints cost.

SHAPE
    A GeoJSON FeatureCollection, which is what Leaflet consumes natively
    (`L.geoJSON(fc, {pointToLayer})`). One Point feature per cell at the cell
    CENTER -- the same center the model was sampled at. GeoJSON permits foreign
    members, so the state/date/rainfall-provenance/summary metadata rides at the
    top level of the same document instead of needing a wrapper the frontend
    would have to unpick.

HONESTY RULES (the same ones the full response obeys)
      * A cell the backend did not score is still PRESENT, with
        probability=None, risk_class=None, exceeds_decision_threshold=None and
        status="UNAVAILABLE". A missing prediction must stay visible on the map
        rather than being dropped and read as "safe".
      * The rainfall block is copied from the prediction verbatim, including
        data_quality_status / is_fallback / source_kind, so a FALLBACK series can
        never be presented here as an official live IMERG observation.
      * `generated_from` is copied from the prediction, which now derives its
        rainfall clause from the series that was actually used.
"""

CELL_PROPERTY_KEYS = (
    "cell_id",
    "status",
    "probability",
    "risk_class",
    "exceeds_decision_threshold",
)

TOP_LEVEL_KEYS = (
    "type",
    "state",
    "pilot_area",
    "target_date",
    "generated_from",
    "decision_threshold",
    "aoi",
    "grid",
    "rainfall",
    "summary",
    "view",
    "features",
)

# Copied through so the map can label its own provenance without a second
# request. Every one of these is O(1) in the cell count.
RAINFALL_VIEW_KEYS = (
    "source",
    "source_kind",
    "run_type",
    "is_fallback",
    "data_quality_status",
    "units",
    "requested_date",
    "rainfall_observation_date",
    "fetched_at_utc",
    "freshness",
    "window_days",
    "aoi_uniform",
    "note",
    "caveats",
)

# Deliberately NOT copied per cell: `features` (the 11-value vector), `bbox`,
# `reasons`, `row`, `col`. They stay available on /predict/<state>/grid.
OMITTED_CELL_KEYS = ("features", "bbox", "reasons", "row", "col")

VIEW_NOTE = (
    "Lightweight map view: a pure projection of the /predict/<state>/grid result "
    "for this same request. Per-cell terrain/rainfall feature vectors, cell "
    "bboxes and per-cell reasons are omitted here and remain available on "
    "/predict/<state>/grid. No probability, threshold, model or rainfall value "
    "is recomputed; cells are copied, never filtered, so UNAVAILABLE cells stay "
    "visible with a null probability."
)


class MapViewError(ValueError):
    """The supplied object is not a pilot grid prediction."""


def _require_mapping(prediction):
    if not isinstance(prediction, dict):
        raise MapViewError(
            "expected a prediction dict from predict_<state>_grid, got %s"
            % type(prediction).__name__
        )
    if "cells" not in prediction:
        raise MapViewError("prediction has no 'cells'; refusing to invent a map")
    if not isinstance(prediction["cells"], list):
        raise MapViewError(
            "prediction 'cells' is %s, not a list" % type(prediction["cells"]).__name__
        )


def _rainfall_view(prediction):
    """
    The O(1) provenance subset. `daily_series_mm` (14 numbers) and the 5-value
    `features` dict are dropped: the map does not plot them and the grid response
    still carries them.
    """
    rainfall = prediction.get("rainfall")
    if not isinstance(rainfall, dict):
        return None
    view = {}
    for key in RAINFALL_VIEW_KEYS:
        if key in rainfall:
            view[key] = rainfall[key]
    coverage = rainfall.get("coverage")
    if isinstance(coverage, dict):
        # Keep the window semantics string; drop the repeated AOI bbox (already
        # at the top level as `aoi`).
        view["coverage"] = {
            k: coverage[k]
            for k in ("state", "aoi_uniform", "window_days", "window_semantics")
            if k in coverage
        }
    return view or None


def _feature_for_cell(cell, index):
    if not isinstance(cell, dict):
        raise MapViewError("cell %d is %s, not a dict" % (index, type(cell).__name__))
    for key in ("latitude", "longitude"):
        if cell.get(key) is None:
            raise MapViewError(
                "cell %d has no %s; refusing to place it on a map" % (index, key)
            )
    probability = cell.get("susceptibility_probability")
    properties = {
        "cell_id": cell.get("cell_id"),
        "status": cell.get("status"),
        "probability": None if probability is None else float(probability),
        "risk_class": cell.get("risk_class"),
        "exceeds_decision_threshold": cell.get("exceeds_decision_threshold"),
    }
    return {
        "type": "Feature",
        "id": cell.get("cell_id"),
        "geometry": {
            "type": "Point",
            # GeoJSON is [lon, lat] -- the opposite order to Leaflet's LatLng.
            "coordinates": [float(cell["longitude"]), float(cell["latitude"])],
        },
        "properties": properties,
    }


def to_map_geojson(prediction):
    """
    Project one already-computed pilot grid prediction into a Leaflet-ready
    GeoJSON FeatureCollection with top-level provenance and summary metadata.

    Raises MapViewError (-> HTTP 500 would be wrong; the callers only pass their
    own freshly computed prediction, so this is an internal invariant) if handed
    something that is not a pilot grid prediction.
    """
    _require_mapping(prediction)

    features = [
        _feature_for_cell(cell, idx) for idx, cell in enumerate(prediction["cells"])
    ]

    document = {
        "type": "FeatureCollection",
        "state": prediction.get("state"),
        "pilot_area": prediction.get("pilot_area"),
        "target_date": prediction.get("target_date"),
        "generated_from": prediction.get("generated_from"),
        "decision_threshold": prediction.get("decision_threshold"),
        "aoi": prediction.get("aoi"),
        "grid": prediction.get("grid"),
        "rainfall": _rainfall_view(prediction),
        "summary": prediction.get("summary"),
        "view": {
            "kind": "pilot_grid_map_view",
            "geometry": "Point (cell center)",
            "cell_property_keys": list(CELL_PROPERTY_KEYS),
            "omitted_cell_keys": list(OMITTED_CELL_KEYS),
            "full_response_endpoint_note": (
                "The complete per-cell feature vectors, cell bboxes and reasons "
                "are unchanged on /predict/<state>/grid."
            ),
            "note": VIEW_NOTE,
        },
        "features": features,
    }
    provenance = prediction.get("rainfall_provenance")
    if provenance is not None:
        # The route-level normalised block, when the caller added it.
        document["rainfall_provenance"] = provenance
    return document
