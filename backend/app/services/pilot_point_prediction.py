"""
PILOT POINT PREDICTION SERVICE (used by /risk/current)

Answers a SINGLE point inside one of the four canonical pilot AOIs by running the
same persisted, rainfall-coupled pilot model, over the same real rasters, with the
same live rainfall series that /predict/<state>/grid already uses -- evaluated at
one location instead of a grid of cell centers.

WHY THIS EXISTS.
`/risk/current` computes Option-C fusion (ml_pipeline.dynamic_risk_module), which
needs a RAINFALL-INDEPENDENT susceptibility_score because it applies rainfall
separately as a trigger multiplier. Every persisted pilot model is the 11-feature
`static_plus_rainfall` artifact, so risk_inputs.resolve_model_input DELIBERATELY
refuses it (using its output as susceptibility_score and then multiplying by the
rainfall trigger would double-count rainfall), and no static-only model exists.
Rather than fabricate a susceptibility or bypass that gate, this module serves the
number the system can honestly produce: the pilot model's RAW rainfall-conditioned
probability, explicitly labelled as such, with Option-C fusion reported as NOT
APPLIED and the gate's reason stated in the response.

WHAT IT REUSES (nothing here is new science, and nothing is retrained).
  * app.core.config_states.PILOT_AOIS       -- canonical AOI rectangles
  * sikkim_prediction._resolve_model_evidence / _default_rainfall_provider /
    _validate_rainfall_features / _warning_level / _round_features /
    _model_report / _rainfall_report / _score_cells / _default_terrain_sampler /
    _disclosures, and DECISION_THRESHOLD + MODEL_FEATURE_ORDER
  * assam/arunachal/meghalaya _<state>_terrain_sampler, _<state>_land_cover,
    _score_cells_categorical and _<state>_disclosures (real ESA WorldCover land
    cover, scored as the categorical the model was fit on)
  * risk_inputs.land_cover_class_from_elevation for Sikkim's documented
    elevation-binned proxy
The four predict_<state>_grid functions, risk_inputs, ml_pipeline, thresholds,
weather_ingestion and rainfall_service are NOT modified or called differently.

COST. The samplers and scorers already take a list of points, so this path passes a
one-element list: five or six single-pixel raster samples plus one rainfall lookup
that is served from rainfall_service's 30-minute AOI cache (the AOI bounds used here
are byte-identical to the grid path's, so the two share one cache entry rather than
issuing a second fetch).

DATA-INTEGRITY CONTRACT (as everywhere else in this backend): answer only from real
resolved inputs, or refuse. Nothing is defaulted, imputed, back-filled or
hard-coded. Missing terrain, nodata land cover, an unusable model artifact or an
unobtainable rainfall window all raise PredictionUnavailable (-> HTTP 503) with the
reason; there is no placeholder probability anywhere in this file.
"""

from app.core.config_states import PILOT_AOIS, get_pilot_aoi_bounds
from app.services import risk_inputs
from app.services import arunachal_prediction as arp
from app.services import assam_prediction as ap
from app.services import meghalaya_prediction as mp
from app.services import sikkim_prediction as sp


# Shared, single-sourced contract (identical objects, so the paths cannot drift).
PredictionUnavailable = sp.PredictionUnavailable
MODEL_FEATURE_ORDER = sp.MODEL_FEATURE_ORDER
TERRAIN_FEATURES = sp.TERRAIN_FEATURES
LAND_COVER_FEATURE = sp.LAND_COVER_FEATURE
RAINFALL_FEATURES = sp.RAINFALL_FEATURES
RAINFALL_WINDOW_DAYS = sp.RAINFALL_WINDOW_DAYS
DECISION_THRESHOLD = sp.DECISION_THRESHOLD

# The label that names what this endpoint returns, so no client can mistake it for
# the Option-C fused score.
METHOD = "pilot_rainfall_coupled_model_point"

# Why Option-C fusion is not applied.
OPTION_C_UNAVAILABLE_REASON = (
    "Option-C fusion (ml_pipeline.dynamic_risk_module) was NOT applied. It requires "
    "a rainfall-independent susceptibility_score, and the persisted pilot model is "
    "rainfall-coupled (its features include %s): using this model's output as "
    "susceptibility_score and then multiplying it by the rainfall trigger "
    "multiplier would double-count rainfall. No rainfall-independent (static-only) "
    "model is persisted for the pilot states, and none is invented here. The "
    "probability reported under 'hazard' is the model's own rainfall-conditioned "
    "output, not a fused final_risk_score."
    % ", ".join(sorted(RAINFALL_FEATURES))
)

# Accepted spellings of the four pilot states for the additive `?state=` parameter.
STATE_ALIASES = {
    "sikkim": "Sikkim",
    "assam": "Assam",
    "arunachal": "Arunachal Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "arunachal_pradesh": "Arunachal Pradesh",
    "meghalaya": "Meghalaya",
}


class PilotStateError(Exception):
    """
    The pilot state for a point could not be established from the request. Carries
    a machine-readable `reason` and `details`; the API layer renders it as HTTP 400
    because the client can fix it (by naming a state). It never carries a
    prediction.
    """

    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class PilotStateAmbiguous(PilotStateError):
    """The point lies inside more than one canonical pilot AOI; `?state=` required."""


class PilotStateInvalid(PilotStateError):
    """`?state=` named a non-pilot state, or a state whose AOI excludes the point."""


class PointOutsidePilotAoi(PilotStateError):
    """
    The point lies in no canonical pilot AOI. Not an error for the caller to fix by
    naming a state: the API layer falls back to the pre-existing Option-C path,
    whose behaviour is unchanged.
    """


def pilot_states_containing(lat, lon):
    """Canonical pilot state names whose AOI rectangle contains the point."""
    return [
        name for name in sorted(PILOT_AOIS)
        if risk_inputs.point_within_pilot_aoi(lat, lon, name)
    ]


def resolve_pilot_state(lat, lon, state=None):
    """
    Establish EXPLICITLY which pilot state answers for this point.

    There is no silent default: an unqualified point is resolved only when exactly
    one canonical AOI contains it. Two pilot AOIs overlap (Assam/Meghalaya and
    Assam/Arunachal Pradesh share bands), and in those bands the caller must name
    the state.

    Raises PilotStateInvalid (unknown state, or a named state whose AOI excludes the
    point), PilotStateAmbiguous (>1 candidate, no state named) or
    PointOutsidePilotAoi (no candidate).
    """
    candidates = pilot_states_containing(lat, lon)
    if state is not None and str(state).strip() != "":
        key = str(state).strip().lower().replace("-", " ")
        resolved = STATE_ALIASES.get(key)
        if resolved is None:
            raise PilotStateInvalid(
                "Unknown pilot state %r. The pilot states with a canonical AOI and a "
                "persisted model are: %s." % (state, ", ".join(sorted(PILOT_AOIS))),
                details={"requested_state": state, "pilot_states": sorted(PILOT_AOIS)},
            )
        if not risk_inputs.point_within_pilot_aoi(lat, lon, resolved):
            bounds = get_pilot_aoi_bounds(resolved)
            raise PilotStateInvalid(
                "Point (lat=%s, lon=%s) is outside the canonical %s pilot AOI "
                "(min_lat=%s, max_lat=%s, min_lon=%s, max_lon=%s), so the %s rasters "
                "and model do not cover it."
                % (lat, lon, resolved, bounds["min_lat"], bounds["max_lat"],
                   bounds["min_lon"], bounds["max_lon"], resolved),
                details={"requested_state": resolved, "pilot_aoi": bounds,
                         "pilot_states_containing_point": candidates},
            )
        return resolved
    if not candidates:
        raise PointOutsidePilotAoi(
            "Point (lat=%s, lon=%s) is inside no canonical pilot AOI, so no pilot "
            "model or raster set covers it." % (lat, lon),
            details={"pilot_aois": {n: get_pilot_aoi_bounds(n) for n in sorted(PILOT_AOIS)}},
        )
    if len(candidates) > 1:
        raise PilotStateAmbiguous(
            "Point (lat=%s, lon=%s) lies inside %d overlapping canonical pilot AOIs "
            "(%s), so the pilot state cannot be resolved from the coordinates alone. "
            "Re-issue the request with ?state=<one of: %s>. No state was assumed."
            % (lat, lon, len(candidates), ", ".join(candidates), ", ".join(candidates)),
            details={"pilot_states_containing_point": candidates,
                     "pilot_aois": {n: get_pilot_aoi_bounds(n) for n in candidates}},
        )
    return candidates[0]


# ---------------------------------------------------------------------------
# Per-state dispatch: every entry points at the EXISTING collaborators of that
# state's grid service. No behaviour is re-implemented here; the only thing this
# table does is name, per state, which real sampler / land-cover resolver / scorer
# / disclosure set the grid path already uses.
# ---------------------------------------------------------------------------
def _sikkim_proxy_land_cover(elevation_m):
    """
    Sikkim's documented ELEVATION-BINNED land-cover proxy, returned in the same
    {"value": int|None, "problems": [str]} shape the WorldCover resolvers use, so
    the assembly below has one code path. Never substitutes a class: an elevation
    the proxy cannot bin comes back value=None with the reason.
    """
    try:
        return {"value": int(risk_inputs.land_cover_class_from_elevation(elevation_m)),
                "problems": []}
    except (ValueError, KeyError) as exc:
        return {"value": None, "problems": [str(exc)]}



PILOT_SPECS = {
    "Sikkim": {
        "module": sp,
        "pilot_area": "East Sikkim",
        "terrain_sampler": lambda centers, data_dir: sp._default_terrain_sampler(centers, data_dir),
        # Sikkim has no observed land-cover raster: the model was fit on the
        # documented elevation-binned proxy, so inference must use the same proxy.
        "land_cover_resolver": None,
        "land_cover_status": risk_inputs.STATUS_DERIVED_PROXY,
        "land_cover_source": (
            "documented elevation-binned proxy "
            "(risk_inputs.land_cover_class_from_elevation)"
        ),
        "scorer": lambda model, rows: sp._score_cells(model, rows),
        "disclosures": sp._disclosures,
        "generated_from": (
            "persisted Sikkim LightGBM (static_plus_rainfall, 11 features) + real "
            "terrain rasters + elevation-proxy land cover + live rainfall_service "
            "antecedent rainfall, evaluated at one point"
        ),
    },
    "Assam": {
        "module": ap,
        "pilot_area": ap.PILOT_AREA,
        "terrain_sampler": lambda centers, data_dir: ap._assam_terrain_sampler(centers, data_dir),
        "land_cover_resolver": lambda centers, data_dir: ap._assam_land_cover(centers, data_dir),
        "land_cover_status": risk_inputs.STATUS_REAL,
        "land_cover_source": "ESA WorldCover v200 (2021), nominal groups 1..6",
        "scorer": lambda model, rows: ap._score_cells_categorical(model, rows),
        "disclosures": ap._assam_disclosures,
        "generated_from": (
            "persisted Assam LightGBM (static_plus_rainfall, 11 features; land cover "
            "scored as a categorical) + real terrain rasters + real ESA WorldCover "
            "land cover + live rainfall_service antecedent rainfall, evaluated at one "
            "point"
        ),
    },
    "Arunachal Pradesh": {
        "module": arp,
        "pilot_area": arp.PILOT_AREA,
        "terrain_sampler": lambda centers, data_dir: arp._arunachal_terrain_sampler(centers, data_dir),
        "land_cover_resolver": lambda centers, data_dir: arp._arunachal_land_cover(centers, data_dir),
        "land_cover_status": risk_inputs.STATUS_REAL,
        "land_cover_source": "ESA WorldCover v200 (2021), nominal groups 1..6",
        "scorer": lambda model, rows: arp._score_cells_categorical(model, rows),
        "disclosures": arp._arunachal_disclosures,
        "generated_from": (
            "persisted Arunachal Pradesh LightGBM (static_plus_rainfall, 11 features; "
            "land cover scored as a categorical) + real terrain rasters + real ESA "
            "WorldCover land cover + live rainfall_service antecedent rainfall, "
            "evaluated at one point"
        ),
    },
    "Meghalaya": {
        "module": mp,
        "pilot_area": mp.PILOT_AREA,
        "terrain_sampler": lambda centers, data_dir: mp._meghalaya_terrain_sampler(centers, data_dir),
        "land_cover_resolver": lambda centers, data_dir: mp._meghalaya_land_cover(centers, data_dir),
        "land_cover_status": risk_inputs.STATUS_REAL,
        "land_cover_source": "ESA WorldCover v200 (2021), nominal groups 1..6",
        "scorer": lambda model, rows: mp._score_cells_categorical(model, rows),
        "disclosures": mp._meghalaya_disclosures,
        "generated_from": (
            "persisted Meghalaya LightGBM (static_plus_rainfall, 11 features; land "
            "cover scored as a categorical) + real terrain rasters + real ESA "
            "WorldCover land cover + live rainfall_service antecedent rainfall, "
            "evaluated at one point"
        ),
    },
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _point_disclosures(state, spec):
    """
    The state's own grid disclosures, plus the three that are specific to serving a
    POINT through /risk/current. The grid disclosures are copied, never edited.
    """
    extra = [
        "METHOD=%s. The number under hazard.rainfall_conditioned_probability is the "
        "persisted pilot model's RAW positive-class probability WITH live antecedent "
        "rainfall among its 11 features. It is NOT a rainfall-independent "
        "susceptibility_score and NOT the Option-C fused final_risk_score." % METHOD,
        OPTION_C_UNAVAILABLE_REASON,
        "Rainfall is the AOI-mean antecedent series (T-1..T-%d, event day excluded) "
        "for the whole %s pilot AOI -- the same series and the same rainfall_service "
        "cache entry the /predict/<state>/grid endpoint uses -- not a rainfall value "
        "measured at this exact point." % (RAINFALL_WINDOW_DAYS, state),
    ]
    return list(spec["disclosures"]()) + extra


def predict_pilot_point(lat, lon, target_date, state=None, run_type="Early",
                        data_dir=None, artifact_dir=None, model_evidence=None,
                        terrain_sampler=None, land_cover_resolver=None,
                        rainfall_provider=None):
    """
    Run the persisted pilot model for ONE point and return its rainfall-conditioned
    hazard probability, labelled as such.

    `state` may be None (resolved from the coordinates when unambiguous). The
    collaborators are injectable purely so the assembly and the no-fabrication
    invariants are testable offline with fakes; the defaults run the real host-only
    pipeline (rasterio raster reads, joblib model load, live rainfall_service).

    Returns a JSON-safe dict. Raises PredictionUnavailable (-> HTTP 503) when the
    model, terrain, land cover or rainfall cannot be obtained for this point, and
    PilotStateError (-> HTTP 400) when the pilot state cannot be established.
    """
    state_name = resolve_pilot_state(lat, lon, state=state)
    spec = PILOT_SPECS[state_name]
    target_date = sp._as_datetime(target_date)
    point = (float(lat), float(lon))
    centers = [point]

    # 1. Model (real persisted artifacts, exact feature order) -- refuse if unusable.
    evidence = sp._resolve_model_evidence(model_evidence, state_name, artifact_dir)
    model = evidence["model"]

    # 2. Rainfall: the canonical AOI bounds, so this shares the grid path's
    #    rainfall_service cache entry instead of triggering a second fetch.
    bounds = get_pilot_aoi_bounds(state_name)
    provider = rainfall_provider or sp._default_rainfall_provider
    try:
        rainfall = provider(bounds, target_date, run_type)
    except PredictionUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced honestly as 503, never swallowed
        raise PredictionUnavailable(
            "Real antecedent rainfall could not be obtained for the %s pilot AOI "
            "(%s: %s)." % (state_name, type(exc).__name__, exc)
        )
    if not isinstance(rainfall, dict) or "features" not in rainfall:
        raise PredictionUnavailable("Rainfall provider returned no 'features'.")
    rain_features = rainfall["features"]
    sp._validate_rainfall_features(rain_features)

    # 3. Terrain at the point (the state's own real rasters).
    sampler = terrain_sampler or spec["terrain_sampler"]
    samples = sampler(centers, data_dir)
    if len(samples) != 1:
        raise PredictionUnavailable(
            "terrain sampler returned %d rows for 1 point." % len(samples)
        )
    terrain_values = (samples[0] or {}).get("values")
    if not terrain_values:
        raise PredictionUnavailable(
            "Real terrain is unavailable at (lat=%s, lon=%s) in the %s pilot AOI; no "
            "placeholder was substituted." % (lat, lon, state_name),
            details={"problems": list((samples[0] or {}).get("problems")
                                      or ["terrain unavailable at this point"])},
        )

    # 4. Land cover at the point: real WorldCover, or Sikkim's documented proxy.
    resolver = land_cover_resolver or spec["land_cover_resolver"]
    if resolver is None:
        land_cover = _sikkim_proxy_land_cover(terrain_values["elevation"])
    else:
        rows = resolver(centers, data_dir)
        if len(rows) != 1:
            raise PredictionUnavailable(
                "land-cover resolver returned %d rows for 1 point." % len(rows)
            )
        land_cover = rows[0] or {}
    lc_value = land_cover.get("value")
    if lc_value is None:
        raise PredictionUnavailable(
            "Land cover is unavailable at (lat=%s, lon=%s) in the %s pilot AOI; no "
            "class was substituted." % (lat, lon, state_name),
            details={"problems": list(land_cover.get("problems")
                                      or ["land cover unavailable at this point"])},
        )

    # 5. Assemble the single 11-feature row and score it with the state's own scorer.
    row = {feat: float(terrain_values[feat]) for feat in TERRAIN_FEATURES}
    row[LAND_COVER_FEATURE] = int(lc_value)
    for feat in RAINFALL_FEATURES:
        row[feat] = float(rain_features[feat])
    scorer = spec["scorer"]
    probabilities = scorer(model, [row])
    if not probabilities:
        raise PredictionUnavailable(
            "The %s model returned no probability for this point." % state_name
        )
    probability = float(probabilities[0])
    if not (0.0 <= probability <= 1.0):
        raise PredictionUnavailable(
            "The %s model returned %r, which is not a probability in [0, 1]; "
            "refusing to clamp it." % (state_name, probability),
            details={"raw_output": probability},
        )

    rainfall_report = sp._rainfall_report(rainfall)
    return {
        "state": state_name,
        "pilot_area": spec["pilot_area"],
        "method": METHOD,
        "generated_from": spec["generated_from"],
        "point": {"latitude": point[0], "longitude": point[1]},
        "state_resolution": {
            "resolved_state": state_name,
            "requested_state": state,
            "pilot_states_containing_point": pilot_states_containing(lat, lon),
        },
        "target_date": target_date.strftime("%Y-%m-%d"),
        "aoi": bounds,
        "decision_threshold": DECISION_THRESHOLD,
        "model": sp._model_report(evidence),
        "rainfall": rainfall_report,
        "hazard": {
            "status": "OK",
            "rainfall_conditioned_probability": round(probability, 6),
            "risk_class": sp._warning_level(probability),
            "exceeds_decision_threshold": bool(probability >= DECISION_THRESHOLD),
            "features": sp._round_features(row),
            "is_option_c_fused_risk": False,
            "is_rainfall_independent_susceptibility": False,
        },
        # Stated explicitly, with the gate's reason, so the absence of the fused
        # score is never mistaken for an omission -- and so no client can read the
        # probability above as susceptibility_score or final_risk_score.
        "option_c_fusion": {
            "available": False,
            "applied": False,
            "reason": OPTION_C_UNAVAILABLE_REASON,
            "susceptibility_score": None,
            "trigger_multiplier": None,
            "final_risk_score": None,
        },
        "resolved_inputs": {
            "model": {
                "status": evidence.get("status"),
                "source": "persisted %s artifacts (11-feature static_plus_rainfall)"
                          % state_name,
            },
            "terrain": {
                "status": risk_inputs.STATUS_REAL,
                "source": "real %s terrain rasters sampled at the point" % state_name,
            },
            "land_cover": {
                "status": spec["land_cover_status"],
                "source": spec["land_cover_source"],
            },
            "rainfall": {
                # Passed through from the producer; this layer holds no rainfall
                # status of its own and computes nothing.
                "status": rainfall_report.get("data_quality_status"),
                "source": rainfall_report.get("source"),
                "is_fallback": bool(rainfall_report.get("is_fallback")),
            },
        },
        "disclosures": _point_disclosures(state_name, spec),
    }
