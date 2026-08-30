#!/usr/bin/env python3
"""
LIVE PREDICTION VERIFICATION -- host-only, read-only.

Exercises the REAL local prediction and risk path for all four pilots (Sikkim,
Assam, Arunachal Pradesh, Meghalaya) against the ACTUAL on-disk artifacts:

  * backend/data/models/<state>_model.pkl + _feature_schema.json + _metrics.json
  * backend/data/processed/<state>_pilot_{slope,aspect,roughness,tpi}.tif
  * backend/data/raw/<state>_pilot_landcover.tif (Assam / Arunachal / Meghalaya)
  * backend/data/raw/<state>_osm.geojson (exposure)
  * app.services.rainfall_service for rainfall (real IMERG, or the labelled
    Open-Meteo ERA5 fallback when IMERG is unavailable)

Requires the full runtime stack (rasterio, lightgbm, sklearn, xarray, requests)
and outbound network access, so it CANNOT run in the offline sandbox.

Nothing is written, committed or deployed. No application module is modified on
disk; the only in-process substitutions are (a) call-counting proxies wrapped
around the real rainfall fetchers and (b) explicitly labelled injected rainfall
providers used to prove causality. Both are per-process only.

Usage (from backend/):
    python scripts/verify_live_prediction.py
    python scripts/verify_live_prediction.py --states Sikkim Meghalaya
    python scripts/verify_live_prediction.py --skip-live      # no network calls
"""

import argparse
import copy
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.api import routes                                    # noqa: E402
from app.services import arunachal_prediction as arp          # noqa: E402
from app.services import assam_prediction as ap               # noqa: E402
from app.services import meghalaya_prediction as mp           # noqa: E402
from app.services import rainfall_service as rs               # noqa: E402
from app.services import risk_inputs as ri                    # noqa: E402
from app.services import sikkim_prediction as sp              # noqa: E402
from app.services import weather_ingestion                    # noqa: E402

PILOTS = {
    "Sikkim": (sp, "predict_sikkim_grid", (27.33, 88.62)),
    "Assam": (ap, "predict_assam_grid", (26.14, 91.77)),
    "Arunachal Pradesh": (arp, "predict_arunachal_grid", (27.10, 93.60)),
    "Meghalaya": (mp, "predict_meghalaya_grid", (25.57, 91.88)),
}

INJECTED_SOURCE = "INJECTED_CONTROLLED_VALUE(verification only -- not a real observation)"

RESULTS = []
CALLS = {"imerg": 0, "fallback": 0, "legacy": 0}
LIVE_SUMMARY = {}
INJECTED_SUMMARY = {}
RISK_SUMMARY = {}
DEFAULT_DATE = None          # set in main() from --date or today's UTC date
FORCE_FALLBACK = False       # set in main() from --force-fallback
LOW_MM = 1.0
HIGH_MM = 90.0
# The services' own DEFAULT_STEP_DEG is 0.05, which is ~1500 cells over the
# Arunachal AOI and needs a real model inference per cell. The rainfall
# assertions are step-independent, so the verifier runs a coarser grid by
# default (override with --step) and uses STEP/2 -- 4x the cells -- for item 7.
COARSE_STEP = 0.25
# Left EMPTY on the host: the point of this script is to run against the real
# terrain/model/exposure artifacts. An out-of-repo dry-run driver may populate it
# with offline fakes purely to exercise this file's own control flow.
EXTRA_KWARGS = {}
_REAL_IMERG = rs._default_imerg_fetcher
_REAL_FALLBACK = rs._default_fallback_fetcher
_REAL_FUSION = routes.dynamic_risk_module
_REAL_LEGACY = weather_ingestion.fetch_imerg_precipitation
FUSION_SPY = []


def check(item, label, ok, detail):
    RESULTS.append((item, label, bool(ok), detail))
    print("  [%s] item %s / %s -- %s"
          % ("PASS" if ok else "FAIL", item, label, detail))


def note(text):
    print("  ... %s" % text)


# ---------------------------------------------------------------------------
# In-process instrumentation (never persisted)
# ---------------------------------------------------------------------------
def count_real_fetchers(force_fallback=False):
    """Wrap the REAL fetchers in counters. force_fallback simulates an IMERG outage."""
    def imerg(session, day, bounds, run_type):
        CALLS["imerg"] += 1
        if force_fallback:
            raise RuntimeError(
                "IMERG deliberately disabled for this verification run"
            )
        return _REAL_IMERG(session, day, bounds, run_type)

    def fallback(bounds, end_date, days):
        CALLS["fallback"] += 1
        return _REAL_FALLBACK(bounds, end_date, days)

    def legacy(*args, **kwargs):
        CALLS["legacy"] += 1
        raise AssertionError(
            "weather_ingestion.fetch_imerg_precipitation was called: the serving "
            "path is supposed to go through rainfall_service"
        )

    rs._default_imerg_fetcher = imerg
    rs._default_fallback_fetcher = fallback
    weather_ingestion.fetch_imerg_precipitation = legacy
    rs.clear_cache()
    for key in CALLS:
        CALLS[key] = 0


def restore():
    rs._default_imerg_fetcher = _REAL_IMERG
    rs._default_fallback_fetcher = _REAL_FALLBACK
    weather_ingestion.fetch_imerg_precipitation = _REAL_LEGACY
    routes.dynamic_risk_module = _REAL_FUSION


def injected_provider(mm_per_day, svc):
    """A controlled, explicitly labelled rainfall payload -- for causality only."""
    def provider(bounds, target_date, run_type="Early", session=None):
        daily = [float(mm_per_day)] * svc.RAINFALL_WINDOW_DAYS
        return {
            "source": INJECTED_SOURCE,
            "run_type": run_type,
            "aoi_uniform": True,
            "window_days": svc.RAINFALL_WINDOW_DAYS,
            "daily_series_mm": daily,
            "features": sp._derive_rainfall_features(daily),
            "source_kind": "INJECTED",
            "is_fallback": False,
            "data_quality_status": "INJECTED_FOR_VERIFICATION",
            "units": "mm",
        }
    return provider


def dead_provider(svc):
    def provider(bounds, target_date, run_type="Early", session=None):
        raise svc.PredictionUnavailable(
            "no rainfall source answered (deliberate outage for verification)",
            details={"simulated": True},
        )
    return provider


def run_grid(state, step=None, rainfall_provider=None, date=None):
    svc, fn_name, _point = PILOTS[state]
    kwargs = {"step_deg": COARSE_STEP if step is None else step}
    kwargs.update(EXTRA_KWARGS.get(state, {}))
    if rainfall_provider is not None:
        kwargs["rainfall_provider"] = rainfall_provider
    target = date or DEFAULT_DATE
    return getattr(svc, fn_name)(target, **kwargs)


def risk_outputs(payload):
    s = payload["summary"]
    return {
        "cells_total": s["cells_total"],
        "cells_scored": s["cells_scored"],
        "mean_probability": s["mean_probability"],
        "max_probability": s["max_probability"],
        "cells_exceeding_threshold": s["cells_exceeding_threshold"],
        "risk_class_counts": dict(s["risk_class_counts"]),
    }


def static_inputs(payload):
    """Everything that must NOT move when only rainfall changes (item 10)."""
    terrain_only = []
    for cell in payload["cells"]:
        feats = cell.get("features") or {}
        terrain_only.append((
            cell.get("lat"), cell.get("lon"), cell.get("status"),
            tuple(sorted((k, v) for k, v in feats.items()
                         if k not in sp.RAINFALL_FEATURES)),
        ))
    return {
        "model": payload["model"],
        "aoi": payload["aoi"],
        "grid": payload["grid"],
        "decision_threshold": payload["decision_threshold"],
        "terrain_and_landcover": terrain_only,
    }


def verify_state(state, live=True):
    svc, fn_name, (lat, lon) = PILOTS[state]
    print("\n=== %s :: /predict/%s/grid ===" % (state, fn_name.split("_")[1]))

    # ---- items 6, 4: one real end-to-end request on the real artifacts --------
    if live:
        count_real_fetchers(force_fallback=FORCE_FALLBACK)
        try:
            payload = run_grid(state)
        except svc.PredictionUnavailable as exc:
            check("6", state, False,
                  "PredictionUnavailable: %s" % getattr(exc, "reason", exc))
            return
        rain = payload["rainfall"]
        out = risk_outputs(payload)
        check("6", "%s endpoint works on real artifacts" % state,
              out["cells_scored"] > 0,
              "%d/%d cells scored, mean_p=%r max_p=%r bands=%s"
              % (out["cells_scored"], out["cells_total"], out["mean_probability"],
                 out["max_probability"], out["risk_class_counts"]))
        prov = (routes._with_rainfall_provenance(copy.deepcopy(payload))
                .get("rainfall_provenance") or {})
        missing = [f for f in routes.RAINFALL_PROVENANCE_FIELDS if f not in prov]
        check("4", "%s provenance survives into the response" % state,
              prov and not missing,
              "source=%r source_kind=%r quality=%r observed=%r cache_hit=%r missing=%s"
              % (rain.get("source"), rain.get("source_kind"),
                 rain.get("data_quality_status"),
                 rain.get("rainfall_observation_date"),
                 (rain.get("freshness") or {}).get("cache_hit"), missing))
        check("2", "%s no hardcoded rainfall" % state,
              CALLS["legacy"] == 0 and (CALLS["imerg"] + CALLS["fallback"]) > 0
              and rain.get("daily_series_mm"),
              "rainfall_service fetches: imerg=%d fallback=%d; legacy fetcher calls=%d; "
              "rain_1d=%r mm" % (CALLS["imerg"], CALLS["fallback"], CALLS["legacy"],
                                 (rain.get("features") or {}).get("rain_1d")))

        # ---- item 8: cache reuse on an identical repeat request ---------------
        before = (CALLS["imerg"], CALLS["fallback"])
        repeat = run_grid(state)
        rep_fresh = repeat["rainfall"].get("freshness") or {}
        check("8", "%s cache reuse" % state,
              (CALLS["imerg"], CALLS["fallback"]) == before
              and rep_fresh.get("cache_hit") is True,
              "new fetches=%d cache_hit=%r age_s=%r ttl_s=%r"
              % (CALLS["imerg"] + CALLS["fallback"] - before[0] - before[1],
                 rep_fresh.get("cache_hit"), rep_fresh.get("age_seconds"),
                 rep_fresh.get("ttl_seconds")))

        # ---- item 7: a 4x finer grid must not add a single fetch --------------
        coarse_cells = out["cells_total"]
        before = CALLS["imerg"] + CALLS["fallback"]
        fine = run_grid(state, step=COARSE_STEP / 2.0)
        check("7", "%s no per-cell downloads" % state,
              CALLS["imerg"] + CALLS["fallback"] == before
              and fine["summary"]["cells_total"] > coarse_cells,
              "cells %d -> %d with 0 additional fetches (total this AOI: %d, "
              "window_days=%d)"
              % (coarse_cells, fine["summary"]["cells_total"], before,
                 svc.RAINFALL_WINDOW_DAYS))
        LIVE_SUMMARY[state] = {
            "rainfall_source": rain.get("source"),
            "source_kind": rain.get("source_kind"),
            "data_quality_status": rain.get("data_quality_status"),
            "is_fallback": rain.get("is_fallback"),
            "rainfall_observation_date": rain.get("rainfall_observation_date"),
            "rain_1d_mm": (rain.get("features") or {}).get("rain_1d"),
            "antecedent_rain_14d_mm": (rain.get("features") or {}).get(
                "antecedent_rain_14d"),
            "risk_outputs": out,
        }
    else:
        note("live request skipped (--skip-live)")

    # ---- items 1, 3, 10: controlled low vs high rainfall ----------------------
    low = run_grid(state, rainfall_provider=injected_provider(LOW_MM, svc))
    high = run_grid(state, rainfall_provider=injected_provider(HIGH_MM, svc))
    lo_out, hi_out = risk_outputs(low), risk_outputs(high)

    lo_feats = (low["rainfall"].get("features") or {})
    hi_feats = (high["rainfall"].get("features") or {})
    reached = []
    for cell in high["cells"]:
        feats = cell.get("features") or {}
        if feats:
            reached = [(k, feats.get(k)) for k in sorted(sp.RAINFALL_FEATURES)]
            break
    check("1", "%s rainfall reaches the per-cell feature vector" % state,
          reached and all(v is not None for _k, v in reached)
          and abs((dict(reached).get("rain_1d") or -1) - HIGH_MM) < 1e-6,
          "injected %.1f mm/day -> cell features %s" % (HIGH_MM, reached))

    moved = (lo_out["mean_probability"] != hi_out["mean_probability"]
             or lo_out["max_probability"] != hi_out["max_probability"]
             or lo_out["risk_class_counts"] != hi_out["risk_class_counts"])
    check("3", "%s low vs high rainfall changes the risk output" % state, moved,
          "low(%.1f mm) mean_p=%r max_p=%r bands=%s  ||  high(%.1f mm) mean_p=%r "
          "max_p=%r bands=%s  || exceeding %d -> %d"
          % (LOW_MM, lo_out["mean_probability"], lo_out["max_probability"],
             lo_out["risk_class_counts"], HIGH_MM, hi_out["mean_probability"],
             hi_out["max_probability"], hi_out["risk_class_counts"],
             lo_out["cells_exceeding_threshold"],
             hi_out["cells_exceeding_threshold"]))
    if not moved:
        note("item 3 did not move: check that a REAL model is loaded "
             "(model=%r) -- a feature-insensitive stand-in model would look "
             "exactly like this." % (high.get("model") or {}).get("artifact"))

    lo_static, hi_static = static_inputs(low), static_inputs(high)
    differing = [k for k in lo_static if lo_static[k] != hi_static[k]]
    check("10", "%s terrain/model/exposure inputs unchanged" % state,
          not differing,
          "identical across the two runs: model artifact, aoi, grid, "
          "decision_threshold and all %d cells' non-rainfall features "
          "(differing keys: %s)"
          % (len(lo_static["terrain_and_landcover"]), differing or "none"))

    INJECTED_SUMMARY[state] = {
        "low": {"rain_1d": lo_feats.get("rain_1d"), **lo_out},
        "high": {"rain_1d": hi_feats.get("rain_1d"), **hi_out},
        "static_inputs_identical": not differing,
    }

    # ---- item 9: no rainfall at all must fail honestly -----------------------
    try:
        bad = run_grid(state, rainfall_provider=dead_provider(svc))
    except svc.PredictionUnavailable as exc:
        blob = json.dumps({"reason": getattr(exc, "reason", str(exc)),
                           "details": getattr(exc, "details", {})}, default=str)
        clean = ("probability" not in blob and "risk_class" not in blob
                 and "0.0" not in blob)
        check("9", "%s unavailable rainfall fails honestly" % state, clean,
              "PredictionUnavailable raised (API layer renders HTTP 503 "
              "DATA_UNAVAILABLE); no probability/risk_class/zero-fill in the "
              "error payload: %s" % blob)
    else:
        check("9", "%s unavailable rainfall fails honestly" % state, False,
              "NO exception: returned %s -- synthetic zeros suspected"
              % risk_outputs(bad))


def verify_risk_path(state, live=True):
    """
    Items 1, 2, 5 for /risk/current -- now the PILOT POINT path.

    Every representative point below is inside exactly one canonical pilot AOI, so
    routes.get_current_risk answers from app.services.pilot_point_prediction: the
    persisted 11-feature model run at the point with the live rainfall_service
    series. Option-C fusion is deliberately NOT applied there (the persisted pilot
    models are rainfall-coupled, so feeding their output into dynamic_risk_module as
    susceptibility_score would double-count rainfall), so this asserts:

      item 1 -- the rainfall_service series really reached the model's features,
      item 2 -- no hardcoded/legacy rainfall was used, and
      item 5 -- REAL vs FALLBACK provenance is labelled, and dynamic_risk_module was
                NOT called and reported no fused score.
    """
    _svc, _fn, (lat, lon) = PILOTS[state]
    print("\n=== %s :: /risk/current pilot point (lat=%s lon=%s) ===" % (state, lat, lon))
    if not live:
        note("live risk request skipped (--skip-live)")
        return

    count_real_fetchers(force_fallback=FORCE_FALLBACK)
    FUSION_SPY.clear()

    def spy(**kwargs):
        FUSION_SPY.append(dict(kwargs))
        return _REAL_FUSION(**kwargs)

    routes.dynamic_risk_module = spy
    try:
        try:
            body = routes.get_current_risk(lat=lat, lon=lon)
        except Exception as exc:                      # HTTPException or similar
            detail = getattr(exc, "detail", None)
            check("6", "%s /risk/current" % state, False,
                  "refused: %s" % (json.dumps(detail, default=str) if detail
                                   else repr(exc)))
            return
    finally:
        routes.dynamic_risk_module = _REAL_FUSION

    rain = body.get("rainfall") or {}
    provenance = body.get("rainfall_provenance") or {}
    hazard = body.get("hazard") or {}
    fusion = body.get("option_c_fusion") or {}
    features = hazard.get("features") or {}

    # The AOI-mean antecedent series the model consumed, recomputed from the series
    # the response reports, so this compares the response against itself rather than
    # against an assumption.
    series = rain.get("daily_series_mm") or []
    expected = None
    if len(series) >= sp.RAINFALL_WINDOW_DAYS:
        expected = sp._derive_rainfall_features([float(v) for v in series])

    rain_ok = bool(
        expected
        and all(features.get(k) is not None
                and abs(float(features[k]) - float(v)) < 1e-3
                for k, v in expected.items())
    )
    check("1", "%s rainfall_service series reaches the pilot model features" % state,
          rain_ok,
          "hazard.features rainfall = %s; derived from response "
          "rainfall.daily_series_mm[%d] = %s; rainfall_conditioned_probability=%r "
          "risk_class=%r"
          % ({k: features.get(k) for k in sorted(sp.RAINFALL_FEATURES)},
             len(series), expected,
             hazard.get("rainfall_conditioned_probability"), hazard.get("risk_class")))

    check("2", "%s risk path has no hardcoded rainfall" % state,
          CALLS["legacy"] == 0 and (CALLS["imerg"] + CALLS["fallback"]) > 0,
          "rainfall_service fetches: imerg=%d fallback=%d; "
          "weather_ingestion.fetch_imerg_precipitation calls=%d"
          % (CALLS["imerg"], CALLS["fallback"], CALLS["legacy"]))

    # The coupled probability must never be dressed up as fused Option-C risk.
    check("4/11", "%s pilot point reports no fabricated fused risk" % state,
          not FUSION_SPY
          and fusion.get("applied") is False
          and fusion.get("susceptibility_score") is None
          and fusion.get("final_risk_score") is None
          and hazard.get("is_option_c_fused_risk") is False
          and hazard.get("is_rainfall_independent_susceptibility") is False
          and body.get("risk") is None,
          "dynamic_risk_module calls=%d; option_c_fusion.applied=%r "
          "susceptibility_score=%r final_risk_score=%r; method=%r"
          % (len(FUSION_SPY), fusion.get("applied"),
             fusion.get("susceptibility_score"), fusion.get("final_risk_score"),
             body.get("method")))

    is_fallback = bool(rain.get("is_fallback") or provenance.get("is_fallback"))
    quality = rain.get("data_quality_status") or provenance.get("data_quality_status")
    if is_fallback:
        ok = (quality == rs.QUALITY_FALLBACK
              and "FALLBACK" in (provenance.get("fallback_warning") or "")
              and "FALLBACK" in (rain.get("note") or ""))
        detail = ("FALLBACK in use: quality=%r warning=%r note=%r"
                  % (quality, (provenance.get("fallback_warning") or "")[:70],
                     (rain.get("note") or "")[:90]))
    else:
        ok = quality == rs.QUALITY_REAL and is_fallback is False
        detail = ("live IMERG in use: quality=%r source=%r (fallback labelling not "
                  "exercised on this run -- use --force-fallback)"
                  % (quality, rain.get("source")))
    check("5", "%s rainfall quality is labelled honestly" % state, ok, detail)

    RISK_SUMMARY[state] = {
        "endpoint": "/risk/current?lat=%s&lon=%s" % (lat, lon),
        "method": body.get("method"),
        "resolved_state": (body.get("state_resolution") or {}).get("resolved_state"),
        "rainfall_source": rain.get("source"),
        "data_quality_status": quality,
        "is_fallback": is_fallback,
        "rainfall_features": {k: features.get(k) for k in sorted(sp.RAINFALL_FEATURES)},
        "rainfall_conditioned_probability":
            hazard.get("rainfall_conditioned_probability"),
        "risk_class": hazard.get("risk_class"),
        "option_c_applied": fusion.get("applied"),
        "dynamic_risk_module_calls": len(FUSION_SPY),
    }


def main():
    global DEFAULT_DATE, FORCE_FALLBACK, COARSE_STEP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", choices=sorted(PILOTS),
                        default=sorted(PILOTS))
    parser.add_argument("--date", default=None,
                        help="target date YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--step", type=float, default=COARSE_STEP,
                        help="grid step in degrees (default %(default)s; the "
                             "services' own default is 0.05)")
    parser.add_argument("--skip-live", action="store_true",
                        help="skip every network request (injected checks only)")
    parser.add_argument("--force-fallback", action="store_true",
                        help="disable IMERG in-process so the REAL Open-Meteo "
                             "ERA5 fallback is exercised end to end")
    args = parser.parse_args()

    import datetime
    DEFAULT_DATE = args.date or datetime.datetime.utcnow().strftime("%Y-%m-%d")
    FORCE_FALLBACK = args.force_fallback
    COARSE_STEP = args.step
    live = not args.skip_live

    print("LIVE PREDICTION VERIFICATION")
    print("  target_date         : %s" % DEFAULT_DATE)
    print("  states              : %s" % ", ".join(args.states))
    print("  grid step (deg)     : %s (item-7 comparison at %s)"
          % (COARSE_STEP, COARSE_STEP / 2.0))
    print("  live network        : %s" % ("yes" if live else "no"))
    print("  IMERG               : %s"
          % ("DISABLED in-process (forcing the real Open-Meteo fallback)"
             if FORCE_FALLBACK else "enabled (preferred source)"))
    print("  rainfall cache TTL  : %ss" % rs.cache_ttl_seconds())
    print("  data dir            : %s" % ri.default_data_dir())

    try:
        for state in args.states:
            try:
                verify_state(state, live=live)
            except Exception:
                check("--", "%s grid path crashed" % state, False,
                      traceback.format_exc().strip().splitlines()[-1])
                traceback.print_exc()
            try:
                verify_risk_path(state, live=live)
            except Exception:
                check("--", "%s risk path crashed" % state, False,
                      traceback.format_exc().strip().splitlines()[-1])
                traceback.print_exc()
    finally:
        restore()
        rs.clear_cache()

    print("\n---- OBSERVED LIVE RAINFALL + RISK OUTPUTS ----")
    print(json.dumps({"prediction_grid": LIVE_SUMMARY,
                      "injected_causality": INJECTED_SUMMARY,
                      "risk_current": RISK_SUMMARY}, indent=2, default=str))

    failed = [r for r in RESULTS if not r[2]]
    print("\n---- SUMMARY: %d checks, %d passed, %d failed ----"
          % (len(RESULTS), len(RESULTS) - len(failed), len(failed)))
    for item, label, _ok, detail in failed:
        print("  FAIL item %s / %s -- %s" % (item, label, detail))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())


