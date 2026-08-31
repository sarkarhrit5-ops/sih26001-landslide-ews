/**
 * Node built-in tests for the LIVE rainfall panel logic.
 *
 * There is no browser test runner in this project, so the honesty-critical logic
 * lives in the pure module `components/pilot/liveRainfallView.ts` and is
 * exercised here with `node:test`. Run with:
 *
 *   cd frontend && node --experimental-strip-types \
 *     --import ./src/__tests__/tsResolve.mjs --test src/__tests__/liveRainfall.test.ts
 *
 * The properties asserted are the ones the UI's truthfulness depends on:
 *   * all four pilots resolve to /rainfall/latest with THEIR OWN backend AOI
 *     label, and never to another state's;
 *   * a REAL record is named "NASA IMERG Early" / "NASA IMERG Late" and requires
 *     BOTH a REAL status and an IMERG source_kind;
 *   * a FALLBACK record is named "Open-Meteo", is marked FALLBACK, and its
 *     display strings never contain the substring "IMERG";
 *   * an UNAVAILABLE record reads "Latest rainfall unavailable", carries the
 *     backend's reason, and produces no zeroes anywhere;
 *   * an incomplete accumulation window reports its reason rather than a sum;
 *   * the auto-refresh interval can never drop below the backend cache TTL.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  LIVE_RAINFALL_CACHE_TTL_SECONDS,
  LIVE_RAINFALL_REFRESH_MS,
  formatInterval,
  formatLiveAge,
  formatMm,
  liveRainfallEndpoint,
  liveRainfallRefreshMs,
  liveRainfallSourceDisplayName,
  liveRainfallTone,
  liveRainfallView,
} from '../components/pilot/liveRainfallView.ts';
import { PILOT_STATE_KEYS, getPilotConfig } from '../data/nerStates.ts';
import type { LiveRainfallResponse } from '../services/api.ts';

const BOUNDS = { min_lat: 27.0, max_lat: 28.1, min_lon: 88.0, max_lon: 88.9 };

function freshness(overrides: Record<string, unknown> = {}) {
  return {
    observed_at_utc: '2026-08-31T06:00:00Z',
    fetched_at_utc: '2026-08-31T10:00:00Z',
    age_seconds: 14400,
    age_minutes: 240,
    freshness_label: 'RECENT',
    is_stale: false,
    staleness_threshold_minutes: 600,
    near_real_time_threshold_minutes: 90,
    measured_from: 'fetched_at_utc - observed_at_utc',
    cache_hit: false,
    ...overrides,
  };
}

/** A REAL IMERG record, shaped exactly as live_rainfall._observed_record emits. */
function realRecord(overrides: Partial<LiveRainfallResponse> = {}): LiveRainfallResponse {
  return {
    state: 'Sikkim',
    aoi_bounds: BOUNDS,
    latest_available_rainfall_mm: 1.25,
    interval_minutes: 30,
    units: 'mm',
    observed_at_utc: '2026-08-31T06:00:00Z',
    fetched_at_utc: '2026-08-31T10:00:00Z',
    age_seconds: 14400,
    age_minutes: 240,
    latency_minutes: 240,
    freshness_label: 'RECENT',
    freshness: freshness(),
    is_stale: false,
    staleness_threshold_minutes: 600,
    expected_product_latency_minutes: 240,
    source: 'NASA GPM IMERG Early half-hourly (GPM_3IMERGHHE.07)',
    source_kind: 'IMERG_HHR_EARLY',
    data_quality_status: 'REAL',
    granules_used: 12,
    attempts: [{ source_kind: 'IMERG_HHR_EARLY', outcome: 'ok' }],
    value_semantics: 'latest available observation for the AOI, NOT a nowcast',
    accum_3h_mm: 4.5,
    accum_3h_unavailable_reason: null,
    accum_6h_mm: 9.0,
    accum_6h_unavailable_reason: null,
    cache_hit: false,
    served_from_cache: false,
    ...overrides,
  } as LiveRainfallResponse;
}

function fallbackRecord(overrides: Partial<LiveRainfallResponse> = {}): LiveRainfallResponse {
  return realRecord({
    interval_minutes: 60,
    source: 'Open-Meteo hourly precipitation (FALLBACK)',
    source_kind: 'OPEN_METEO_FALLBACK',
    data_quality_status: 'FALLBACK',
    expected_product_latency_minutes: null,
    granules_used: 0,
    attempts: [
      { source_kind: 'IMERG_HHR_EARLY', outcome: 'unavailable', detail: 'not published' },
      { source_kind: 'IMERG_HHR_LATE', outcome: 'unavailable', detail: 'not published' },
      { source_kind: 'OPEN_METEO_FALLBACK', outcome: 'ok' },
    ],
    ...overrides,
  });
}

/** An UNAVAILABLE record, shaped as live_rainfall._unavailable_record emits. */
function unavailableRecord(
  overrides: Partial<LiveRainfallResponse> = {},
): LiveRainfallResponse {
  return {
    state: 'Meghalaya',
    aoi_bounds: BOUNDS,
    latest_available_rainfall_mm: null,
    interval_minutes: null,
    units: 'mm',
    observed_at_utc: null,
    fetched_at_utc: '2026-08-31T10:00:00Z',
    age_seconds: null,
    age_minutes: null,
    latency_minutes: null,
    freshness_label: null,
    freshness: freshness({
      observed_at_utc: null,
      age_seconds: null,
      age_minutes: null,
      freshness_label: null,
      is_stale: null,
    }),
    is_stale: null,
    staleness_threshold_minutes: 600,
    expected_product_latency_minutes: null,
    source: null,
    source_kind: null,
    data_quality_status: 'UNAVAILABLE',
    granules_used: 0,
    attempts: [
      { source_kind: 'IMERG_HHR_EARLY', outcome: 'unavailable', detail: 'HTTP 401' },
      { source_kind: 'OPEN_METEO_FALLBACK', outcome: 'error', detail: 'timeout' },
    ],
    unavailable_reason:
      'no IMERG granule was retrievable and the Open-Meteo FALLBACK request failed',
    value_semantics: 'no observation could be retrieved',
    accum_3h_mm: null,
    accum_3h_unavailable_reason: 'no observation was retrieved, so no accumulation exists',
    accum_6h_mm: null,
    accum_6h_unavailable_reason: 'no observation was retrieved, so no accumulation exists',
    cache_hit: false,
    served_from_cache: false,
    ...overrides,
  } as LiveRainfallResponse;
}

/* --------------------------------------------------------------------------
 * Endpoint resolution — all four states
 * ----------------------------------------------------------------------- */

test('every pilot resolves to the live rainfall endpoint with its own AOI label', () => {
  const seen = new Set<string>();
  for (const key of PILOT_STATE_KEYS) {
    const endpoint = liveRainfallEndpoint(key);
    assert.ok(
      endpoint.startsWith('/api/v1/rainfall/latest?state='),
      `${key} must use the monitoring endpoint, got ${endpoint}`,
    );
    const label = decodeURIComponent(endpoint.split('state=')[1]);
    assert.equal(label, getPilotConfig(key).name);
    assert.ok(!seen.has(endpoint), `${key} reused another state's endpoint`);
    seen.add(endpoint);
  }
  assert.equal(seen.size, PILOT_STATE_KEYS.length);
});

test('the state parameter is the backend AOI label, not the route token or state_id', () => {
  // Arunachal is the case where all three differ.
  assert.equal(
    liveRainfallEndpoint('arunachal'),
    '/api/v1/rainfall/latest?state=Arunachal%20Pradesh',
  );
  assert.equal(getPilotConfig('arunachal').stateId, 'arunachal_pradesh');
});

test('an unknown state key throws instead of resolving to another state', () => {
  assert.throws(() => liveRainfallEndpoint('nagaland'), /Unknown pilot state key/);
});

test('the live endpoint is never a prediction endpoint', () => {
  for (const key of PILOT_STATE_KEYS) {
    const endpoint = liveRainfallEndpoint(key);
    assert.ok(!endpoint.includes('/predict/'), endpoint);
    assert.ok(!endpoint.includes('/grid'), endpoint);
    assert.ok(!endpoint.includes('/map'), endpoint);
  }
});

/* --------------------------------------------------------------------------
 * REAL
 * ----------------------------------------------------------------------- */

test('a REAL Early record is named "NASA IMERG Early"', () => {
  const view = liveRainfallView(realRecord());
  assert.equal(view.tone, 'real');
  assert.equal(view.label, 'REAL');
  assert.equal(view.sourceDisplayName, 'NASA IMERG Early');
  assert.equal(view.isRealSatellite, true);
  assert.equal(view.isFallback, false);
  assert.equal(view.unavailableReason, null);
  assert.equal(view.latestMm, 1.25);
  assert.equal(view.intervalLabel, '30-minute interval');
});

test('a REAL Late record is named "NASA IMERG Late"', () => {
  const view = liveRainfallView(
    realRecord({
      source_kind: 'IMERG_HHR_LATE',
      source: 'NASA GPM IMERG Late half-hourly (GPM_3IMERGHHL.07)',
    }),
  );
  assert.equal(view.tone, 'real');
  assert.equal(view.sourceDisplayName, 'NASA IMERG Late');
});

test('REAL requires both a REAL status and an IMERG source_kind', () => {
  // Status says REAL but the source is the fallback → must not read as real.
  assert.equal(
    liveRainfallTone(realRecord({ source_kind: 'OPEN_METEO_FALLBACK' })),
    'fallback',
  );
  // IMERG source but the status is not REAL → not promoted to real.
  assert.equal(
    liveRainfallTone(realRecord({ data_quality_status: 'SOMETHING_ELSE' })),
    'unreported',
  );
});

test('a REAL record surfaces both accumulation windows', () => {
  const view = liveRainfallView(realRecord());
  assert.deepEqual(
    view.accumulations.map((a) => [a.hours, a.mm, a.unavailableReason]),
    [
      [3, 4.5, null],
      [6, 9.0, null],
    ],
  );
});

/* --------------------------------------------------------------------------
 * FALLBACK
 * ----------------------------------------------------------------------- */

test('a FALLBACK record is named "Open-Meteo" and marked FALLBACK', () => {
  const view = liveRainfallView(fallbackRecord());
  assert.equal(view.tone, 'fallback');
  assert.equal(view.label, 'FALLBACK');
  assert.equal(view.sourceDisplayName, 'Open-Meteo');
  assert.equal(view.isFallback, true);
  assert.equal(view.isRealSatellite, false);
  assert.ok(view.headline.includes('FALLBACK'), view.headline);
});

test('fallback display strings never contain the substring IMERG', () => {
  const view = liveRainfallView(fallbackRecord());
  for (const text of [view.label, view.headline, view.sourceDisplayName ?? '']) {
    assert.ok(!text.toUpperCase().includes('IMERG'), `leaked IMERG in: ${text}`);
  }
  // The acquisition trail may legitimately name the IMERG runs that FAILED.
  assert.ok(view.attempts.some((a) => a.source_kind.includes('IMERG')));
});

test('a fallback record is never presented as a satellite observation', () => {
  const view = liveRainfallView(fallbackRecord());
  assert.equal(view.dataQualityStatus, 'FALLBACK');
  assert.equal(view.expectedLatencyMinutes, null);
  assert.equal(view.sourceDetail, 'Open-Meteo hourly precipitation (FALLBACK)');
});

test('an incomplete window reports its reason instead of a partial sum', () => {
  const view = liveRainfallView(
    fallbackRecord({
      accum_6h_mm: null,
      accum_6h_unavailable_reason:
        'incomplete window: 4 of 6 contiguous 60-minute intervals were retrieved',
    }),
  );
  const six = view.accumulations.find((a) => a.hours === 6);
  assert.equal(six?.mm, null);
  assert.ok(six?.unavailableReason?.includes('incomplete window'));
  assert.equal(formatMm(six?.mm), '—');
});

/* --------------------------------------------------------------------------
 * UNAVAILABLE
 * ----------------------------------------------------------------------- */

test('an UNAVAILABLE record reads "Latest rainfall unavailable" with the backend reason', () => {
  const view = liveRainfallView(unavailableRecord());
  assert.equal(view.tone, 'unavailable');
  assert.equal(view.label, 'UNAVAILABLE');
  assert.equal(view.headline, 'Latest rainfall unavailable');
  assert.equal(
    view.unavailableReason,
    'no IMERG granule was retrievable and the Open-Meteo FALLBACK request failed',
  );
  assert.equal(view.sourceDisplayName, null);
});

test('an UNAVAILABLE record yields no zeroes anywhere in the view', () => {
  const view = liveRainfallView(unavailableRecord());
  assert.equal(view.latestMm, null);
  assert.equal(view.intervalLabel, null);
  assert.equal(view.ageLabel, null);
  assert.equal(view.freshnessLabel, null);
  assert.equal(view.observedAtUtc, null);
  for (const accum of view.accumulations) {
    assert.equal(accum.mm, null);
    assert.ok(accum.unavailableReason);
  }
  assert.equal(formatMm(view.latestMm), '—');
});

test('a null record (nothing fetched, or a failed request) is UNREPORTED, not zero', () => {
  const view = liveRainfallView(null);
  assert.equal(view.tone, 'unreported');
  assert.equal(view.label, 'UNREPORTED');
  assert.equal(view.latestMm, null);
  assert.equal(view.sourceDisplayName, null);
  assert.deepEqual(view.attempts, []);
  assert.equal(formatMm(view.latestMm), '—');
});

/* --------------------------------------------------------------------------
 * All four states, all three quality states
 * ----------------------------------------------------------------------- */

test('every pilot renders all three quality states with the right tone', () => {
  for (const key of PILOT_STATE_KEYS) {
    const name = getPilotConfig(key).name;
    assert.equal(liveRainfallView(realRecord({ state: name })).tone, 'real');
    assert.equal(liveRainfallView(fallbackRecord({ state: name })).tone, 'fallback');
    assert.equal(liveRainfallView(unavailableRecord({ state: name })).tone, 'unavailable');
  }
});

/* --------------------------------------------------------------------------
 * Freshness, cache provenance and refresh cadence
 * ----------------------------------------------------------------------- */

test('freshness and cache_hit are reported, never inferred', () => {
  const fresh = liveRainfallView(realRecord());
  assert.equal(fresh.cacheHit, false);
  assert.equal(fresh.freshnessLabel, 'RECENT');
  assert.equal(fresh.isStale, false);

  const replayed = liveRainfallView(
    realRecord({ cache_hit: true, freshness: freshness({ cache_hit: true }) }),
  );
  assert.equal(replayed.cacheHit, true);
});

test('a stale observation keeps its STALE label', () => {
  const view = liveRainfallView(
    realRecord({
      is_stale: true,
      freshness_label: 'STALE',
      age_seconds: 90000,
    }),
  );
  assert.equal(view.isStale, true);
  assert.equal(view.freshnessLabel, 'STALE');
  assert.equal(view.ageLabel, '25.0 h ago');
});

test('age formatting never invents a value for a missing age', () => {
  assert.equal(formatLiveAge(null), null);
  assert.equal(formatLiveAge(undefined), null);
  assert.equal(formatLiveAge(45), '45s ago');
  assert.equal(formatLiveAge(3600), '60 min ago');
});

test('interval formatting never invents a value', () => {
  assert.equal(formatInterval(null), null);
  assert.equal(formatInterval(30), '30-minute interval');
});

test('the refresh interval can never drop below the backend cache TTL', () => {
  assert.equal(LIVE_RAINFALL_REFRESH_MS, LIVE_RAINFALL_CACHE_TTL_SECONDS * 1000);
  assert.equal(liveRainfallRefreshMs(30), LIVE_RAINFALL_CACHE_TTL_SECONDS * 1000);
  assert.equal(liveRainfallRefreshMs(undefined), LIVE_RAINFALL_REFRESH_MS);
  assert.equal(liveRainfallRefreshMs(null), LIVE_RAINFALL_REFRESH_MS);
  assert.equal(liveRainfallRefreshMs(1800), 1800 * 1000);
});

test('source display names are derived only from source_kind', () => {
  assert.equal(liveRainfallSourceDisplayName(realRecord()), 'NASA IMERG Early');
  assert.equal(liveRainfallSourceDisplayName(fallbackRecord()), 'Open-Meteo');
  assert.equal(liveRainfallSourceDisplayName(unavailableRecord()), null);
  // An unrecognised kind is echoed verbatim rather than guessed at.
  assert.equal(
    liveRainfallSourceDisplayName(realRecord({ source_kind: 'SOME_NEW_PRODUCT' })),
    'SOME_NEW_PRODUCT',
  );
});
