/**
 * Node built-in tests for the shared pilot map-cell logic.
 *
 * There is no browser test runner in this project, so the honesty-critical logic
 * lives in the pure module `components/pilot/pilotMapCells.ts` and is exercised
 * here with `node:test`. Run with:
 *
 *   cd frontend && node --experimental-strip-types --test src/__tests__/pilotMap.test.ts
 *
 * These tests assert the properties the UI depends on for truthfulness:
 *   * each of the four pilots resolves to ITS OWN /map endpoint (never another
 *     state's), and /grid stays a separate, audit-only path;
 *   * a cell rectangle is reconstructed only from the grid's own cell size;
 *   * an UNAVAILABLE cell never becomes a coloured (implicitly safe) cell;
 *   * an unknown risk class never falls back to LOW;
 *   * rainfall provenance is never labelled REAL/IMERG unless the payload says so.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  RISK_FILL,
  UNKNOWN_RISK_FILL,
  cellBoundsFromCenter,
  formatCacheAge,
  formatObservationLag,
  hasUnavailableCells,
  isPilotStateKey,
  pilotGridEndpoint,
  pilotMapEndpoint,
  rainfallProvenanceTone,
  rainfallProvenanceView,
  riskFill,
  scoredRiskClasses,
  toPilotMapCells,
} from '../components/pilot/pilotMapCells.ts';
import { PILOT_STATE_KEYS, getPilotConfig } from '../data/nerStates.ts';
import type { PilotMapResponse, SikkimPredictionGrid } from '../services/api.ts';

const GRID: SikkimPredictionGrid = {
  step_deg: 0.05,
  n_lat: 3,
  n_lon: 3,
  cell_count: 9,
  cell_height_deg: 0.05,
  cell_width_deg: 0.05,
};

function mapResponse(
  features: PilotMapResponse['features'],
  grid: SikkimPredictionGrid | null = GRID,
): PilotMapResponse {
  return {
    type: 'FeatureCollection',
    features,
    grid,
  } as PilotMapResponse;
}

function feature(
  lon: number,
  lat: number,
  props: Record<string, unknown>,
): PilotMapResponse['features'][number] {
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [lon, lat] },
    properties: props,
  } as unknown as PilotMapResponse['features'][number];
}

/* ---------------------------------------------------------------- *
 * Endpoint selection — one per state, from the registry
 * ---------------------------------------------------------------- */

test('each pilot state resolves to its own /map endpoint', () => {
  assert.equal(pilotMapEndpoint('sikkim'), '/api/v1/predict/sikkim/map');
  assert.equal(pilotMapEndpoint('assam'), '/api/v1/predict/assam/map');
  assert.equal(pilotMapEndpoint('arunachal'), '/api/v1/predict/arunachal/map');
  assert.equal(pilotMapEndpoint('meghalaya'), '/api/v1/predict/meghalaya/map');
});

test('all four registry keys produce distinct map endpoints', () => {
  const endpoints = PILOT_STATE_KEYS.map((k) => pilotMapEndpoint(k));
  assert.equal(endpoints.length, 4);
  assert.equal(new Set(endpoints).size, 4);
});

test('/grid stays a separate audit-only path per state', () => {
  for (const key of PILOT_STATE_KEYS) {
    const grid = pilotGridEndpoint(key);
    assert.ok(grid.endsWith('/grid'), `${key} grid path: ${grid}`);
    assert.notEqual(grid, pilotMapEndpoint(key));
  }
});

test('map endpoint forwards only the options it was given', () => {
  assert.equal(pilotMapEndpoint('assam'), '/api/v1/predict/assam/map');
  assert.equal(
    pilotMapEndpoint('assam', { date: '2026-08-30' }),
    '/api/v1/predict/assam/map?date=2026-08-30',
  );
  assert.equal(
    pilotMapEndpoint('meghalaya', { date: '2026-08-30', step: 0.05 }),
    '/api/v1/predict/meghalaya/map?date=2026-08-30&step=0.05',
  );
});

test('an unknown state key throws rather than resolving to another state', () => {
  assert.equal(isPilotStateKey('sikkim'), true);
  assert.equal(isPilotStateKey('nagaland'), false);
  // @ts-expect-error deliberately invalid key
  assert.throws(() => pilotMapEndpoint('nagaland'));
});

test('registry keys carry their real backend state ids and unchanged AOIs', () => {
  assert.equal(getPilotConfig('arunachal').stateId, 'arunachal_pradesh');
  assert.equal(getPilotConfig('sikkim').stateId, 'sikkim');
  const meg = getPilotConfig('meghalaya').aoi;
  assert.deepEqual(
    [meg.minLat, meg.maxLat, meg.minLon, meg.maxLon],
    [25.0, 25.99, 91.0, 92.8],
  );
  const sik = getPilotConfig('sikkim').aoi;
  assert.deepEqual([sik.minLat, sik.maxLat, sik.minLon, sik.maxLon], [27.0, 28.1, 88.0, 88.9]);
});

/* ---------------------------------------------------------------- *
 * Cell geometry reconstruction
 * ---------------------------------------------------------------- */

test('bounds are reconstructed from the grid cell size around the centre', () => {
  const bounds = cellBoundsFromCenter(27.5, 88.4, GRID);
  assert.ok(bounds);
  assert.ok(Math.abs(bounds![0][0] - 27.475) < 1e-9);
  assert.ok(Math.abs(bounds![0][1] - 88.375) < 1e-9);
  assert.ok(Math.abs(bounds![1][0] - 27.525) < 1e-9);
  assert.ok(Math.abs(bounds![1][1] - 88.425) < 1e-9);
});

test('no rectangle is invented when the grid reports no usable cell size', () => {
  assert.equal(cellBoundsFromCenter(27.5, 88.4, null), null);
  assert.equal(
    cellBoundsFromCenter(27.5, 88.4, { ...GRID, cell_height_deg: 0 }),
    null,
  );
  assert.equal(
    cellBoundsFromCenter(27.5, 88.4, {
      ...GRID,
      cell_width_deg: undefined as unknown as number,
    }),
    null,
  );
});

/* ---------------------------------------------------------------- *
 * Cell projection honesty
 * ---------------------------------------------------------------- */

test('a scored cell keeps its probability and class', () => {
  const cells = toPilotMapCells(
    mapResponse([
      feature(88.4, 27.5, {
        cell_id: 'r0c0',
        status: 'OK',
        probability: 0.42,
        risk_class: 'HIGH',
        exceeds_decision_threshold: true,
      }),
    ]),
  );
  assert.equal(cells.length, 1);
  assert.equal(cells[0].scored, true);
  assert.equal(cells[0].probability, 0.42);
  assert.equal(cells[0].riskClass, 'HIGH');
  assert.equal(cells[0].exceedsDecisionThreshold, true);
  assert.equal(cells[0].cellId, 'r0c0');
  assert.ok(cells[0].bounds);
});

test('an UNAVAILABLE cell is kept but never coloured or given a probability', () => {
  const cells = toPilotMapCells(
    mapResponse([
      feature(91.5, 25.4, {
        cell_id: 'r1c1',
        status: 'UNAVAILABLE',
        probability: null,
        risk_class: null,
        exceeds_decision_threshold: null,
      }),
    ]),
  );
  assert.equal(cells.length, 1);
  assert.equal(cells[0].scored, false);
  assert.equal(cells[0].probability, null);
  assert.equal(cells[0].riskClass, null);
  assert.equal(cells[0].status, 'UNAVAILABLE');
  assert.equal(hasUnavailableCells(cells), true);
  assert.equal(scoredRiskClasses(cells).size, 0);
});

test('a cell with OK status but no class/probability is not treated as scored', () => {
  const cells = toPilotMapCells(
    mapResponse([feature(92.0, 26.0, { status: 'OK', probability: null, risk_class: null })]),
  );
  assert.equal(cells[0].scored, false);
  assert.equal(cells[0].riskClass, null);
});

test('features without a usable coordinate are skipped, not placed at 0,0', () => {
  const response = mapResponse([
    { type: 'Feature', geometry: { type: 'Point', coordinates: [] }, properties: {} },
    feature(93.0, 26.2, { status: 'OK', probability: 0.1, risk_class: 'LOW' }),
  ] as unknown as PilotMapResponse['features']);
  const cells = toPilotMapCells(response);
  assert.equal(cells.length, 1);
  assert.equal(cells[0].lon, 93.0);
});

test('a null/empty map document yields no cells at all', () => {
  assert.deepEqual(toPilotMapCells(null), []);
  assert.deepEqual(toPilotMapCells(undefined), []);
  assert.deepEqual(toPilotMapCells(mapResponse([])), []);
});

test('cells still project when the grid is missing, just without rectangles', () => {
  const cells = toPilotMapCells(
    mapResponse([feature(88.4, 27.5, { status: 'OK', probability: 0.3, risk_class: 'MEDIUM' })], null),
  );
  assert.equal(cells.length, 1);
  assert.equal(cells[0].bounds, null);
  assert.equal(cells[0].scored, true);
});

test('the legend only advertises classes that were actually scored', () => {
  const cells = toPilotMapCells(
    mapResponse([
      feature(88.4, 27.5, { status: 'OK', probability: 0.9, risk_class: 'EXTREME' }),
      feature(88.5, 27.5, { status: 'OK', probability: 0.1, risk_class: 'LOW' }),
      feature(88.6, 27.5, { status: 'UNAVAILABLE', probability: null, risk_class: null }),
    ]),
  );
  const classes = scoredRiskClasses(cells);
  assert.deepEqual([...classes].sort(), ['EXTREME', 'LOW']);
  assert.equal(hasUnavailableCells(cells), true);
});

/* ---------------------------------------------------------------- *
 * Colour mapping
 * ---------------------------------------------------------------- */

test('an unknown or absent risk class never falls back to the LOW colour', () => {
  assert.equal(riskFill('LOW'), RISK_FILL.LOW);
  assert.equal(riskFill('EXTREME'), RISK_FILL.EXTREME);
  assert.equal(riskFill(null), UNKNOWN_RISK_FILL);
  assert.equal(riskFill(undefined), UNKNOWN_RISK_FILL);
  assert.equal(riskFill('SOMETHING_NEW'), UNKNOWN_RISK_FILL);
  assert.notEqual(riskFill('SOMETHING_NEW'), RISK_FILL.LOW);
});

/* ---------------------------------------------------------------- *
 * Rainfall provenance
 * ---------------------------------------------------------------- */

test('IMERG is claimed only when the payload says IMERG', () => {
  assert.equal(
    rainfallProvenanceTone({ source_kind: 'IMERG', data_quality_status: 'REAL' } as never),
    'real',
  );
  assert.equal(rainfallProvenanceTone({ data_quality_status: 'REAL' } as never), 'real');
});

test('a fallback result is labelled FALLBACK, never REAL', () => {
  const byFlag = rainfallProvenanceView({ is_fallback: true, source: 'x' } as never);
  assert.equal(byFlag.tone, 'fallback');
  assert.equal(byFlag.label, 'RAINFALL SOURCE - ERA5 FALLBACK');

  assert.equal(
    rainfallProvenanceTone({ source_kind: 'OPEN_METEO_FALLBACK' } as never),
    'fallback',
  );
  assert.equal(rainfallProvenanceTone({ data_quality_status: 'FALLBACK' } as never), 'fallback');
});

test('an explicit fallback flag outranks a REAL status claim', () => {
  assert.equal(
    rainfallProvenanceTone({
      is_fallback: true,
      data_quality_status: 'REAL',
      source_kind: 'IMERG',
    } as never),
    'fallback',
  );
});

test('missing rainfall provenance is unreported, not REAL', () => {
  assert.equal(rainfallProvenanceTone(null), 'unreported');
  assert.equal(rainfallProvenanceTone(undefined), 'unreported');
  assert.equal(rainfallProvenanceTone({} as never), 'unreported');
  assert.equal(rainfallProvenanceView(null).label, 'UNREPORTED rainfall provenance');
});

test('an UNAVAILABLE rainfall status is reported as unavailable', () => {
  assert.equal(rainfallProvenanceTone({ data_quality_status: 'UNAVAILABLE' } as never), 'unavailable');
});

test('the provenance block is used when the rainfall view omits the fields', () => {
  const view = rainfallProvenanceView(null, {
    source_kind: 'OPEN_METEO_FALLBACK',
    data_quality_status: 'FALLBACK',
    source: 'Open-Meteo ERA5 archive (FALLBACK)',
    rainfall_observation_date: '2026-08-29',
    requested_date: '2026-08-30',
    fetched_at_utc: '2026-08-30T06:00:00Z',
    is_fallback: true,
    fallback_warning: 'IMERG unavailable',
    caveats: ['reanalysis, not satellite'],
    freshness: { cache_hit: true, age_seconds: 120, observation_lag_days: 1 },
  } as never);
  assert.equal(view.tone, 'fallback');
  assert.equal(view.source, 'Open-Meteo ERA5 archive (FALLBACK)');
  assert.equal(view.observationDate, '2026-08-29');
  assert.equal(view.requestedDate, '2026-08-30');
  assert.equal(view.observationLagDays, 1);
  assert.equal(view.cacheHit, true);
  assert.equal(view.fallbackWarning, 'IMERG unavailable');
  assert.deepEqual(view.caveats, ['reanalysis, not satellite']);
});

test('freshness formatting returns null rather than inventing a value', () => {
  assert.equal(formatObservationLag(null), null);
  assert.equal(formatObservationLag(0), 'same day');
  assert.equal(formatObservationLag(1), '1 day behind the requested date');
  assert.equal(formatObservationLag(3), '3 days behind the requested date');

  assert.equal(formatCacheAge(rainfallProvenanceView(null)), null);
  assert.equal(
    formatCacheAge(rainfallProvenanceView({ freshness: { cache_hit: false } } as never)),
    'fetched fresh for this request',
  );
  assert.equal(
    formatCacheAge(
      rainfallProvenanceView({ freshness: { cache_hit: true, age_seconds: 30 } } as never),
    ),
    'served from cache, 30s old',
  );
  assert.equal(
    formatCacheAge(
      rainfallProvenanceView({ freshness: { cache_hit: true, age_seconds: 600 } } as never),
    ),
    'served from cache, 10m old',
  );
});
