/**
 * Shared, PURE map-cell logic for the four pilot consoles.
 *
 * Why this module exists
 *   The four *Map.tsx components were near-identical clones, each rebuilding the
 *   same cell geometry and risk-colour decisions. Everything here is a pure
 *   function over the /predict/<state>/map document: no Leaflet, no React, no
 *   fetch, no module-level state. That keeps the honesty rules in ONE place and
 *   makes them executable under `node --test` without a browser test runner.
 *
 * Honesty rules encoded here (they mirror the backend's own)
 *   * A cell is never invented and never dropped. An UNAVAILABLE cell is
 *     classified as UNAVAILABLE and drawn hollow; it is not silently rendered as
 *     LOW risk, and it carries no probability.
 *   * A cell rectangle is reconstructed ONLY from the grid's own
 *     cell_height_deg / cell_width_deg. If those are absent or non-finite we
 *     return null so the caller draws nothing, rather than assuming a cell size.
 *   * Rainfall provenance is read from the response; when the response says
 *     FALLBACK it is labelled FALLBACK, and when it reports nothing the label is
 *     "unreported", never "REAL".
 */
import type {
  PilotMapFeature,
  PilotMapResponse,
  PilotMapRainfallView,
  PilotStateKey,
  RainfallProvenanceBlock,
  SikkimPredictionGrid,
  WarningLevel,
} from '../../services/api';
import { PILOT_STATE_KEYS, getPilotConfig } from '../../data/nerStates';

/** Fill colour per model warning class — the app's existing risk palette, unchanged. */
export const RISK_FILL: Record<string, string> = {
  LOW: '#22c55e',
  MEDIUM: '#eab308',
  HIGH: '#f97316',
  EXTREME: '#ef4444',
};

/** Colour used when a class arrives that is not in the palette (never guessed as LOW). */
export const UNKNOWN_RISK_FILL = '#64748b';

export const RISK_ORDER: WarningLevel[] = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'];

/** Leaflet bounds: [[southLat, westLon], [northLat, eastLon]]. */
export type LeafletBounds = [[number, number], [number, number]];

export interface PilotMapCell {
  cellId: string | null;
  /** Cell centre in Leaflet order (lat, lon) — the point the model was sampled at. */
  lat: number;
  lon: number;
  bounds: LeafletBounds | null;
  scored: boolean;
  probability: number | null;
  riskClass: WarningLevel | null;
  exceedsDecisionThreshold: boolean | null;
  status: string | null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

/**
 * Path for one pilot's compact map endpoint. The state token comes from the
 * registry (getPilotConfig throws on an unknown key), so a typo can never
 * silently resolve to another state's data.
 */
export function pilotMapEndpoint(
  state: PilotStateKey,
  options?: { date?: string; step?: number },
): string {
  const config = getPilotConfig(state);
  const params = new URLSearchParams();
  if (options?.date) params.set('date', options.date);
  if (options?.step != null) params.set('step', String(options.step));
  const query = params.toString();
  return `/api/v1/predict/${config.key}/map${query ? `?${query}` : ''}`;
}

/** Path for the FULL grid endpoint — used only by the lazy audit panel, never for the map. */
export function pilotGridEndpoint(
  state: PilotStateKey,
  options?: { date?: string; step?: number },
): string {
  const config = getPilotConfig(state);
  const params = new URLSearchParams();
  if (options?.date) params.set('date', options.date);
  if (options?.step != null) params.set('step', String(options.step));
  const query = params.toString();
  return `/api/v1/predict/${config.key}/grid${query ? `?${query}` : ''}`;
}

export function isPilotStateKey(value: string): value is PilotStateKey {
  return (PILOT_STATE_KEYS as string[]).includes(value);
}

/**
 * Rebuild a cell's rectangle from the grid's own cell size around the feature
 * centre. Returns null when the grid does not report a usable cell size — the
 * caller then draws no rectangle rather than inventing an extent.
 */
export function cellBoundsFromCenter(
  lat: number,
  lon: number,
  grid: SikkimPredictionGrid | null | undefined,
): LeafletBounds | null {
  if (!grid) return null;
  const height = grid.cell_height_deg;
  const width = grid.cell_width_deg;
  if (!isFiniteNumber(height) || !isFiniteNumber(width)) return null;
  if (height <= 0 || width <= 0) return null;
  const halfLat = height / 2;
  const halfLon = width / 2;
  return [
    [lat - halfLat, lon - halfLon],
    [lat + halfLat, lon + halfLon],
  ];
}

/**
 * Project one GeoJSON feature into the shape the map layer draws.
 *
 * `scored` is true ONLY when the backend said status OK, gave a risk_class and
 * gave a probability. Any other combination is treated as not scored, which is
 * what makes an UNAVAILABLE cell render hollow instead of coloured.
 */
export function toPilotMapCell(
  feature: PilotMapFeature,
  grid: SikkimPredictionGrid | null | undefined,
): PilotMapCell | null {
  const coords = feature?.geometry?.coordinates;
  if (!Array.isArray(coords) || coords.length < 2) return null;
  const lon = coords[0];
  const lat = coords[1];
  if (!isFiniteNumber(lat) || !isFiniteNumber(lon)) return null;
  const props = feature.properties ?? ({} as PilotMapFeature['properties']);
  const probability = isFiniteNumber(props.probability) ? props.probability : null;
  const riskClass = props.risk_class ?? null;
  const scored = props.status === 'OK' && riskClass != null && probability != null;
  return {
    cellId: props.cell_id ?? feature.id ?? null,
    lat,
    lon,
    bounds: cellBoundsFromCenter(lat, lon, grid),
    scored,
    probability: scored ? probability : null,
    riskClass: scored ? riskClass : null,
    exceedsDecisionThreshold: props.exceeds_decision_threshold ?? null,
    status: props.status ?? null,
  };
}

/**
 * All drawable cells from a map document, in response order. Features that carry
 * no usable coordinate are skipped (they cannot be placed honestly) rather than
 * being defaulted to 0,0.
 */
export function toPilotMapCells(response: PilotMapResponse | null | undefined): PilotMapCell[] {
  const features = response?.features;
  if (!Array.isArray(features)) return [];
  const grid = response?.grid ?? null;
  const cells: PilotMapCell[] = [];
  for (const feature of features) {
    const cell = toPilotMapCell(feature, grid);
    if (cell) cells.push(cell);
  }
  return cells;
}

export function riskFill(riskClass: string | null | undefined): string {
  if (!riskClass) return UNKNOWN_RISK_FILL;
  return RISK_FILL[riskClass] ?? UNKNOWN_RISK_FILL;
}

/** Risk classes actually scored, so the legend never advertises a class we don't have. */
export function scoredRiskClasses(cells: PilotMapCell[]): Set<string> {
  const seen = new Set<string>();
  for (const cell of cells) {
    if (cell.scored && cell.riskClass) seen.add(cell.riskClass);
  }
  return seen;
}

export function hasUnavailableCells(cells: PilotMapCell[]): boolean {
  return cells.some((cell) => !cell.scored);
}

/* ------------------------------------------------------------------ *
 * Rainfall provenance labelling
 * ------------------------------------------------------------------ */

export type RainfallProvenanceTone = 'real' | 'fallback' | 'unavailable' | 'unreported';

export interface RainfallProvenanceView {
  tone: RainfallProvenanceTone;
  /** Short badge, e.g. "REAL / NASA IMERG" or "RAINFALL SOURCE - ERA5 FALLBACK". */
  label: string;
  /** The producer's own source string, when it gave one. */
  source: string | null;
  sourceKind: string | null;
  dataQualityStatus: string | null;
  runType: string | null;
  /** The day the rainfall was actually observed for (may lag the requested day). */
  observationDate: string | null;
  requestedDate: string | null;
  fetchedAtUtc: string | null;
  observationLagDays: number | null;
  cacheHit: boolean | null;
  ageSeconds: number | null;
  expiresInSeconds: number | null;
  windowDays: number | null;
  note: string | null;
  caveats: string[];
  fallbackWarning: string | null;
}

/**
 * Derive the badge from what the response ACTUALLY reports, in this precedence:
 * an explicit fallback flag or FALLBACK status wins; an explicit UNAVAILABLE
 * status is reported as unavailable; IMERG is claimed only when the payload says
 * IMERG; anything else is "unreported provenance". There is no branch in which a
 * missing field yields a REAL/IMERG label.
 */
export function rainfallProvenanceTone(
  rainfall: PilotMapRainfallView | null | undefined,
  provenance?: RainfallProvenanceBlock | null,
): RainfallProvenanceTone {
  const isFallback = rainfall?.is_fallback ?? provenance?.is_fallback ?? null;
  const quality = (rainfall?.data_quality_status ?? provenance?.data_quality_status ?? null) as
    | string
    | null;
  const sourceKind = (rainfall?.source_kind ?? provenance?.source_kind ?? null) as string | null;
  if (isFallback === true) return 'fallback';
  if (quality === 'FALLBACK' || sourceKind === 'OPEN_METEO_FALLBACK') return 'fallback';
  if (quality === 'UNAVAILABLE') return 'unavailable';
  if (quality === 'REAL' || sourceKind === 'IMERG') return 'real';
  return 'unreported';
}

const TONE_LABELS: Record<RainfallProvenanceTone, string> = {
  real: 'REAL / NASA IMERG',
  fallback: 'RAINFALL SOURCE - ERA5 FALLBACK',
  unavailable: 'UNAVAILABLE / no rainfall obtained',
  unreported: 'UNREPORTED rainfall provenance',
};

export function rainfallProvenanceView(
  rainfall: PilotMapRainfallView | null | undefined,
  provenance?: RainfallProvenanceBlock | null,
): RainfallProvenanceView {
  const tone = rainfallProvenanceTone(rainfall, provenance);
  const freshness = rainfall?.freshness ?? provenance?.freshness ?? null;
  const caveats = rainfall?.caveats ?? provenance?.caveats ?? [];
  return {
    tone,
    label: TONE_LABELS[tone],
    source: rainfall?.source ?? provenance?.source ?? null,
    sourceKind: (rainfall?.source_kind ?? provenance?.source_kind ?? null) as string | null,
    dataQualityStatus: (rainfall?.data_quality_status ??
      provenance?.data_quality_status ??
      null) as string | null,
    runType: rainfall?.run_type ?? null,
    observationDate:
      rainfall?.rainfall_observation_date ?? provenance?.rainfall_observation_date ?? null,
    requestedDate: rainfall?.requested_date ?? provenance?.requested_date ?? null,
    fetchedAtUtc: rainfall?.fetched_at_utc ?? provenance?.fetched_at_utc ?? null,
    observationLagDays: isFiniteNumber(freshness?.observation_lag_days)
      ? (freshness?.observation_lag_days as number)
      : null,
    cacheHit: typeof freshness?.cache_hit === 'boolean' ? freshness.cache_hit : null,
    ageSeconds: isFiniteNumber(freshness?.age_seconds) ? (freshness?.age_seconds as number) : null,
    expiresInSeconds: isFiniteNumber(freshness?.expires_in_seconds)
      ? (freshness?.expires_in_seconds as number)
      : null,
    windowDays: isFiniteNumber(rainfall?.window_days) ? (rainfall?.window_days as number) : null,
    note: rainfall?.note ?? null,
    caveats: Array.isArray(caveats) ? [...caveats] : [],
    fallbackWarning: provenance?.fallback_warning ?? null,
  };
}

/** Human-readable observation age, or null when the payload did not report one. */
export function formatObservationLag(lagDays: number | null): string | null {
  if (lagDays == null) return null;
  if (lagDays <= 0) return 'same day';
  if (lagDays === 1) return '1 day behind the requested date';
  return `${lagDays} days behind the requested date`;
}

/** Human-readable cache age, or null when the payload did not report one. */
export function formatCacheAge(view: RainfallProvenanceView): string | null {
  if (view.cacheHit == null && view.ageSeconds == null) return null;
  if (view.cacheHit === false) return 'fetched fresh for this request';
  if (view.ageSeconds == null) return 'served from cache';
  const seconds = Math.max(0, Math.round(view.ageSeconds));
  if (seconds < 60) return `served from cache, ${seconds}s old`;
  return `served from cache, ${Math.round(seconds / 60)}m old`;
}
