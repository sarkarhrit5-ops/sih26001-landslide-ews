/**
 * Pure view-model for the LIVE rainfall monitoring panel.
 *
 * All honesty-critical decisions about the live rainfall read live in this
 * module rather than in JSX, so `node:test` can exercise them without a browser
 * runner (the same split already used by pilotMapCells.ts):
 *
 *   * a REAL IMERG record is named "NASA IMERG Early" or "NASA IMERG Late" from
 *     its source_kind — never from the quality status alone;
 *   * a FALLBACK record is named "Open-Meteo" and carries a prominent FALLBACK
 *     marker. The string "IMERG" NEVER appears in a fallback display name, even
 *     though the attempts trail legitimately mentions IMERG runs that failed;
 *   * an UNAVAILABLE record reads "Latest rainfall unavailable" plus the
 *     backend's own reason. No numeric field is defaulted to 0 anywhere here:
 *     absence formats as an em dash and the reason is shown beside it;
 *   * the refresh cadence is floored at the backend cache TTL, so the panel
 *     cannot poll faster than the cache it is reading through.
 *
 * This module is entirely separate from the antecedent model rainfall (T-1..T-14)
 * used by the prediction path; it shares no types and no formatting with it.
 */
import type {
  LiveRainfallAttempt,
  LiveRainfallResponse,
  PilotStateKey,
} from '../../services/api';
import { getPilotConfig } from '../../data/nerStates';

/**
 * Backend cache TTL for live_rainfall (SIH_LIVE_RAINFALL_CACHE_TTL_SECONDS
 * default). The panel must not refresh faster than this: inside the TTL every
 * request is served from the same cached record, so a faster interval would add
 * traffic without ever producing a newer observation.
 */
export const LIVE_RAINFALL_CACHE_TTL_SECONDS = 900;

/** Default automatic refresh interval — exactly the TTL, in milliseconds. */
export const LIVE_RAINFALL_REFRESH_MS = LIVE_RAINFALL_CACHE_TTL_SECONDS * 1000;

/**
 * Clamps a requested auto-refresh interval up to the cache TTL. A caller asking
 * for 30 s gets the TTL; a caller asking for 30 min gets 30 min.
 */
export function liveRainfallRefreshMs(requestedSeconds?: number | null): number {
  if (requestedSeconds == null || !Number.isFinite(requestedSeconds)) {
    return LIVE_RAINFALL_REFRESH_MS;
  }
  return Math.max(LIVE_RAINFALL_CACHE_TTL_SECONDS, requestedSeconds) * 1000;
}

/**
 * The monitoring endpoint for one pilot. The `?state=` value is the backend AOI
 * LABEL from the registry (`config.name`, e.g. "Arunachal Pradesh"), which is
 * deliberately not the route token `key` nor the `stateId`. An unknown key
 * throws in getPilotConfig rather than resolving to another state's AOI.
 */
export function liveRainfallEndpoint(state: PilotStateKey | string): string {
  const config = getPilotConfig(state);
  return `/api/v1/rainfall/latest?state=${encodeURIComponent(config.name)}`;
}

export type LiveRainfallTone = 'real' | 'fallback' | 'unavailable' | 'unreported';

/** Display names, keyed by the backend source_kind vocabulary. */
export const LIVE_SOURCE_DISPLAY_NAMES: Record<string, string> = {
  IMERG_HHR_EARLY: 'NASA IMERG Early',
  IMERG_HHR_LATE: 'NASA IMERG Late',
  OPEN_METEO_FALLBACK: 'Open-Meteo',
};

const IMERG_SOURCE_KINDS = new Set(['IMERG_HHR_EARLY', 'IMERG_HHR_LATE']);

/**
 * The short display name for the source that actually produced the value, or
 * null when the record names no source (an UNAVAILABLE refusal). Derived ONLY
 * from source_kind, so a FALLBACK record can never inherit an IMERG name.
 */
export function liveRainfallSourceDisplayName(
  record: LiveRainfallResponse | null | undefined,
): string | null {
  const kind = record?.source_kind;
  if (!kind) return null;
  return LIVE_SOURCE_DISPLAY_NAMES[kind] ?? kind;
}

/**
 * Presentation tone. REAL requires BOTH a REAL quality status and an IMERG
 * source_kind, so neither field alone can promote a record to a satellite
 * observation claim.
 */
export function liveRainfallTone(
  record: LiveRainfallResponse | null | undefined,
): LiveRainfallTone {
  if (!record) return 'unreported';
  const status = (record.data_quality_status || '').toUpperCase();
  const kind = record.source_kind || '';
  if (status === 'UNAVAILABLE') return 'unavailable';
  if (status === 'FALLBACK' || kind === 'OPEN_METEO_FALLBACK') return 'fallback';
  if (status === 'REAL' && IMERG_SOURCE_KINDS.has(kind)) return 'real';
  return 'unreported';
}

/** A millimetre reading, or an em dash. Never 0 as a stand-in for "unknown". */
export function formatMm(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—';
  return `${value.toFixed(2)} mm`;
}

/** Human age from a measured second count. null stays null (not "0 min"). */
export function formatLiveAge(ageSeconds: number | null | undefined): string | null {
  if (ageSeconds == null || !Number.isFinite(ageSeconds)) return null;
  const seconds = Math.max(0, Math.round(ageSeconds));
  if (seconds < 90) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} min ago`;
  const hours = seconds / 3600;
  if (hours < 48) return `${hours.toFixed(1)} h ago`;
  return `${(hours / 24).toFixed(1)} d ago`;
}

/** The observation interval, e.g. "30-minute interval". */
export function formatInterval(intervalMinutes: number | null | undefined): string | null {
  if (intervalMinutes == null || !Number.isFinite(intervalMinutes)) return null;
  return `${intervalMinutes}-minute interval`;
}

export interface LiveAccumulation {
  hours: number;
  mm: number | null;
  /** Why the window is absent — shown instead of a partial or zero sum. */
  unavailableReason: string | null;
}

export interface LiveRainfallView {
  tone: LiveRainfallTone;
  /** Uppercase tag: REAL / FALLBACK / UNAVAILABLE / UNREPORTED. */
  label: string;
  /** One-line headline the panel prints above the numbers. */
  headline: string;
  /** "NASA IMERG Early" | "NASA IMERG Late" | "Open-Meteo" | null. */
  sourceDisplayName: string | null;
  /** The backend's long product label, for the audit line. */
  sourceDetail: string | null;
  isFallback: boolean;
  /** True only for a genuine IMERG observation. */
  isRealSatellite: boolean;
  dataQualityStatus: string | null;
  latestMm: number | null;
  intervalLabel: string | null;
  accumulations: LiveAccumulation[];
  observedAtUtc: string | null;
  fetchedAtUtc: string | null;
  ageLabel: string | null;
  freshnessLabel: string | null;
  isStale: boolean | null;
  stalenessThresholdMinutes: number | null;
  expectedLatencyMinutes: number | null;
  cacheHit: boolean | null;
  granulesUsed: number | null;
  /** The backend's own reason string; null when the record is not a refusal. */
  unavailableReason: string | null;
  attempts: LiveRainfallAttempt[];
  valueSemantics: string | null;
}

const TONE_LABELS: Record<LiveRainfallTone, string> = {
  real: 'REAL',
  fallback: 'FALLBACK',
  unavailable: 'UNAVAILABLE',
  unreported: 'UNREPORTED',
};

const UNAVAILABLE_HEADLINE = 'Latest rainfall unavailable';

function headlineFor(tone: LiveRainfallTone, sourceName: string | null): string {
  if (tone === 'unavailable') return UNAVAILABLE_HEADLINE;
  if (tone === 'unreported') {
    return 'Latest rainfall provenance not reported by the backend';
  }
  if (tone === 'fallback') {
    return `FALLBACK — ${sourceName ?? 'non-satellite source'}, not a satellite observation`;
  }
  return `${sourceName ?? 'Satellite observation'} — latest available observation`;
}

/**
 * Folds a live-rainfall record into everything the panel renders. A null record
 * (nothing fetched yet, or a transport failure) yields the UNREPORTED view with
 * no numbers — it never borrows the previous record's values.
 */
export function liveRainfallView(
  record: LiveRainfallResponse | null | undefined,
): LiveRainfallView {
  const tone = liveRainfallTone(record);
  const sourceDisplayName = liveRainfallSourceDisplayName(record);
  const freshness = record?.freshness ?? null;
  const ageSeconds = record?.age_seconds ?? freshness?.age_seconds ?? null;

  return {
    tone,
    label: TONE_LABELS[tone],
    headline: headlineFor(tone, sourceDisplayName),
    sourceDisplayName,
    sourceDetail: record?.source ?? null,
    isFallback: tone === 'fallback',
    isRealSatellite: tone === 'real',
    dataQualityStatus: record?.data_quality_status ?? null,
    latestMm: record?.latest_available_rainfall_mm ?? null,
    intervalLabel: formatInterval(record?.interval_minutes),
    accumulations: [
      {
        hours: 3,
        mm: record?.accum_3h_mm ?? null,
        unavailableReason: record?.accum_3h_unavailable_reason ?? null,
      },
      {
        hours: 6,
        mm: record?.accum_6h_mm ?? null,
        unavailableReason: record?.accum_6h_unavailable_reason ?? null,
      },
    ],
    observedAtUtc: record?.observed_at_utc ?? null,
    fetchedAtUtc: record?.fetched_at_utc ?? null,
    ageLabel: formatLiveAge(ageSeconds),
    freshnessLabel: record?.freshness_label ?? freshness?.freshness_label ?? null,
    isStale: record?.is_stale ?? null,
    stalenessThresholdMinutes: record?.staleness_threshold_minutes ?? null,
    expectedLatencyMinutes: record?.expected_product_latency_minutes ?? null,
    cacheHit: record?.cache_hit ?? freshness?.cache_hit ?? null,
    granulesUsed: record?.granules_used ?? null,
    unavailableReason: record?.unavailable_reason ?? null,
    attempts: record?.attempts ?? [],
    valueSemantics: record?.value_semantics ?? null,
  };
}
