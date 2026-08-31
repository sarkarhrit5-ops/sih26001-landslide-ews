/**
 * Shared LIVE rainfall panel — one component, all four pilot consoles.
 *
 * This panel is the MONITORING read (GET /api/v1/rainfall/latest?state=<state>),
 * and it is deliberately distinct from the antecedent/model rainfall provenance
 * banner that accompanies a prediction:
 *
 *   * it issues ONE request per refresh for the whole AOI — never one per map
 *     cell, and never a /predict/<state>/map or /grid call;
 *   * it refreshes no faster than the backend cache TTL (15 min), plus an
 *     explicit operator refresh button;
 *   * REAL records are named "NASA IMERG Early" / "NASA IMERG Late"; a FALLBACK
 *     record is named "Open-Meteo" and marked FALLBACK prominently, and the word
 *     IMERG never appears in its display name;
 *   * an UNAVAILABLE record prints "Latest rainfall unavailable" and the
 *     backend's own reason. No missing number is ever rendered as 0.
 *
 * Every honesty-critical decision is taken by the pure module
 * components/pilot/liveRainfallView.ts, which is unit-tested under node:test.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CloudOff,
  CloudRain,
  HelpCircle,
  LoaderCircle,
  RefreshCw,
  Satellite,
} from 'lucide-react';
import { apiService } from '../../services/api';
import type { LiveRainfallResponse, PilotStateKey } from '../../services/api';
import { getPilotConfig } from '../../data/nerStates';
import {
  LIVE_RAINFALL_CACHE_TTL_SECONDS,
  formatMm,
  liveRainfallRefreshMs,
  liveRainfallView,
} from '../pilot/liveRainfallView';
import type { LiveRainfallTone } from '../pilot/liveRainfallView';
import { Eyebrow, cn } from './ui';

interface LiveRainfallPanelProps {
  /** Pilot route token; the ?state= label is read from the registry. */
  state: PilotStateKey;
  /** Optional auto-refresh interval in seconds; floored at the cache TTL. */
  refreshSeconds?: number;
  className?: string;
}

/** Per-tone presentation. Colour never contradicts the label. */
const TONE_STYLES: Record<
  LiveRainfallTone,
  { border: string; bg: string; icon: string; title: string; Icon: typeof CloudRain }
> = {
  real: {
    border: 'border-emerald-800/70',
    bg: 'bg-emerald-950/30',
    icon: 'text-emerald-400',
    title: 'text-emerald-200',
    Icon: Satellite,
  },
  fallback: {
    border: 'border-amber-700/70',
    bg: 'bg-amber-950/30',
    icon: 'text-amber-400',
    title: 'text-amber-200',
    Icon: AlertTriangle,
  },
  unavailable: {
    border: 'border-rose-900/70',
    bg: 'bg-rose-950/30',
    icon: 'text-rose-400',
    title: 'text-rose-200',
    Icon: CloudOff,
  },
  unreported: {
    border: 'border-slate-700',
    bg: 'bg-slate-900/50',
    icon: 'text-slate-400',
    title: 'text-slate-200',
    Icon: HelpCircle,
  },
};

function Readout({
  label,
  value,
  hint,
  accent = 'slate',
}: {
  label: string;
  value: string;
  hint?: string | null;
  accent?: 'emerald' | 'amber' | 'rose' | 'slate';
}) {
  const color =
    accent === 'emerald'
      ? 'text-emerald-300'
      : accent === 'amber'
        ? 'text-amber-300'
        : accent === 'rose'
          ? 'text-rose-300'
          : 'text-slate-100';
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2.5">
      <div className={cn('font-mono text-base font-bold tabular-nums leading-none', color)}>
        {value}
      </div>
      <div className="mt-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </div>
      {hint ? <div className="mt-1 text-[10px] leading-snug text-slate-500">{hint}</div> : null}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <span className="whitespace-nowrap">
      <span className="text-slate-500">{label} </span>
      <span className="text-slate-300">{value}</span>
    </span>
  );
}

export function LiveRainfallPanel({ state, refreshSeconds, className }: LiveRainfallPanelProps) {
  const config = getPilotConfig(state);
  const intervalMs = liveRainfallRefreshMs(refreshSeconds);

  const [record, setRecord] = useState<LiveRainfallResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [transportError, setTransportError] = useState<string | null>(null);
  const [lastAttemptAt, setLastAttemptAt] = useState<string | null>(null);
  /** Counts completed requests so the caller can see one call per refresh. */
  const requestCount = useRef(0);

  const load = useCallback(async () => {
    setLoading(true);
    requestCount.current += 1;
    try {
      const next = await apiService.getLatestRainfall(state);
      setRecord(next);
      setTransportError(null);
    } catch (err) {
      // A refusal is HTTP 200 with UNAVAILABLE, so reaching here means the
      // request itself failed. The stale record is dropped rather than shown as
      // if it were current.
      setRecord(null);
      setTransportError(err instanceof Error ? err.message : String(err));
    } finally {
      setLastAttemptAt(new Date().toISOString());
      setLoading(false);
    }
  }, [state]);

  useEffect(() => {
    void load();
    const timer = setInterval(() => {
      void load();
    }, intervalMs);
    return () => clearInterval(timer);
  }, [load, intervalMs]);

  const view = useMemo(() => liveRainfallView(record), [record]);
  const style = TONE_STYLES[view.tone];
  const Icon = style.Icon;
  const accent =
    view.tone === 'real' ? 'emerald' : view.tone === 'fallback' ? 'amber' : 'slate';

  return (
    <div
      className={cn(
        'rounded-lg border p-3 text-xs leading-relaxed text-slate-300',
        style.border,
        style.bg,
        className,
      )}
    >
      <div className="mb-2.5 flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-2">
          <Icon className={cn('mt-0.5 h-4 w-4 shrink-0', style.icon)} />
          <div className="min-w-0">
            <Eyebrow>Live rainfall monitor · {config.name} AOI</Eyebrow>
            <div className="mt-1">
              <span
                className={cn(
                  'font-mono font-semibold uppercase tracking-wide',
                  style.title,
                )}
              >
                {view.label}
              </span>
              <span className="text-slate-400"> · {view.headline}</span>
            </div>
          </div>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="inline-flex shrink-0 items-center gap-1.5 rounded border border-slate-700 bg-slate-900 px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-slate-300 transition hover:border-slate-600 hover:text-slate-100 disabled:opacity-50"
        >
          {loading ? (
            <LoaderCircle className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          Refresh
        </button>
      </div>

      {view.tone === 'fallback' && (
        <p className="mb-2.5 rounded border border-amber-700/70 bg-amber-950/50 p-2 font-semibold text-amber-200">
          FALLBACK — this value is {view.sourceDisplayName ?? 'a non-satellite source'} model
          precipitation, not a NASA IMERG satellite observation.
        </p>
      )}

      {view.tone === 'unavailable' && (
        <p className="mb-2.5 rounded border border-rose-900/70 bg-rose-950/50 p-2 text-rose-200">
          <span className="font-semibold">Latest rainfall unavailable.</span>{' '}
          {view.unavailableReason ?? 'The backend reported no machine-readable reason.'} No value
          is shown rather than substituting zero.
        </p>
      )}

      {transportError && (
        <p className="mb-2.5 rounded border border-rose-900/70 bg-rose-950/50 p-2 text-rose-200">
          <span className="font-semibold">Live rainfall request failed.</span>{' '}
          <span className="font-mono text-[10px]">{transportError}</span>
        </p>
      )}

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        <Readout
          label="latest interval"
          value={formatMm(view.latestMm)}
          hint={view.intervalLabel}
          accent={accent}
        />
        {view.accumulations.map((accum) => (
          <Readout
            key={accum.hours}
            label={`${accum.hours} h accumulation`}
            value={formatMm(accum.mm)}
            hint={accum.mm == null ? accum.unavailableReason : null}
            accent={accum.mm == null ? 'slate' : accent}
          />
        ))}
      </div>

      <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px]">
        <Field label="source" value={view.sourceDisplayName ?? 'not reported'} />
        <Field label="data_quality_status" value={view.dataQualityStatus ?? 'not reported'} />
        <Field label="observed_at_utc" value={view.observedAtUtc ?? 'not reported'} />
        <Field label="age" value={view.ageLabel ?? 'not reported'} />
        <Field label="freshness" value={view.freshnessLabel ?? 'not reported'} />
        {view.isStale === true ? <span className="text-amber-300">STALE</span> : null}
        <Field label="fetched_at_utc" value={view.fetchedAtUtc ?? 'not reported'} />
        <Field
          label="cache_hit"
          value={view.cacheHit == null ? 'not reported' : String(view.cacheHit)}
        />
      </div>

      <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
        This is the latest AVAILABLE observation, not a nowcast, and it is NOT the antecedent
        T-1..T-14 rainfall the prediction model consumes. One AOI-level request per refresh;
        auto-refresh every {Math.round(intervalMs / 60000)} min (backend cache TTL{' '}
        {Math.round(LIVE_RAINFALL_CACHE_TTL_SECONDS / 60)} min).
        {view.sourceDetail ? ` Product: ${view.sourceDetail}.` : ''}
        {lastAttemptAt ? ` Last request ${lastAttemptAt}.` : ''}
      </p>

      {view.attempts.length > 0 && (
        <details className="mt-2">
          <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider text-slate-400">
            Acquisition trail · {view.attempts.length} attempt
            {view.attempts.length === 1 ? '' : 's'}
          </summary>
          <ul className="mt-1.5 space-y-1 font-mono text-[10px] text-slate-400">
            {view.attempts.map((attempt, i) => (
              <li key={`${attempt.source_kind}-${i}`}>
                {attempt.source_kind} → {attempt.outcome}
                {attempt.detail ? ` · ${attempt.detail}` : ''}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
