/**
 * NER operations console — all 8 North Eastern Region states.
 *
 * Every dynamic value on this page comes live from GET /api/v1/validation/status.
 * Nothing is fabricated: when the backend is unreachable the page says so and
 * falls back only to static geography (which state is the pilot), never to
 * invented metrics.
 */
import { useEffect, useMemo, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  RefreshCw,
  LoaderCircle,
  Database,
  Cpu,
  Mountain,
  CloudRain,
  ShieldCheck,
  TriangleAlert,
  MapPin,
} from 'lucide-react';
import { BrandLockup } from '../components/brand/BrandMark';
import { Eyebrow, StatusPill, Metric, cn } from '../components/common/ui';
import { NERMap } from '../components/ner/NERMap';
import type { NerMapEntry } from '../components/ner/NERMap';
import { NER_STATES, validationTone } from '../data/nerStates';
import type { NerStateMeta } from '../data/nerStates';
import { apiService } from '../services/api';
import type { StateValidationReport } from '../services/api';

type LoadState = 'loading' | 'ready' | 'error';

/** Match a live report to a static state by id / name (case + spacing tolerant). */
function reportKey(r: StateValidationReport): string {
  const raw = r.state_id || r.id || r.state || r.state_name || '';
  return raw.trim().toLowerCase().replace(/\s+/g, '_');
}

function matchReport(
  meta: NerStateMeta,
  reports: StateValidationReport[],
): StateValidationReport | undefined {
  return reports.find((r) => {
    const k = reportKey(r);
    return k === meta.id || k === meta.name.toLowerCase().replace(/\s+/g, '_');
  });
}

/** Honest colour for a raw backend status string — defaults to neutral, never green. */
function fieldToneClass(value: string | undefined): string {
  const s = (value || '').toUpperCase();
  if (/READY|COMPLETE|AVAILABLE|PRESENT|VALID|TRAINED|PASS|REAL|OK\b/.test(s))
    return 'text-emerald-300';
  if (/MISSING|UNAVAILABLE|FAIL|ERROR|ABSENT|NONE|NOT[_ ]/.test(s)) return 'text-red-300';
  if (/PROXY|DERIVED|PARTIAL|PENDING|LIMITED|LOW/.test(s)) return 'text-amber-300';
  return 'text-slate-300';
}

/** A compact label:value readout for a backend status field. */
function FieldChip({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: string | undefined;
}) {
  return (
    <div className="flex items-center justify-between gap-2 rounded border border-slate-800 bg-slate-950/50 px-2 py-1.5">
      <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-500">
        <Icon className="h-3 w-3" />
        {label}
      </span>
      <span className={cn('font-mono text-[10px] font-semibold uppercase', fieldToneClass(value))}>
        {value && value.trim() ? value : '—'}
      </span>
    </div>
  );
}

interface StateCardProps {
  meta: NerStateMeta;
  report: StateValidationReport | undefined;
  selected: boolean;
  onSelect: () => void;
  onOpen?: () => void;
}

function StateCard({ meta, report, selected, onSelect, onOpen }: StateCardProps) {
  const tone = report ? validationTone(report.overall_status) : meta.isPilot ? 'pilot' : 'pending';
  const isPilot = tone === 'pilot';
  const events = report?.inventory_events;
  const usable = report?.usable_events;
  const blocking = report?.blocking_reasons ?? [];

  return (
    <div
      onClick={onSelect}
      className={cn(
        'cursor-pointer rounded-xl border p-4 transition',
        selected
          ? 'border-emerald-500/60 bg-emerald-500/[0.06] ring-1 ring-emerald-500/30'
          : isPilot
            ? 'border-emerald-500/30 bg-slate-900/50 hover:border-emerald-500/50'
            : 'border-slate-800 bg-slate-900/40 hover:border-slate-700',
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[15px] font-semibold text-slate-100">{meta.name}</div>
          <div className="mt-0.5 inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-slate-500">
            <MapPin className="h-3 w-3" />
            {meta.capital}
          </div>
        </div>
        <StatusPill tone={tone}>{isPilot ? 'Validated pilot' : 'Pending'}</StatusPill>
      </div>

      <div className="mt-3 flex items-center gap-4">
        <div>
          <div className="font-mono text-xl font-bold tabular-nums leading-none text-slate-100">
            {events ?? '—'}
          </div>
          <div className="mt-1 text-[10px] uppercase tracking-wider text-slate-500">
            catalog events
          </div>
        </div>
        {typeof usable === 'number' && usable !== events && (
          <div>
            <div className="font-mono text-xl font-bold tabular-nums leading-none text-slate-300">
              {usable}
            </div>
            <div className="mt-1 text-[10px] uppercase tracking-wider text-slate-500">usable</div>
          </div>
        )}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-1.5">
        <FieldChip icon={Mountain} label="DEM" value={report?.dem_status} />
        <FieldChip icon={Database} label="Exposure" value={report?.exposure_status} />
        <FieldChip icon={CloudRain} label="Rainfall" value={report?.rainfall_status} />
        <FieldChip icon={Cpu} label="Model" value={report?.model_status} />
      </div>

      {blocking.length > 0 && (
        <div className="mt-3 rounded-lg border border-amber-500/20 bg-amber-500/[0.05] p-2.5">
          <div className="inline-flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-amber-300">
            <TriangleAlert className="h-3 w-3" />
            Blocking reasons
          </div>
          <ul className="mt-1.5 space-y-1">
            {blocking.map((b, i) => (
              <li key={i} className="flex gap-1.5 text-[11px] leading-snug text-slate-300">
                <span className="text-amber-400/70">·</span>
                {b}
              </li>
            ))}
          </ul>
        </div>
      )}

      {onOpen && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onOpen();
          }}
          className={cn(
            'mt-3 inline-flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-[13px] font-semibold transition',
            isPilot
              ? 'bg-emerald-500 text-[#04120c] hover:bg-emerald-400'
              : 'border border-slate-700 bg-slate-900/60 text-slate-200 hover:border-emerald-500/50 hover:text-emerald-300',
          )}
        >
          Open {meta.name} console <ArrowRight className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}

interface NERDashboardProps {
  onBack: () => void;
  onOpenSikkim: () => void;
  onOpenAssam: () => void;
  onOpenArunachal: () => void;
}

export function NERDashboard({ onBack, onOpenSikkim, onOpenAssam, onOpenArunachal }: NERDashboardProps) {
  const [reports, setReports] = useState<StateValidationReport[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let alive = true;
    setLoadState('loading');
    apiService
      .getValidationStatus()
      .then((data) => {
        if (!alive) return;
        setReports(Array.isArray(data) ? data : []);
        setLoadState('ready');
      })
      .catch(() => alive && setLoadState('error'));
    return () => {
      alive = false;
    };
  }, [reloadKey]);

  // Join static geography to live reports (works even before/without data).
  const joined = useMemo(
    () =>
      NER_STATES.map((meta) => ({ meta, report: matchReport(meta, reports) })).sort((a, b) => {
        const ap = (a.report ? validationTone(a.report.overall_status) : a.meta.isPilot ? 'pilot' : 'pending') === 'pilot' ? 0 : 1;
        const bp = (b.report ? validationTone(b.report.overall_status) : b.meta.isPilot ? 'pilot' : 'pending') === 'pilot' ? 0 : 1;
        if (ap !== bp) return ap - bp;
        return (b.report?.inventory_events ?? 0) - (a.report?.inventory_events ?? 0);
      }),
    [reports],
  );

  const mapEntries: NerMapEntry[] = useMemo(
    () =>
      joined.map(({ meta, report }) => {
        const tone = report ? validationTone(report.overall_status) : meta.isPilot ? 'pilot' : 'pending';
        return { ...meta, tone, statusLabel: report?.overall_status ?? (meta.isPilot ? 'VALIDATED_PILOT' : 'PENDING') };
      }),
    [joined],
  );

  const validatedCount = mapEntries.filter((e) => e.tone === 'pilot').length;
  const pendingCount = mapEntries.length - validatedCount;
  const totalEvents = reports.reduce((sum, r) => sum + (r.inventory_events || 0), 0);

  return (
    <div className="min-h-screen bg-[#0a0d12] text-slate-100">
      {/* Top bar */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0a0d12]/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/60 px-2.5 py-1.5 text-[13px] font-medium text-slate-300 transition hover:border-slate-600 hover:text-slate-100"
            >
              <ArrowLeft className="h-4 w-4" /> Home
            </button>
            <BrandLockup size={30} showDevanagari={false} />
          </div>
          <div className="flex items-center gap-3">
            <span className="hidden items-center gap-2 font-mono text-[11px] text-slate-400 sm:inline-flex">
              {loadState === 'loading' ? (
                <>
                  <LoaderCircle className="h-3.5 w-3.5 animate-spin text-slate-500" /> loading status…
                </>
              ) : loadState === 'ready' ? (
                <>
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_1px_rgba(52,211,153,0.7)]" />
                  live · /validation/status
                </>
              ) : (
                <>
                  <span className="h-2 w-2 rounded-full bg-amber-400" /> backend unreachable
                </>
              )}
            </span>
            <button
              onClick={() => setReloadKey((k) => k + 1)}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/60 px-2.5 py-1.5 text-[13px] font-medium text-slate-300 transition hover:border-emerald-500/50 hover:text-emerald-300"
            >
              <RefreshCw className={cn('h-3.5 w-3.5', loadState === 'loading' && 'animate-spin')} />
              Refresh
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* Title + summary */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <Eyebrow>North Eastern Region · 8-state validation sweep</Eyebrow>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
              Operations console
            </h1>
          </div>
          <div className="grid grid-cols-3 gap-2.5">
            <Metric value={mapEntries.length} label="States" sub="monitored" accent="slate" />
            <Metric value={validatedCount} label="Validated" sub="pilot" accent="emerald" />
            <Metric value={pendingCount} label="Pending" sub="validation" accent="amber" />
          </div>
        </div>

        {loadState === 'error' && (
          <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-500/25 bg-amber-500/[0.05] p-4">
            <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
            <div className="text-[13px] leading-relaxed text-slate-300">
              <span className="font-semibold text-amber-300">Live validation status is unavailable.</span>{' '}
              The backend at <code className="font-mono text-slate-400">/api/v1/validation/status</code>{' '}
              could not be reached, so per-state metrics are not shown. The map below reflects static
              geography only — no figures are substituted. Start the backend and press Refresh.
            </div>
          </div>
        )}

        {/* Map + cards */}
        <div className="mt-6 grid gap-5 lg:grid-cols-12">
          <div className="lg:col-span-7">
            <div className="lg:sticky lg:top-20 h-[520px]">
              <NERMap
                entries={mapEntries}
                selectedStateId={selectedId}
                onSelectState={(id) => setSelectedId((cur) => (cur === id ? null : id))}
              />
              <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-900/40 px-4 py-2.5">
                <span className="inline-flex items-center gap-2 font-mono text-[11px] text-slate-400">
                  <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
                  {totalEvents > 0
                    ? `${totalEvents.toLocaleString()} real catalog events across NER`
                    : 'Catalog totals load with live status'}
                </span>
                {selectedId && (
                  <button
                    onClick={() => setSelectedId(null)}
                    className="font-mono text-[11px] text-slate-400 underline-offset-2 hover:text-emerald-300 hover:underline"
                  >
                    clear selection
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="lg:col-span-5">
            {loadState === 'loading' ? (
              <div className="flex h-64 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40">
                <span className="inline-flex items-center gap-2 font-mono text-[13px] text-slate-400">
                  <LoaderCircle className="h-4 w-4 animate-spin" /> Loading state validation…
                </span>
              </div>
            ) : (
              <div className="space-y-3">
                {joined.map(({ meta, report }) => (
                  <StateCard
                    key={meta.id}
                    meta={meta}
                    report={report}
                    selected={selectedId === meta.id}
                    onSelect={() => setSelectedId((cur) => (cur === meta.id ? null : meta.id))}
                    onOpen={
                      meta.id === 'sikkim'
                        ? onOpenSikkim
                        : meta.id === 'assam'
                          ? onOpenAssam
                          : meta.id === 'arunachal_pradesh'
                            ? onOpenArunachal
                            : undefined
                    }
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

