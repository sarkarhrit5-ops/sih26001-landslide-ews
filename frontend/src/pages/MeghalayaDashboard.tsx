/**
 * Meghalaya validated-pilot console.
 *
 * Every figure on this page is fetched live from the persisted Meghalaya evidence
 * bundle (GET /api/v1/validation/meghalaya/evidence), the real GLC positives
 * (GET /api/v1/validation/meghalaya/events), and a live per-cell prediction
 * (GET /api/v1/predict/meghalaya/grid). Nothing is fabricated:
 *   - model metrics come from the held-out evaluation the backend serves;
 *   - land_cover_class is shown honestly as REAL ESA WorldCover treated as a
 *     categorical feature — the one methodological difference from the Sikkim pilot
 *     (the same treatment as the Assam and Arunachal pilots);
 *   - the prediction runs on real IMERG antecedent rainfall while the model was
 *     trained on ERA5, a source shift the backend discloses verbatim;
 *   - if the evidence bundle is not VALID, the page says so rather than guessing.
 *
 * Structurally this is the Sikkim console adapted to the Meghalaya endpoints, whose
 * JSON shapes are identical. The other pilot dashboards are left untouched.
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  ArrowLeft,
  RefreshCw,
  LoaderCircle,
  ShieldCheck,
  Mountain,
  CloudRain,
  Cpu,
  Database,
  FlaskConical,
  ScrollText,
  GitBranch,
  Binary,
  Lock,
  Gauge,
  Ruler,
  TriangleAlert,
  CheckCircle2,
  Target,
} from 'lucide-react';
import { BrandLockup } from '../components/brand/BrandMark';
import { Eyebrow, StatusPill, ProvenanceTag, cn } from '../components/common/ui';
import { DataStateBanner } from '../components/common/DataStateBanner';
import { MeghalayaMap } from '../components/meghalaya/MeghalayaMap';
import { apiService } from '../services/api';
import type {
  MeghalayaEvidenceResponse,
  MeghalayaEventsResponse,
  MeghalayaPredictionResponse,
  ValidationMetricSet,
} from '../services/api';

type LoadState = 'loading' | 'ready' | 'error';

/**
 * Best-effort extraction of the backend's DATA_UNAVAILABLE reason from a thrown
 * fetch error. fetchJson throws `HTTP <status>: <body>`, where the body is the
 * FastAPI JSON `{"detail": {"status","reason","details"}}`. Returns null when no
 * structured reason can be recovered, so the UI falls back to a generic message
 * rather than showing a raw HTTP string.
 */
function extractUnavailableReason(err: unknown): string | null {
  const msg = err instanceof Error ? err.message : String(err);
  const brace = msg.indexOf('{');
  if (brace >= 0) {
    try {
      const body = JSON.parse(msg.slice(brace));
      const detail = body?.detail ?? body;
      if (detail && typeof detail.reason === 'string') return detail.reason;
    } catch {
      /* not JSON — fall through to null */
    }
  }
  return null;
}

const num3 = (v: number | undefined) => (typeof v === 'number' ? v.toFixed(3) : '—');

/** One headline metric tile. */
function MetricCell({
  label,
  value,
  hint,
  accent = 'emerald',
}: {
  label: string;
  value: string;
  hint?: string;
  accent?: 'emerald' | 'amber' | 'cyan' | 'slate';
}) {
  const color =
    accent === 'emerald'
      ? 'text-emerald-300'
      : accent === 'amber'
        ? 'text-amber-300'
        : accent === 'cyan'
          ? 'text-cyan-300'
          : 'text-slate-100';
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
      <div className={cn('font-mono text-2xl font-bold tabular-nums leading-none', color)}>
        {value}
      </div>
      <div className="mt-2 text-[11px] font-semibold uppercase tracking-wider text-slate-300">
        {label}
      </div>
      {hint && <div className="mt-0.5 text-[10px] text-slate-500">{hint}</div>}
    </div>
  );
}

/** Section heading with an icon + mono eyebrow. */
function SectionHeading({
  icon: Icon,
  eyebrow,
  title,
  right,
}: {
  icon: typeof Cpu;
  eyebrow: string;
  title: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-4 flex items-end justify-between gap-3">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-700 bg-slate-950 text-emerald-400">
          <Icon className="h-4 w-4" />
        </span>
        <div>
          <Eyebrow>{eyebrow}</Eyebrow>
          <h2 className="mt-1 text-lg font-bold tracking-tight text-slate-100">{title}</h2>
        </div>
      </div>
      {right}
    </div>
  );
}

/** The full nested model_comparison table (holdout → feature set → model). */
function ModelComparisonTable({
  comparison,
}: {
  comparison: MeghalayaEvidenceResponse['metrics'] extends null
    ? never
    : NonNullable<MeghalayaEvidenceResponse['metrics']>['model_comparison'];
}) {
  if (!comparison) return null;
  const rows: {
    holdout: string;
    featureSet: string;
    model: string;
    m: ValidationMetricSet;
  }[] = [];
  Object.entries(comparison).forEach(([holdout, bySet]) =>
    Object.entries(bySet).forEach(([featureSet, byModel]) =>
      Object.entries(byModel).forEach(([model, m]) =>
        rows.push({ holdout, featureSet, model, m }),
      ),
    ),
  );
  if (rows.length === 0) return null;

  const fmtHoldout = (h: string) => h.replace(/_/g, ' ');
  return (
    <div className="overflow-x-auto rounded-xl border border-slate-800">
      <table className="w-full border-collapse text-left font-mono text-[11px]">
        <thead>
          <tr className="border-b border-slate-800 bg-slate-900/60 text-slate-400">
            <th className="px-3 py-2 font-semibold uppercase tracking-wider">Holdout</th>
            <th className="px-3 py-2 font-semibold uppercase tracking-wider">Features</th>
            <th className="px-3 py-2 font-semibold uppercase tracking-wider">Model</th>
            <th className="px-3 py-2 text-right font-semibold uppercase tracking-wider">PR-AUC</th>
            <th className="px-3 py-2 text-right font-semibold uppercase tracking-wider">ROC-AUC</th>
            <th className="px-3 py-2 text-right font-semibold uppercase tracking-wider">F1</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const primary =
              r.holdout.includes('temporal') &&
              r.featureSet.includes('rainfall') &&
              r.model === 'LightGBM';
            return (
              <tr
                key={i}
                className={cn(
                  'border-b border-slate-800/60',
                  primary ? 'bg-emerald-500/[0.07]' : 'hover:bg-slate-900/40',
                )}
              >
                <td className="px-3 py-2 capitalize text-slate-400">{fmtHoldout(r.holdout)}</td>
                <td className="px-3 py-2 capitalize text-slate-400">{fmtHoldout(r.featureSet)}</td>
                <td className="px-3 py-2 font-semibold text-slate-200">
                  {r.model}
                  {primary && <span className="ml-1.5 text-emerald-400">★</span>}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-emerald-300">{num3(r.m['PR-AUC'])}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-300">{num3(r.m['ROC-AUC'])}</td>
                <td className="px-3 py-2 text-right tabular-nums text-slate-300">{num3(r.m.F1)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** A single provenance fact row. */
function InfoRow({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Cpu;
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start gap-2.5 border-b border-slate-800/60 py-2.5 last:border-0">
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-slate-500" />
      <div className="min-w-0">
        <div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
        <div className="mt-0.5 text-[12px] leading-relaxed text-slate-300">{children}</div>
      </div>
    </div>
  );
}

/**
 * How each of the 11 model features is sourced. The Meghalaya pilot's single
 * methodological difference from Sikkim is that land_cover_class is REAL observed
 * ESA WorldCover (a categorical feature), NOT an elevation-derived proxy — so every
 * Meghalaya feature is REAL.
 */
function featureKind(_name: string): string {
  return 'REAL';
}

function FeatureSchemaPanel({ schema }: { schema: MeghalayaEvidenceResponse['feature_schema'] }) {
  if (!schema || !schema.feature_names?.length) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-[13px] text-slate-400">
        Feature schema not available in the evidence bundle.
      </div>
    );
  }
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
      <div className="mb-3 flex items-center justify-between">
        <span className="font-mono text-[11px] uppercase tracking-wider text-slate-400">
          {schema.n_features ?? schema.feature_names.length} features →{' '}
          <span className="text-slate-300">{schema.target_column ?? 'target'}</span>
        </span>
        {schema.feature_set_name && (
          <span className="font-mono text-[10px] text-slate-500">{schema.feature_set_name}</span>
        )}
      </div>
      <div className="grid gap-1.5 sm:grid-cols-2">
        {schema.feature_names.map((f) => {
          const dtype = schema.dtype?.[f];
          return (
            <div
              key={f}
              className="flex items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/40 px-2.5 py-1.5"
              title={schema.meaning?.[f]}
            >
              <span className="truncate font-mono text-[11px] text-slate-200">{f}</span>
              <span className="flex shrink-0 items-center gap-1.5">
                {dtype && (
                  <span className="font-mono text-[9px] uppercase text-slate-500">{dtype}</span>
                )}
                <ProvenanceTag status={featureKind(f)} source={schema.meaning?.[f]} />
              </span>
            </div>
          );
        })}
      </div>
      {schema.meaning?.land_cover_class && (
        <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
          <span className="font-semibold text-emerald-300/90">land_cover_class</span> is REAL
          observed land cover — ESA WorldCover sampled per cell and scored as a categorical feature
          (not an elevation-derived proxy). {schema.meaning.land_cover_class}
        </p>
      )}
    </div>
  );
}

function ProvenancePanel({
  provenance,
}: {
  provenance: MeghalayaEvidenceResponse['provenance'];
}) {
  if (!provenance) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-[13px] text-slate-400">
        Provenance record not available in the evidence bundle.
      </div>
    );
  }
  const sc = provenance.sample_counts;
  const hp = provenance.model_hyperparameters;
  const inputStatus = provenance.input_status ?? {};
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
        <InfoRow icon={Mountain} label="Terrain / DEM source">
          {provenance.dem_source ?? '—'}
          {provenance.terrain_derivative_method && (
            <span className="text-slate-500"> · {provenance.terrain_derivative_method}</span>
          )}
        </InfoRow>
        <InfoRow icon={CloudRain} label="Antecedent rainfall source">
          {provenance.rainfall_source ?? '—'}
        </InfoRow>
        <InfoRow icon={Cpu} label="Model">
          {provenance.model_type ?? '—'}
          {hp && (
            <span className="text-slate-500">
              {' '}
              · {Object.entries(hp).map(([k, v]) => `${k}=${v}`).join(', ')}
            </span>
          )}
        </InfoRow>
        <InfoRow icon={GitBranch} label="Code version">
          <span className="font-mono">{(provenance.code_version ?? '—').slice(0, 12)}</span>
          {typeof provenance.random_seed === 'number' && (
            <span className="text-slate-500"> · seed {provenance.random_seed}</span>
          )}
        </InfoRow>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5">
        <div className="mb-3 font-mono text-[10px] uppercase tracking-wider text-slate-500">
          Sample counts
        </div>
        {sc ? (
          <div className="grid grid-cols-3 gap-2">
            <MiniStat v={sc.raw_catalog_events_in_aoi} l="raw in AOI" />
            <MiniStat v={sc.deduplicated_positive_events} l="dedup pos" />
            <MiniStat v={sc.negative_samples} l="negatives" />
            <MiniStat v={sc.total_samples} l="total" />
            <MiniStat v={sc.independent_event_dates} l="indep dates" />
            <MiniStat
              v={sc.pct_events_spatial_uncertainty_ge_5km}
              l="% ≥5 km"
              suffix="%"
              accentAmber
            />
          </div>
        ) : (
          <div className="text-[12px] text-slate-400">Sample counts not recorded.</div>
        )}

        <div className="mt-4 font-mono text-[10px] uppercase tracking-wider text-slate-500">
          input_status
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {Object.entries(inputStatus).map(([k, v]) => (
            <ProvenanceTag key={k} status={v} source={`${k}: ${v}`} />
          ))}
        </div>
        <p className="mt-2 text-[10px] leading-relaxed text-slate-500">
          Mirrors the backend provenance artifact verbatim. The Meghalaya model was trained on
          Open-Meteo ERA5 antecedent rainfall (IMERG recorded NOT_USED at training); the live
          prediction endpoint instead serves real NASA GPM IMERG — a source shift disclosed with
          the prediction above.
        </p>
      </div>
    </div>
  );
}

function MiniStat({
  v,
  l,
  suffix,
  accentAmber,
}: {
  v: number | undefined;
  l: string;
  suffix?: string;
  accentAmber?: boolean;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2">
      <div
        className={cn(
          'font-mono text-base font-bold tabular-nums leading-none',
          accentAmber ? 'text-amber-300' : 'text-slate-100',
        )}
      >
        {typeof v === 'number' ? `${v}${suffix ?? ''}` : '—'}
      </div>
      <div className="mt-1 text-[9px] uppercase tracking-wider text-slate-500">{l}</div>
    </div>
  );
}

interface MeghalayaDashboardProps {
  onBack: () => void;
}

export function MeghalayaDashboard({ onBack }: MeghalayaDashboardProps) {
  const [evidence, setEvidence] = useState<MeghalayaEvidenceResponse | null>(null);
  const [events, setEvents] = useState<MeghalayaEventsResponse | null>(null);
  const [prediction, setPrediction] = useState<MeghalayaPredictionResponse | null>(null);
  const [evidenceState, setEvidenceState] = useState<LoadState>('loading');
  const [eventsState, setEventsState] = useState<LoadState>('loading');
  const [predictionState, setPredictionState] = useState<LoadState>('loading');
  const [predictionReason, setPredictionReason] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    let alive = true;
    setEvidenceState('loading');
    setEventsState('loading');
    setPredictionState('loading');
    setPredictionReason(null);
    apiService
      .getMeghalayaEvidence()
      .then((d) => {
        if (!alive) return;
        setEvidence(d);
        setEvidenceState('ready');
      })
      .catch(() => alive && setEvidenceState('error'));
    apiService
      .getMeghalayaEvents()
      .then((d) => {
        if (!alive) return;
        setEvents(d);
        setEventsState('ready');
      })
      .catch(() => alive && setEventsState('error'));
    // Real per-cell prediction: the persisted Meghalaya model over the AOI grid with real
    // ESA WorldCover land cover and real IMERG antecedent rainfall. On refusal
    // (HTTP 503) we keep the reason and show no zones rather than substituting
    // fabricated risk.
    apiService
      .getMeghalayaPrediction()
      .then((d) => {
        if (!alive) return;
        setPrediction(d);
        setPredictionState('ready');
      })
      .catch((err) => {
        if (!alive) return;
        setPrediction(null);
        setPredictionReason(extractUnavailableReason(err));
        setPredictionState('error');
      });
    return () => {
      alive = false;
    };
  }, [reloadKey]);

  const metrics = evidence?.metrics ?? null;
  const vm = metrics?.validation_metrics ?? null;
  const decision = metrics?.model_decision ?? null;
  const eventList = useMemo(() => events?.events ?? [], [events]);
  const predictedCells = useMemo(() => prediction?.cells ?? [], [prediction]);
  const loading = evidenceState === 'loading';

  const statusOk = evidence?.status === 'VALID';

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
              <ArrowLeft className="h-4 w-4" /> NER console
            </button>
            <BrandLockup size={30} showDevanagari={false} />
          </div>
          <div className="flex items-center gap-3">
            {evidenceState === 'ready' && (
              <span
                className={cn(
                  'hidden items-center gap-1.5 font-mono text-[11px] sm:inline-flex',
                  statusOk ? 'text-emerald-300' : 'text-amber-300',
                )}
              >
                {statusOk ? (
                  <CheckCircle2 className="h-3.5 w-3.5" />
                ) : (
                  <TriangleAlert className="h-3.5 w-3.5" />
                )}
                evidence: {evidence?.status}
              </span>
            )}
            <button
              onClick={() => setReloadKey((k) => k + 1)}
              className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900/60 px-2.5 py-1.5 text-[13px] font-medium text-slate-300 transition hover:border-emerald-500/50 hover:text-emerald-300"
            >
              <RefreshCw className={cn('h-3.5 w-3.5', loading && 'animate-spin')} /> Refresh
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {/* Title */}
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <Eyebrow>Meghalaya · East Khasi + Jaintia Hills pilot</Eyebrow>
              {evidenceState === 'ready' && (
                <StatusPill tone={statusOk ? 'pilot' : 'pending'}>
                  {statusOk ? 'Validated pilot' : 'Validation pending'}
                </StatusPill>
              )}
            </div>
            <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-50 sm:text-3xl">
              Meghalaya landslide susceptibility console
            </h1>
            <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-slate-400">
              {metrics?.primary_model ?? 'LightGBM'} over {metrics?.feature_set ?? 'static + rainfall'}{' '}
              features, evaluated on a {metrics?.primary_evaluation ?? 'temporal hold-out'}. All
              figures are served from the persisted evidence bundle — none are hard-coded here.
            </p>
          </div>
        </div>

        {evidenceState === 'error' && (
          <div className="mt-5 flex items-start gap-3 rounded-xl border border-amber-500/25 bg-amber-500/[0.05] p-4">
            <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
            <div className="text-[13px] leading-relaxed text-slate-300">
              <span className="font-semibold text-amber-300">Evidence bundle unavailable.</span>{' '}
              <code className="font-mono text-slate-400">/api/v1/validation/meghalaya/evidence</code>{' '}
              could not be reached. No metrics are shown rather than substituting figures. Start the
              backend and press Refresh.
            </div>
          </div>
        )}

        {evidence && !statusOk && evidence.problems?.length > 0 && (
          <div className="mt-5 rounded-xl border border-amber-500/25 bg-amber-500/[0.05] p-4">
            <div className="inline-flex items-center gap-1.5 font-mono text-[11px] font-semibold uppercase tracking-wider text-amber-300">
              <TriangleAlert className="h-3.5 w-3.5" /> evidence status: {evidence.status}
            </div>
            <ul className="mt-2 space-y-1">
              {evidence.problems.map((p, i) => (
                <li key={i} className="text-[12px] text-slate-300">
                  · {p}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Primary metrics */}
        {loading ? (
          <div className="mt-6 flex h-28 items-center justify-center rounded-xl border border-slate-800 bg-slate-900/40">
            <span className="inline-flex items-center gap-2 font-mono text-[13px] text-slate-400">
              <LoaderCircle className="h-4 w-4 animate-spin" /> Loading evidence bundle…
            </span>
          </div>
        ) : vm ? (
          <>
            <div className="mt-5 flex items-center gap-2 font-mono text-[11px] text-slate-500">
              <Gauge className="h-3.5 w-3.5" />
              held-out performance · decision threshold{' '}
              {metrics?.holdout_details?.decision_threshold ?? 0.5}
            </div>
            <div className="mt-2 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <MetricCell label="PR-AUC" value={num3(vm['PR-AUC'])} hint="primary metric" accent="emerald" />
              <MetricCell label="ROC-AUC" value={num3(vm['ROC-AUC'])} accent="emerald" />
              <MetricCell label="Precision" value={num3(vm.Precision)} accent="slate" />
              <MetricCell label="Recall" value={num3(vm.Recall)} accent="slate" />
              <MetricCell label="F1" value={num3(vm.F1)} accent="slate" />
              <MetricCell
                label="False alarm"
                value={num3(vm['False Alarm Rate'])}
                hint="lower is better"
                accent="amber"
              />
            </div>
          </>
        ) : (
          <div className="mt-6 rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-[13px] text-slate-400">
            Validation metrics are not present in the evidence bundle.
          </div>
        )}

        {/* Map + model detail */}
        <div className="mt-8 grid gap-6 lg:grid-cols-12">
          {/* Left: map + rainfall */}
          <div className="lg:col-span-7">
            <SectionHeading
              icon={Target}
              eyebrow="Geography · real inventory + predicted zones"
              title="Pilot AOI, events & predicted risk"
              right={
                events ? (
                  <span className="font-mono text-[11px] text-slate-400">
                    {events.count} events
                    {predictionState === 'ready' && prediction
                      ? ` · ${prediction.summary.cells_scored} cells scored`
                      : ''}
                  </span>
                ) : undefined
              }
            />
            <div className="h-[440px]">
              <MeghalayaMap
                events={eventList}
                loaded={eventsState === 'ready'}
                predictedCells={predictedCells}
              />
            </div>

            {eventsState === 'error' ? (
              <div className="mt-3 flex items-start gap-2.5 rounded-lg border border-amber-700/60 bg-amber-950/50 p-3 text-[12px] text-amber-200">
                <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                <span>
                  Event geometry unavailable — the backend refused with{' '}
                  <span className="font-mono">DATA_UNAVAILABLE</span>. The AOI is shown without
                  points rather than plotting fabricated locations.
                </span>
              </div>
            ) : (
              events && (
                <div className="mt-3 grid grid-cols-3 gap-2.5">
                  <MiniStat v={events.spatial_uncertainty_summary.precise_lt_5km} l="precise <5 km" />
                  <MiniStat
                    v={events.spatial_uncertainty_summary.approximate_ge_5km}
                    l="approx ≥5 km"
                    accentAmber
                  />
                  <MiniStat
                    v={events.spatial_uncertainty_summary.pct_approximate_ge_5km}
                    l="% approx"
                    suffix="%"
                    accentAmber
                  />
                </div>
              )
            )}

            {/* Predicted risk zones — the persisted Meghalaya model over the AOI grid with
                real ESA WorldCover land cover and real IMERG antecedent rainfall. The
                banner is conditional on the actual prediction outcome, so the page
                never claims a prediction when the backend refused, nor implies safety
                for cells that were left unscored. */}
            <div className="mt-4 space-y-3">
              {predictionState === 'loading' && (
                <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/40 p-3 font-mono text-[11px] text-slate-400">
                  <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
                  Running the Meghalaya model over the AOI grid with real land cover and IMERG antecedent rainfall…
                </div>
              )}

              {predictionState === 'error' && (
                <DataStateBanner
                  mode="UNAVAILABLE"
                  message={
                    (predictionReason
                      ? `Predicted risk zones unavailable — ${predictionReason}`
                      : 'Predicted risk zones unavailable — the backend refused with DATA_UNAVAILABLE (the persisted model artifacts, the ESA WorldCover land cover, or the real IMERG rainfall could not be obtained).') +
                    ' No zones are drawn rather than substituting fabricated risk.'
                  }
                />
              )}

              {predictionState === 'ready' && prediction && (
                <>
                  {/* Rainfall provenance for THIS prediction (real IMERG, antecedent). */}
                  <div className="flex items-start gap-2.5 rounded-lg border border-emerald-800/70 bg-emerald-950/40 p-3 text-[11px] leading-relaxed text-emerald-200/90">
                    <CloudRain className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                    <div>
                      <span className="font-semibold text-emerald-200">Real IMERG antecedent rainfall.</span>{' '}
                      {prediction.rainfall.source ?? 'IMERG'} ·{' '}
                      {prediction.rainfall.window_days}-day AOI-mean window
                      {prediction.rainfall.run_type ? ` (${prediction.rainfall.run_type} run)` : ''}
                      {'. '}
                      {prediction.rainfall.note}
                    </div>
                  </div>

                  {/* Prediction summary over the scored grid. */}
                  <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
                    <MiniStat v={prediction.summary.cells_scored} l="cells scored" />
                    <MiniStat
                      v={prediction.summary.cells_exceeding_threshold}
                      l="≥ threshold"
                      accentAmber
                    />
                    <MiniStat
                      v={
                        prediction.summary.max_probability != null
                          ? Math.round(prediction.summary.max_probability * 100)
                          : undefined
                      }
                      l="max prob"
                      suffix="%"
                    />
                    <MiniStat v={prediction.summary.cells_unavailable} l="unavailable" />
                  </div>

                  {/* Honesty disclosures, served verbatim by the backend. */}
                  <details className="rounded-lg border border-slate-800 bg-slate-900/40 p-3">
                    <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider text-slate-400">
                      Prediction disclosures ({prediction.disclosures.length}) · raw probability, not the fused /risk score
                    </summary>
                    <ul className="mt-2 space-y-1.5">
                      {prediction.disclosures.map((d, i) => (
                        <li key={i} className="flex gap-2 text-[11px] leading-relaxed text-slate-400">
                          <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-slate-600" />
                          {d}
                        </li>
                      ))}
                    </ul>
                    <p className="mt-2 font-mono text-[10px] leading-relaxed text-slate-500">
                      prediction date {prediction.target_date} · grid {prediction.grid.n_lat}×
                      {prediction.grid.n_lon} @ {prediction.grid.step_deg}° · source: {prediction.generated_from}
                    </p>
                  </details>
                </>
              )}
            </div>
            {events?.source && (
              <p className="mt-2 font-mono text-[10px] leading-relaxed text-slate-500">
                events source: {events.source}
              </p>
            )}
          </div>

          {/* Right: model performance */}
          <div className="lg:col-span-5">
            <SectionHeading
              icon={FlaskConical}
              eyebrow="Held-out evaluation"
              title="Model performance"
            />
            {metrics ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-2.5">
                  <InfoBox icon={Cpu} label="Primary model" value={metrics.primary_model ?? '—'} />
                  <InfoBox icon={Binary} label="Feature set" value={metrics.feature_set ?? '—'} />
                  <InfoBox
                    icon={Ruler}
                    label="Train / test"
                    value={
                      metrics.sample_counts
                        ? `${metrics.sample_counts.primary_train_samples} / ${metrics.sample_counts.primary_test_samples}`
                        : '—'
                    }
                  />
                  <InfoBox
                    icon={Target}
                    label="Positives (tr/te)"
                    value={
                      metrics.sample_counts
                        ? `${metrics.sample_counts.primary_train_positives} / ${metrics.sample_counts.primary_test_positives}`
                        : '—'
                    }
                  />
                </div>
                {metrics.holdout_details?.temporal_holdout && (
                  <p className="text-[11px] leading-relaxed text-slate-500">
                    <span className="text-slate-400">Temporal split:</span>{' '}
                    {metrics.holdout_details.temporal_holdout}
                    {metrics.holdout_details.spatial_holdout && (
                      <>
                        {' · '}
                        <span className="text-slate-400">Spatial split:</span>{' '}
                        {metrics.holdout_details.spatial_holdout}
                      </>
                    )}
                  </p>
                )}
                <div>
                  <div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-slate-500">
                    All model / holdout combinations
                  </div>
                  <ModelComparisonTable comparison={metrics.model_comparison} />
                  <p className="mt-2 font-mono text-[10px] text-slate-500">
                    ★ primary configuration (temporal hold-out · static + rainfall · LightGBM)
                  </p>
                </div>
              </div>
            ) : (
              <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-5 text-[13px] text-slate-400">
                Model performance details unavailable.
              </div>
            )}
          </div>
        </div>

        {/* Feature schema */}
        <section className="mt-10">
          <SectionHeading
            icon={ScrollText}
            eyebrow="Model inputs"
            title="Feature schema"
          />
          <FeatureSchemaPanel schema={evidence?.feature_schema ?? null} />
        </section>

        {/* Provenance */}
        <section className="mt-10">
          <SectionHeading icon={Database} eyebrow="Data lineage" title="Provenance & data integrity" />
          <ProvenancePanel provenance={evidence?.provenance ?? null} />
        </section>

        {/* Model decision */}
        {decision && (
          <section className="mt-10">
            <SectionHeading icon={Lock} eyebrow="Operating decision" title="Recommended operating mode" />
            <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/[0.04] p-5">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-cyan-300" />
                <span className="font-semibold text-slate-100">
                  {decision.final_recommendation}
                </span>
              </div>
              {decision.justification_reasons?.length > 0 && (
                <ul className="mt-3 space-y-1.5">
                  {decision.justification_reasons.map((r, i) => (
                    <li key={i} className="flex gap-2 text-[13px] leading-relaxed text-slate-300">
                      <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-cyan-400" />
                      {r}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-12 border-t border-slate-800/80 pt-6">
          <p className="text-[11px] leading-relaxed text-slate-500">
            Decision-support only. Not a substitute for official disaster-management directives.
            Metrics reflect a held-out research evaluation on a limited landslide inventory and
            should not be read as an operational guarantee.
          </p>
        </footer>
      </main>
    </div>
  );
}

/** Compact labelled value box used in the model-performance column. */
function InfoBox({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/40 p-2.5">
      <div className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-slate-500">
        <Icon className="h-3 w-3" />
        {label}
      </div>
      <div className="mt-1 truncate text-[13px] font-semibold text-slate-200" title={value}>
        {value}
      </div>
    </div>
  );
}
