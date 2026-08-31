/**
 * Rainfall provenance banner - shared by all four pilot consoles.
 *
 * This component reads the rainfall provenance reported by the backend and
 * presents the antecedent/model rainfall source used for the T-1...T-14 model
 * feature window. It does not change source selection or data semantics.
 */
import {
  CloudOff,
  CloudRain,
  Database,
  HelpCircle,
  type LucideIcon,
} from 'lucide-react';
import type { PilotMapRainfallView, RainfallProvenanceBlock } from '../../services/api';
import {
  formatCacheAge,
  formatObservationLag,
  rainfallProvenanceView,
} from '../pilot/pilotMapCells';
import type { RainfallProvenanceTone } from '../pilot/pilotMapCells';

interface RainfallProvenanceBannerProps {
  rainfall: PilotMapRainfallView | null | undefined;
  provenance?: RainfallProvenanceBlock | null;
  /** Optional extra line, e.g. the response's generated_from clause. */
  footnote?: string | null;
}

/** Per-tone presentation. Colour never contradicts the label. */
const TONE_STYLES: Record<
  RainfallProvenanceTone,
  { border: string; bg: string; icon: string; title: string; Icon: LucideIcon }
> = {
  real: {
    border: 'border-emerald-800/70',
    bg: 'bg-emerald-950/40',
    icon: 'text-emerald-400',
    title: 'text-emerald-200',
    Icon: CloudRain,
  },
  fallback: {
    border: 'border-sky-800/70',
    bg: 'bg-sky-950/35',
    icon: 'text-sky-400',
    title: 'text-sky-200',
    Icon: Database,
  },
  unavailable: {
    border: 'border-rose-900/70',
    bg: 'bg-rose-950/40',
    icon: 'text-rose-400',
    title: 'text-rose-200',
    Icon: CloudOff,
  },
  unreported: {
    border: 'border-slate-700',
    bg: 'bg-slate-900/60',
    icon: 'text-slate-400',
    title: 'text-slate-200',
    Icon: HelpCircle,
  },
};

function Field({ label, value }: { label: string; value: string }) {
  return (
    <span className="whitespace-nowrap">
      <span className="text-slate-500">{label} </span>
      <span className="text-slate-300">{value}</span>
    </span>
  );
}

export function RainfallProvenanceBanner({
  rainfall,
  provenance,
  footnote,
}: RainfallProvenanceBannerProps) {
  const view = rainfallProvenanceView(rainfall, provenance);
  const style = TONE_STYLES[view.tone];
  const Icon = style.Icon;
  const lag = formatObservationLag(view.observationLagDays);
  const cacheAge = formatCacheAge(view);
  const supportedFallback = view.tone === 'fallback';

  return (
    <div
      className={`flex items-start gap-2 rounded-lg border ${style.border} ${style.bg} p-3 text-xs leading-relaxed text-slate-300`}
    >
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${style.icon}`} />
      <div className="min-w-0 space-y-1.5">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
          {supportedFallback
            ? 'Rainfall source · ERA5 fallback'
            : 'Antecedent / model rainfall · T-1...T-14 window'}
        </div>
        <div>
          <span className={`font-mono font-semibold uppercase tracking-wide ${style.title}`}>
            {supportedFallback ? 'Open-Meteo ERA5' : view.label}
          </span>
          {view.source && !supportedFallback ? (
            <span className="text-slate-400"> · {view.source}</span>
          ) : null}
          {view.runType ? <span className="text-slate-500"> ({view.runType} run)</span> : null}
          {view.windowDays != null && !supportedFallback ? (
            <span className="text-slate-400"> · {view.windowDays}-day AOI-mean window</span>
          ) : null}
        </div>

        {supportedFallback ? (
          <>
            <div className="text-slate-300">Used because near-real-time IMERG was unavailable.</div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-slate-400">
              14-day antecedent rainfall · T-1...T-14
            </div>
          </>
        ) : null}

        <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px]">
          <Field label="observed for" value={view.observationDate ?? 'not reported'} />
          <Field label="requested" value={view.requestedDate ?? 'not reported'} />
          <Field label="fetched" value={view.fetchedAtUtc ?? 'not reported'} />
          <Field label="lag" value={lag ?? 'not reported'} />
          <Field label="freshness" value={cacheAge ?? 'not reported'} />
          {view.dataQualityStatus ? (
            <Field label="data_quality_status" value={view.dataQualityStatus} />
          ) : null}
        </div>

        {view.note && !supportedFallback ? <div className="text-slate-400">{view.note}</div> : null}

        {supportedFallback ? (
          <div className="text-sky-200/90">
            IMERG was unavailable; ERA5 reanalysis is being used for the model rainfall features.
          </div>
        ) : view.fallbackWarning ? (
          <div className="font-medium text-amber-300">{view.fallbackWarning}</div>
        ) : null}

        {!supportedFallback && view.caveats.length > 0 && (
          <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-amber-200/90">
            {view.caveats.map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        )}

        {footnote ? <div className="text-[10px] text-slate-500">{footnote}</div> : null}
      </div>
    </div>
  );
}
