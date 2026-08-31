/**
 * Rainfall provenance banner — shared by all four pilot consoles.
 *
 * This replaces the old per-dashboard block that asserted "Real IMERG antecedent
 * rainfall." on every successful prediction. That sentence was unconditional, so
 * an Open-Meteo ERA5 fallback series was presented to the operator as an official
 * live NASA IMERG observation. This component instead reads what the response
 * actually reports and labels it accordingly:
 *
 *   REAL / NASA IMERG              data_quality_status=REAL or source_kind=IMERG
 *   FALLBACK / Open-Meteo ERA5     is_fallback, FALLBACK status, or OPEN_METEO_FALLBACK
 *   UNAVAILABLE                    the backend obtained no rainfall at all
 *   UNREPORTED                     the producer supplied no provenance to judge
 *
 * There is no code path here that prints a REAL/IMERG claim from missing data,
 * and no value is defaulted: an unreported field renders as "not reported".
 */
import { AlertTriangle, CloudOff, CloudRain, HelpCircle } from 'lucide-react';
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
  { border: string; bg: string; icon: string; title: string; Icon: typeof CloudRain }
> = {
  real: {
    border: 'border-emerald-800/70',
    bg: 'bg-emerald-950/40',
    icon: 'text-emerald-400',
    title: 'text-emerald-200',
    Icon: CloudRain,
  },
  fallback: {
    border: 'border-amber-800/70',
    bg: 'bg-amber-950/40',
    icon: 'text-amber-400',
    title: 'text-amber-200',
    Icon: AlertTriangle,
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

  return (
    <div
      className={`flex items-start gap-2 rounded-lg border ${style.border} ${style.bg} p-3 text-xs leading-relaxed text-slate-300`}
    >
      <Icon className={`mt-0.5 h-4 w-4 shrink-0 ${style.icon}`} />
      <div className="min-w-0 space-y-1.5">
        <div>
          <span className={`font-mono font-semibold uppercase tracking-wide ${style.title}`}>
            {view.label}
          </span>
          {view.source ? <span className="text-slate-400"> · {view.source}</span> : null}
          {view.runType ? <span className="text-slate-500"> ({view.runType} run)</span> : null}
          {view.windowDays != null ? (
            <span className="text-slate-400"> · {view.windowDays}-day AOI-mean window</span>
          ) : null}
        </div>

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

        {view.note ? <div className="text-slate-400">{view.note}</div> : null}

        {view.fallbackWarning ? (
          <div className="font-medium text-amber-300">{view.fallbackWarning}</div>
        ) : null}

        {view.caveats.length > 0 && (
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
