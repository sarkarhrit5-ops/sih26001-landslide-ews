/**
 * Shared BhūRaksha UI primitives.
 *
 * The ProvenanceTag is the product's signature device: every data point on the
 * screen can carry an honest tag stating whether its underlying input is REAL,
 * a labelled DERIVED_PROXY, NOT_USED, or genuinely UNAVAILABLE. The vocabulary
 * mirrors the backend provenance artifact's `input_status` field exactly.
 */
import type { ReactNode } from 'react';
import { clsx } from 'clsx';
import type { ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

type ProvenanceKind = 'REAL' | 'DERIVED_PROXY' | 'NOT_USED' | 'UNAVAILABLE' | 'COMPUTED';

const PROVENANCE_STYLES: Record<ProvenanceKind, string> = {
  REAL: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  COMPUTED: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  DERIVED_PROXY: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  NOT_USED: 'border-slate-600/50 bg-slate-700/20 text-slate-400',
  UNAVAILABLE: 'border-red-500/40 bg-red-500/10 text-red-300',
};

/** Normalises any backend status string onto the display vocabulary. */
export function toProvenanceKind(raw: string | undefined): ProvenanceKind {
  const s = (raw || '').toUpperCase();
  if (s.includes('NOT_USED')) return 'NOT_USED';
  if (s.includes('DERIVED') || s.includes('PROXY')) return 'DERIVED_PROXY';
  if (
    s.includes('UNAVAILABLE') ||
    s.includes('MISSING') ||
    s.includes('INVALID') ||
    s.includes('NOT_AUTH')
  )
    return 'UNAVAILABLE';
  if (s.includes('COMPUTED') || s.includes('VALID')) return 'COMPUTED';
  return 'REAL';
}

export function ProvenanceTag({
  status,
  source,
  className,
}: {
  status: string | undefined;
  source?: string;
  className?: string;
}) {
  const kind = toProvenanceKind(status);
  const label = kind.replace('_', ' ');
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider',
        PROVENANCE_STYLES[kind],
        className,
      )}
      title={source}
    >
      <span
        className={cn(
          'h-1.5 w-1.5 rounded-full',
          kind === 'REAL' || kind === 'COMPUTED'
            ? 'bg-emerald-400'
            : kind === 'DERIVED_PROXY'
              ? 'bg-amber-400'
              : kind === 'UNAVAILABLE'
                ? 'bg-red-400'
                : 'bg-slate-500',
        )}
      />
      {label}
    </span>
  );
}

/** State validation status pill (emerald pilot vs. amber pending). */
export function StatusPill({
  tone,
  children,
  className,
}: {
  tone: 'pilot' | 'pending' | 'neutral';
  children: ReactNode;
  className?: string;
}) {
  const styles =
    tone === 'pilot'
      ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300'
      : tone === 'pending'
        ? 'border-amber-500/40 bg-amber-500/10 text-amber-300'
        : 'border-slate-600/50 bg-slate-700/20 text-slate-300';
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-semibold tracking-wide',
        styles,
        className,
      )}
    >
      <span
        className={cn(
          'h-1.5 w-1.5 rounded-full',
          tone === 'pilot'
            ? 'bg-emerald-400 shadow-[0_0_8px_1px_rgba(52,211,153,0.7)]'
            : tone === 'pending'
              ? 'bg-amber-400'
              : 'bg-slate-400',
        )}
      />
      {children}
    </span>
  );
}

/** A small uppercase monospace section/eyebrow label. */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        'font-mono text-[10px] font-semibold uppercase tracking-[0.28em] text-slate-500',
        className,
      )}
    >
      {children}
    </div>
  );
}

/** A labelled metric readout — mono value + caption. */
export function Metric({
  value,
  label,
  sub,
  accent = 'emerald',
  className,
}: {
  value: ReactNode;
  label: ReactNode;
  sub?: ReactNode;
  accent?: 'emerald' | 'amber' | 'red' | 'cyan' | 'slate';
  className?: string;
}) {
  const accentText =
    accent === 'emerald'
      ? 'text-emerald-300'
      : accent === 'amber'
        ? 'text-amber-300'
        : accent === 'red'
          ? 'text-red-300'
          : accent === 'cyan'
            ? 'text-cyan-300'
            : 'text-slate-100';
  return (
    <div className={cn('rounded-lg border border-slate-800 bg-slate-900/40 p-3', className)}>
      <div className={cn('font-mono text-2xl font-bold tabular-nums leading-none', accentText)}>
        {value}
      </div>
      <div className="mt-1.5 text-[11px] font-medium text-slate-300">{label}</div>
      {sub && <div className="mt-0.5 text-[10px] text-slate-500">{sub}</div>}
    </div>
  );
}
