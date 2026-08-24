/**
 * BhūRaksha welcome page.
 *
 * Signature: a topographic-contour hero (terrain is the model's primary real
 * input) and the pervasive provenance-tag ethos. Every headline claim here is a
 * stable provenance fact; live model figures live in the Sikkim console.
 */
import { useEffect, useState } from 'react';
import {
  ArrowRight,
  Mountain,
  Database,
  CloudRain,
  Cpu,
  ShieldCheck,
  SatelliteDish,
  Waves,
  Ruler,
} from 'lucide-react';
import { BrandLockup, BrandMark } from '../components/brand/BrandMark';
import { Eyebrow, ProvenanceTag, StatusPill, cn } from '../components/common/ui';
import { apiService } from '../services/api';

/** Nested, scaled contour rings that read as a topographic summit. */
function ContourField() {
  const blob =
    'M100 12 C142 16 188 52 184 100 C180 144 142 190 100 186 C58 190 14 146 18 100 C12 58 58 8 100 12 Z';
  const scales = [1, 0.88, 0.76, 0.64, 0.52, 0.41, 0.31, 0.22, 0.14];
  const cx = 100;
  return (
    <svg
      viewBox="0 0 200 200"
      preserveAspectRatio="xMidYMid slice"
      className="h-full w-full"
      aria-hidden="true"
    >
      <g fill="none" strokeWidth="0.5">
        {scales.map((s, i) => (
          <path
            key={i}
            d={blob}
            transform={`translate(${cx * (1 - s)} ${cx * (1 - s)}) scale(${s})`}
            stroke={i <= 1 ? '#22d3ee' : '#10b981'}
            strokeOpacity={0.1 + i * 0.045}
          />
        ))}
        {/* a lower, secondary ridge, offset */}
        {[0.62, 0.5, 0.38, 0.26].map((s, i) => (
          <path
            key={`b${i}`}
            d={blob}
            transform={`translate(${60 + cx * (1 - s)} ${70 + cx * (1 - s)}) scale(${s})`}
            stroke="#10b981"
            strokeOpacity={0.08 + i * 0.03}
          />
        ))}
      </g>
    </svg>
  );
}

type Health = 'checking' | 'online' | 'offline';

function HealthDot({ status }: { status: Health }) {
  const map = {
    checking: { c: 'bg-slate-500', t: 'Checking backend…' },
    online: { c: 'bg-emerald-400 shadow-[0_0_8px_1px_rgba(52,211,153,0.7)]', t: 'Backend online' },
    offline: { c: 'bg-amber-400', t: 'Backend offline' },
  }[status];
  return (
    <span className="inline-flex items-center gap-2 font-mono text-[11px] text-slate-400">
      <span className={cn('h-2 w-2 rounded-full', map.c)} />
      {map.t}
    </span>
  );
}

interface WelcomePageProps {
  onEnterConsole: () => void;
  onOpenSikkim: () => void;
}

export function WelcomePage({ onEnterConsole, onOpenSikkim }: WelcomePageProps) {
  const [health, setHealth] = useState<Health>('checking');

  useEffect(() => {
    let alive = true;
    apiService
      .checkHealth()
      .then(() => alive && setHealth('online'))
      .catch(() => alive && setHealth('offline'));
    return () => {
      alive = false;
    };
  }, []);

  const method = [
    {
      icon: Mountain,
      title: 'Terrain',
      body: 'Copernicus GLO-30 DEM → slope, aspect, roughness, TPI.',
      status: 'REAL',
      source: 'Copernicus GLO-30 (30 m), tiles N27E088 + N28E088',
    },
    {
      icon: Database,
      title: 'Landslide inventory',
      body: '82 cataloged events inside the East Sikkim pilot AOI.',
      status: 'REAL',
      source: 'NASA Global Landslide Catalog, AOI-filtered & de-duplicated',
    },
    {
      icon: CloudRain,
      title: 'Antecedent rainfall',
      body: 'Open-Meteo ERA5, strictly T-14…T-1. No future leakage.',
      status: 'REAL',
      source: 'Open-Meteo ERA5 archive; no zero-fill or synthetic substitution',
    },
    {
      icon: Cpu,
      title: 'Model',
      body: 'LightGBM over 11 features, evaluated on a temporal hold-out.',
      status: 'COMPUTED',
      source: 'Metrics computed on held-out events (train ≤2014, test ≥2015)',
    },
  ];

  return (
    <div className="min-h-screen bg-[#0a0d12] text-slate-100">
      {/* Top bar */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0a0d12]/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3.5">
          <BrandLockup />
          <div className="flex items-center gap-5">
            <HealthDot status={health} />
            <button
              onClick={onEnterConsole}
              className="hidden items-center gap-1.5 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3.5 py-2 text-sm font-semibold text-emerald-300 transition hover:bg-emerald-500/20 sm:inline-flex"
            >
              Open console <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden border-b border-slate-800/80">
        <div className="topo-contour-bg absolute inset-0" />
        <div className="pointer-events-none absolute right-[-6%] top-1/2 h-[130%] w-[55%] -translate-y-1/2 opacity-70">
          <ContourField />
        </div>
        <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent brk-scan-line" />

        <div className="relative mx-auto max-w-7xl px-6 py-20 sm:py-28">
          <div className="max-w-2xl">
            <div className="brk-rise" style={{ animationDelay: '0.05s' }}>
              <Eyebrow>Operational Geointelligence · North Eastern Region · SIH 2026</Eyebrow>
            </div>
            <h1
              className="brk-rise mt-5 text-4xl font-bold leading-[1.08] tracking-tight text-slate-50 sm:text-6xl"
              style={{ animationDelay: '0.12s' }}
            >
              Landslide early warning for the{' '}
              <span className="text-emerald-400">Eastern Himalaya.</span>
            </h1>
            <p
              className="brk-rise mt-6 max-w-xl text-base leading-relaxed text-slate-300 sm:text-lg"
              style={{ animationDelay: '0.2s' }}
            >
              BhūRaksha turns real terrain, a real landslide inventory, and antecedent rainfall
              into calibrated risk across all eight NER states — validated first in the Sikkim
              pilot, and candid about everything still pending.
            </p>

            <div
              className="brk-rise mt-8 flex flex-wrap items-center gap-3"
              style={{ animationDelay: '0.28s' }}
            >
              <button
                onClick={onEnterConsole}
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-500 px-5 py-3 text-sm font-semibold text-[#04120c] transition hover:bg-emerald-400"
              >
                Open NER operations console <ArrowRight className="h-4 w-4" />
              </button>
              <button
                onClick={onOpenSikkim}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/60 px-5 py-3 text-sm font-semibold text-slate-200 transition hover:border-emerald-500/50 hover:text-emerald-300"
              >
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_1px_rgba(52,211,153,0.7)]" />
                Sikkim validated pilot
              </button>
            </div>
          </div>

          {/* Trust strip */}
          <div
            className="brk-rise mt-16 grid max-w-4xl grid-cols-2 gap-3 sm:grid-cols-4"
            style={{ animationDelay: '0.36s' }}
          >
            <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/[0.06] p-4">
              <StatusPill tone="pilot">Sikkim</StatusPill>
              <div className="mt-2.5 text-sm font-semibold text-slate-100">Validated pilot</div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                LightGBM · terrain + antecedent rainfall
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
              <StatusPill tone="pending">7 states</StatusPill>
              <div className="mt-2.5 text-sm font-semibold text-slate-100">Validation pending</div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                awaiting DEM &amp; exposure inputs
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
              <ProvenanceTag status="REAL" />
              <div className="mt-2.5 text-sm font-semibold text-slate-100">Real inputs only</div>
              <div className="mt-0.5 text-[11px] text-slate-400">
                terrain · inventory · rainfall
              </div>
            </div>
            <div className="rounded-xl border border-red-500/25 bg-red-500/[0.05] p-4">
              <ProvenanceTag status="UNAVAILABLE" />
              <div className="mt-2.5 text-sm font-semibold text-slate-100">IMERG satellite</div>
              <div className="mt-0.5 text-[11px] text-slate-400">Earthdata auth required</div>
            </div>
          </div>
        </div>
      </section>

      {/* Method */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <Eyebrow>The BhūRaksha method</Eyebrow>
        <h2 className="mt-3 max-w-2xl text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
          A physically-grounded pipeline, every input tagged by its true source.
        </h2>
        <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {method.map((m) => {
            const Icon = m.icon;
            return (
              <div
                key={m.title}
                className="group relative flex flex-col rounded-xl border border-slate-800 bg-slate-900/40 p-5 transition hover:border-emerald-500/40"
              >
                <div className="flex items-center justify-between">
                  <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-700 bg-slate-950 text-emerald-400">
                    <Icon className="h-5 w-5" />
                  </span>
                  <ProvenanceTag status={m.status} source={m.source} />
                </div>
                <div className="mt-4 text-sm font-semibold text-slate-100">{m.title}</div>
                <p className="mt-1.5 text-[13px] leading-relaxed text-slate-400">{m.body}</p>
              </div>
            );
          })}
        </div>

        <div className="mt-6 flex items-start gap-3 rounded-xl border border-cyan-500/20 bg-cyan-500/[0.04] p-5">
          <span className="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-cyan-500/30 bg-slate-950 text-cyan-300">
            <ShieldCheck className="h-4 w-4" />
          </span>
          <div>
            <div className="text-sm font-semibold text-slate-100">
              Recommended operating mode — Option C
            </div>
            <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-slate-400">
              Static susceptibility + IMERG rainfall thresholds + forecast risk. Chosen because
              78% of cataloged events carry ≥5&nbsp;km spatial uncertainty and only 72 independent
              event-dates are available — too few for a purely learned spatiotemporal model. The
              exact justification is served live in the Sikkim console.
            </p>
          </div>
        </div>
      </section>

      {/* Honesty panel */}
      <section className="border-t border-slate-800/80 bg-[#0b0f17]">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <Eyebrow>What “validated” means here</Eyebrow>
          <h2 className="mt-3 max-w-2xl text-2xl font-bold tracking-tight text-slate-100 sm:text-3xl">
            Honest about the boundary between evidence and gap.
          </h2>
          <div className="mt-10 grid gap-4 lg:grid-cols-3">
            <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/[0.05] p-6">
              <div className="flex items-center gap-2">
                <ShieldCheck className="h-4 w-4 text-emerald-400" />
                <span className="text-sm font-semibold text-emerald-300">Validated — Sikkim</span>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-slate-300">
                Real DEM terrain, a real NASA landslide inventory, and antecedent ERA5 rainfall.
                A LightGBM model is trained and its metrics are computed on held-out events — never
                copied from a reference figure.
              </p>
            </div>
            <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.05] p-6">
              <div className="flex items-center gap-2">
                <Ruler className="h-4 w-4 text-amber-400" />
                <span className="text-sm font-semibold text-amber-300">
                  Pending — 7 NER states
                </span>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-slate-300">
                Landslide inventories are present, but the DEM and exposure layers are not yet
                acquired, so no model is trained. These states are shown as validation-pending — not
                as false positives.
              </p>
            </div>
            <div className="rounded-xl border border-red-500/25 bg-red-500/[0.05] p-6">
              <div className="flex items-center gap-2">
                <SatelliteDish className="h-4 w-4 text-red-400" />
                <span className="text-sm font-semibold text-red-300">Unavailable — IMERG</span>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-slate-300">
                NASA GPM IMERG satellite rainfall requires Earthdata authentication that is not
                configured. Rather than simulate it, BhūRaksha marks the layer unavailable
                wherever it would appear.
              </p>
            </div>
          </div>

          <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-slate-800 bg-slate-900/40 px-5 py-4 font-mono text-[11px] text-slate-400">
            <span className="inline-flex items-center gap-2">
              <Waves className="h-3.5 w-3.5 text-slate-500" /> input_status
            </span>
            <ProvenanceTag status="REAL" source="dem, terrain, glc, antecedent rainfall" />
            <ProvenanceTag status="DERIVED_PROXY" source="land_cover_class (elevation-binned)" />
            <ProvenanceTag status="NOT_USED" source="osm exposure (this training run)" />
            <ProvenanceTag status="UNAVAILABLE" source="imerg satellite rainfall" />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-[#0a0d12]">
        <div className="mx-auto flex max-w-7xl flex-col gap-6 px-6 py-10 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <BrandMark size={30} />
            <div className="text-[12px] leading-tight text-slate-500">
              <div className="font-semibold text-slate-300">BhūRaksha · भू-रक्षा</div>
              <div>Landslide Early-Warning System · North Eastern Region</div>
            </div>
          </div>
          <div className="text-[11px] leading-relaxed text-slate-500 sm:text-right">
            <div>Smart India Hackathon 2026 · MDoNER / NDMA context</div>
            <div className="mt-0.5">
              Decision-support only. Not a substitute for official disaster-management directives.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
