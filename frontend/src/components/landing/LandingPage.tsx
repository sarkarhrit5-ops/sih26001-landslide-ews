import React from 'react';
import { HeroTopography } from './HeroTopography';
import { Shield, CloudRain, Mountain, Activity, ArrowRight, Layers, Cpu, Eye, Lock } from 'lucide-react';

interface LandingPageProps {
  onNavigateToDashboard: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigateToDashboard }) => {
  const scrollToHowItWorks = () => {
    document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen bg-[#0a0d12] text-slate-100 flex flex-col font-sans selection:bg-emerald-500 selection:text-black">
      {/* Top Navbar */}
      <header className="sticky top-0 z-50 bg-[#0a0d12]/90 backdrop-blur-md border-b border-slate-800/80 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-950 border border-emerald-500/50 flex items-center justify-center text-emerald-400 font-mono font-bold text-sm shadow-md">
              SIH
            </div>
            <div>
              <div className="font-bold tracking-tight text-slate-100 text-sm flex items-center gap-2">
                SIH26001
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded bg-slate-800 text-emerald-400 border border-slate-700">
                  Landslide EWS
                </span>
              </div>
              <p className="text-[11px] text-slate-400">Northeast India Early Warning Intelligence</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={scrollToHowItWorks}
              className="text-xs font-medium text-slate-300 hover:text-white transition-colors px-3 py-2"
            >
              How It Works
            </button>
            <button
              onClick={onNavigateToDashboard}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-slate-950 font-semibold text-xs transition-all shadow-lg shadow-emerald-950/40 cursor-pointer"
            >
              Explore Risk Intelligence
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative px-6 pt-12 pb-20 max-w-7xl mx-auto w-full flex-1 flex flex-col justify-center">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          <div className="lg:col-span-6 space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-medium text-emerald-400">
              <Shield className="w-3.5 h-3.5" />
              <span>SIH26001 • Geospatial Risk Intelligence</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-white leading-[1.1]">
              Predict the Risk.
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 via-teal-300 to-amber-400">
                Protect the Ground
              </span>{' '}
              Beneath Us.
            </h1>

            <p className="text-slate-300 text-base leading-relaxed max-w-xl">
              Combining 30m terrain susceptibility, NASA GPM rainfall triggers, weather forecasts, and infrastructure exposure to deliver dynamic landslide risk intelligence for Northeast India.
            </p>

            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-400 flex items-start gap-2.5">
              <Lock className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <p>
                <strong className="text-slate-200">Scientific Research & Disaster Management Platform.</strong> Does not claim to issue official certified government warning notifications.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={onNavigateToDashboard}
                className="inline-flex items-center gap-2.5 px-6 py-3 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold text-sm transition-all shadow-xl shadow-emerald-950/50 cursor-pointer"
              >
                Explore Risk Intelligence
                <ArrowRight className="w-4 h-4" />
              </button>
              <button
                onClick={scrollToHowItWorks}
                className="inline-flex items-center gap-2 px-5 py-3 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 font-medium text-sm transition-all cursor-pointer"
              >
                How It Works
              </button>
            </div>
          </div>

          <div className="lg:col-span-6 w-full">
            <HeroTopography />
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-16 px-6 bg-slate-950/80 border-t border-slate-800/80">
        <div className="max-w-7xl mx-auto space-y-12">
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <span className="text-xs font-mono text-emerald-400 uppercase tracking-widest">
              SYSTEM ARCHITECTURE
            </span>
            <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
              Option C Structured Risk Pipeline
            </h2>
            <p className="text-slate-400 text-sm">
              Modular integration combining static terrain susceptibility with empirical rainfall triggers and 72-hour forecast escalation.
            </p>
          </div>

          {/* Clean Horizontal Flow */}
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 relative">
            <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between space-y-4">
              <div className="w-10 h-10 rounded-lg bg-emerald-950 border border-emerald-800 flex items-center justify-center text-emerald-400">
                <Mountain className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-500 uppercase">STEP 1</span>
                <h3 className="text-sm font-semibold text-white mt-0.5">Susceptibility</h3>
                <p className="text-xs text-slate-400 mt-1">LightGBM terrain model trained on Copernicus 30m DEM derivatives.</p>
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between space-y-4">
              <div className="w-10 h-10 rounded-lg bg-blue-950 border border-blue-800 flex items-center justify-center text-blue-400">
                <CloudRain className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-500 uppercase">STEP 2</span>
                <h3 className="text-sm font-semibold text-white mt-0.5">Rainfall Trigger</h3>
                <p className="text-xs text-slate-400 mt-1">East Sikkim empirical threshold: I &gt; 14.2 D^-0.62.</p>
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between space-y-4">
              <div className="w-10 h-10 rounded-lg bg-indigo-950 border border-indigo-800 flex items-center justify-center text-indigo-400">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-500 uppercase">STEP 3</span>
                <h3 className="text-sm font-semibold text-white mt-0.5">Forecast Risk</h3>
                <p className="text-xs text-slate-400 mt-1">72-hour forecast precipitation risk escalation factor.</p>
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between space-y-4">
              <div className="w-10 h-10 rounded-lg bg-amber-950 border border-amber-800 flex items-center justify-center text-amber-400">
                <Layers className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-500 uppercase">STEP 4</span>
                <h3 className="text-sm font-semibold text-white mt-0.5">Exposure</h3>
                <p className="text-xs text-slate-400 mt-1">Spatial matching with OSM roads, hospitals, and settlements.</p>
              </div>
            </div>

            <div className="p-5 rounded-xl bg-slate-900/90 border border-emerald-500/50 flex flex-col justify-between space-y-4 bg-gradient-to-b from-slate-900 to-emerald-950/40">
              <div className="w-10 h-10 rounded-lg bg-emerald-500 text-slate-950 flex items-center justify-center font-bold">
                <Shield className="w-5 h-5" />
              </div>
              <div>
                <span className="text-[10px] font-mono text-emerald-400 uppercase">RESULT</span>
                <h3 className="text-sm font-bold text-white mt-0.5">Final Risk</h3>
                <p className="text-xs text-slate-300 mt-1">Uncollapsed risk scores, warning levels, and SHAP explanations.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Built for Northeast Section */}
      <section className="py-16 px-6 max-w-7xl mx-auto w-full">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
            <Cpu className="w-6 h-6 text-emerald-400" />
            <h3 className="font-semibold text-white text-base">Terrain-Aware Monitoring</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              30m resolution Copernicus GLO-30 DEM processing extracting slope, aspect, roughness, and TPI without chunk boundary artifacts.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
            <CloudRain className="w-6 h-6 text-blue-400" />
            <h3 className="font-semibold text-white text-base">Rainfall-Trigger Analysis</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Empirical threshold calculations evaluating 1h to 72h antecedent rainfall against critical intensity-duration limits.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
            <Layers className="w-6 h-6 text-amber-400" />
            <h3 className="font-semibold text-white text-base">Infrastructure Exposure</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              OpenStreetMap spatial intersection pinpointing critical transport corridors, STNM Hospital Gangtok, and local settlements.
            </p>
          </div>

          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
            <Eye className="w-6 h-6 text-indigo-400" />
            <h3 className="font-semibold text-white text-base">Explainable Risk AI</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              SHAP feature attribution explaining exact terrain and rainfall drivers for every high-hazard cell.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 bg-slate-950 px-6 py-8 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div>
            <span className="text-slate-300 font-semibold">SIH26001</span> — AI-Based Early Warning & Landslide Risk Monitoring System in NER
          </div>
          <div className="flex items-center gap-6 text-slate-400">
            <span>Disaster Management Division</span>
            <span>•</span>
            <span>MDoNER</span>
            <span>•</span>
            <span>East Sikkim Pilot</span>
          </div>
        </div>
      </footer>
    </div>
  );
};
