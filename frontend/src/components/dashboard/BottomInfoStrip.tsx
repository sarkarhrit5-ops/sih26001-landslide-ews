import React from 'react';
import { CloudRain, Clock, Database, Radio } from 'lucide-react';

export const BottomInfoStrip: React.FC = () => {
  return (
    <div className="bg-[#0b0e14] border-t border-slate-800/80 px-4 py-2.5 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-4 font-mono shadow-md">
      {/* Rainfall Windows */}
      <div className="flex items-center gap-2">
        <CloudRain className="w-3.5 h-3.5 text-blue-400 shrink-0" />
        <span className="text-[11px] text-slate-500 uppercase tracking-wider">Rainfall Windows:</span>
        <div className="flex items-center gap-1.5 text-[11px]">
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">1h: 4.2mm</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">3h: 12.8mm</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">6h: 24.5mm</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-amber-300">24h: 55.0mm</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">72h: 110.0mm</span>
        </div>
      </div>

      {/* Forecast Windows */}
      <div className="flex items-center gap-2">
        <Clock className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
        <span className="text-[11px] text-slate-500 uppercase tracking-wider">Forecast 72h:</span>
        <div className="flex items-center gap-1.5 text-[11px]">
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">+24h: 35mm</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">+48h: 75mm</span>
          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-amber-300">+72h: 110mm</span>
        </div>
      </div>

      {/* System Status & Data Sources */}
      <div className="flex items-center gap-4 text-[11px]">
        <div className="flex items-center gap-1.5 text-emerald-400">
          <Radio className="w-3 h-3 animate-pulse" />
          <span>System Active</span>
        </div>

        <div className="flex items-center gap-1.5 text-slate-400">
          <Database className="w-3 h-3 text-slate-500" />
          <span>Copernicus GLO-30 DEM • ERA5 Reanalysis</span>
        </div>
      </div>
    </div>
  );
};
