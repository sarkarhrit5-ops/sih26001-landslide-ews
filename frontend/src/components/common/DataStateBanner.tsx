import React from 'react';
import { Database, CheckCircle2, Clock, CloudRain } from 'lucide-react';

export type DataMode = 'LIVE' | 'HISTORICAL' | 'FORECAST' | 'UNAVAILABLE';

interface DataStateBannerProps {
  mode: DataMode;
  message?: string;
  className?: string;
}

export const DataStateBanner: React.FC<DataStateBannerProps> = ({ mode, message, className = '' }) => {
  const getBannerConfig = () => {
    switch (mode) {
      case 'LIVE':
        return {
          bg: 'bg-emerald-950/60 border-emerald-800/80 text-emerald-300',
          badge: 'bg-emerald-500 text-black',
          icon: <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />,
          title: 'Live Satellite Stream Active',
          desc: message || 'Real-time NASA GPM IMERG satellite precipitation feed connected.'
        };
      case 'HISTORICAL':
        return {
          bg: 'bg-blue-950/60 border-blue-800/80 text-blue-300',
          badge: 'bg-blue-500 text-white',
          icon: <Database className="w-4 h-4 text-blue-400 shrink-0" />,
          title: 'Historical ERA5 Reanalysis Data',
          desc: message || '14-day antecedent precipitation extracted from historical reanalysis archives.'
        };
      case 'FORECAST':
        return {
          bg: 'bg-indigo-950/60 border-indigo-800/80 text-indigo-300',
          badge: 'bg-indigo-500 text-white',
          icon: <Clock className="w-4 h-4 text-indigo-400 shrink-0" />,
          title: 'Open-Meteo 72h Forecast Feed',
          desc: message || '72-hour precipitation forecast model active for East Sikkim.'
        };
      case 'UNAVAILABLE':
      default:
        return {
          bg: 'bg-amber-950/70 border-amber-700/80 text-amber-200',
          badge: 'bg-amber-500 text-black',
          icon: <CloudRain className="w-4 h-4 text-amber-400 shrink-0" />,
          title: 'Satellite Rainfall Unavailable',
          desc: message || 'NASA Earthdata authentication is required for live precipitation retrieval. Operating on Copernicus DEM terrain susceptibility baseline.'
        };
    }
  };

  const config = getBannerConfig();

  return (
    <div className={`p-3 rounded-lg border text-xs flex items-start gap-2.5 shadow-sm ${config.bg} ${className}`}>
      {config.icon}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="font-semibold tracking-wide text-slate-100">{config.title}</span>
          <span className={`px-1.5 py-0.2 rounded text-[10px] font-bold uppercase tracking-wider ${config.badge}`}>
            {mode}
          </span>
        </div>
        <p className="text-slate-300 leading-relaxed text-[11px]">{config.desc}</p>
      </div>
    </div>
  );
};
