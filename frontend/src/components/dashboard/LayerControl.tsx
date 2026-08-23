import React from 'react';
import { Layers, Shield, Mountain, CloudRain, CloudLightning, MapPin, History } from 'lucide-react';
import type { MapLayersState, PrimaryLayer, ForecastTime } from './LandslideMap';

interface LayerControlProps {
  layers: MapLayersState;
  onChangeLayer: (layer: PrimaryLayer) => void;
  onChangeForecastTime: (time: ForecastTime) => void;
}

export const LayerControl: React.FC<LayerControlProps> = ({ layers, onChangeLayer, onChangeForecastTime }) => {
  const options: { id: PrimaryLayer; icon: React.ReactNode; label: string }[] = [
    { id: 'risk', icon: <Shield className="w-3.5 h-3.5 text-amber-400" />, label: 'Risk' },
    { id: 'susceptibility', icon: <Mountain className="w-3.5 h-3.5 text-teal-400" />, label: 'Susceptibility' },
    { id: 'rainfall', icon: <CloudRain className="w-3.5 h-3.5 text-blue-400" />, label: 'Rainfall' },
    { id: 'forecast', icon: <CloudLightning className="w-3.5 h-3.5 text-indigo-400" />, label: 'Forecast' },
    { id: 'exposure', icon: <MapPin className="w-3.5 h-3.5 text-rose-400" />, label: 'Exposure' },
    { id: 'events', icon: <History className="w-3.5 h-3.5 text-amber-500" />, label: 'Historical Events' },
  ];

  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg p-3 shadow-lg space-y-2 text-xs">
      <div className="flex items-center gap-2 font-mono text-[11px] font-semibold text-slate-300 uppercase tracking-wider border-b border-slate-800 pb-2">
        <Layers className="w-3.5 h-3.5 text-emerald-400" />
        Map Layers
      </div>

      <div className="space-y-1.5 pt-1">
        {options.map((opt) => (
          <div key={opt.id} className="flex flex-col gap-1.5">
            <label className="flex items-center gap-2.5 cursor-pointer text-slate-300 hover:text-white transition-colors">
              <input
                type="radio"
                name="map-layer"
                checked={layers.primary === opt.id}
                onChange={() => onChangeLayer(opt.id)}
                className="rounded-full border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
              />
              {opt.icon}
              <span>{opt.label}</span>
            </label>
            
            {/* Forecast Sub-options */}
            {opt.id === 'forecast' && layers.primary === 'forecast' && (
              <div className="ml-7 flex items-center gap-2 text-[10px] font-mono mt-1 mb-1">
                {(['24h', '48h', '72h'] as ForecastTime[]).map((time) => (
                  <button
                    key={time}
                    onClick={() => onChangeForecastTime(time)}
                    className={`px-2 py-0.5 rounded border transition-colors ${
                      layers.forecastTime === time
                        ? 'bg-indigo-500/20 border-indigo-500/50 text-indigo-300'
                        : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-300'
                    }`}
                  >
                    {time}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
