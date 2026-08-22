import React from 'react';
import { Layers, Mountain, CloudRain, MapPin, History, Shield } from 'lucide-react';
import type { MapLayersState } from './LandslideMap';

interface LayerControlProps {
  layers: MapLayersState;
  onToggleLayer: (key: keyof MapLayersState) => void;
}

export const LayerControl: React.FC<LayerControlProps> = ({ layers, onToggleLayer }) => {
  return (
    <div className="bg-slate-900/90 backdrop-blur border border-slate-800 rounded-lg p-3 shadow-lg space-y-2 text-xs">
      <div className="flex items-center gap-2 font-mono text-[11px] font-semibold text-slate-300 uppercase tracking-wider border-b border-slate-800 pb-2">
        <Layers className="w-3.5 h-3.5 text-emerald-400" />
        Map Intelligence Layers
      </div>

      <div className="space-y-1.5 pt-1">
        <label className="flex items-center gap-2.5 cursor-pointer text-slate-200 hover:text-white transition-colors">
          <input
            type="checkbox"
            checked={layers.riskOverlay}
            onChange={() => onToggleLayer('riskOverlay')}
            className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
          />
          <Shield className="w-3.5 h-3.5 text-amber-400" />
          <span>Risk Susceptibility Overlay</span>
        </label>

        <label className="flex items-center gap-2.5 cursor-pointer text-slate-300 hover:text-white transition-colors">
          <input
            type="checkbox"
            checked={layers.terrainContours}
            onChange={() => onToggleLayer('terrainContours')}
            className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
          />
          <Mountain className="w-3.5 h-3.5 text-teal-400" />
          <span>Terrain Contours (30m DEM)</span>
        </label>

        <label className="flex items-center gap-2.5 cursor-pointer text-slate-300 hover:text-white transition-colors">
          <input
            type="checkbox"
            checked={layers.rainfallHeatmap}
            onChange={() => onToggleLayer('rainfallHeatmap')}
            className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
          />
          <CloudRain className="w-3.5 h-3.5 text-blue-400" />
          <span>Rainfall Intensity Heatmap</span>
        </label>

        <label className="flex items-center gap-2.5 cursor-pointer text-slate-300 hover:text-white transition-colors">
          <input
            type="checkbox"
            checked={layers.infrastructure}
            onChange={() => onToggleLayer('infrastructure')}
            className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
          />
          <MapPin className="w-3.5 h-3.5 text-indigo-400" />
          <span>Infrastructure Overlays</span>
        </label>

        <label className="flex items-center gap-2.5 cursor-pointer text-slate-300 hover:text-white transition-colors">
          <input
            type="checkbox"
            checked={layers.landslideEvents}
            onChange={() => onToggleLayer('landslideEvents')}
            className="rounded border-slate-700 bg-slate-950 text-emerald-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
          />
          <History className="w-3.5 h-3.5 text-amber-500" />
          <span>Historical Landslide Events</span>
        </label>
      </div>
    </div>
  );
};
