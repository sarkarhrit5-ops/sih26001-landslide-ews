import React, { useState } from 'react';
import { LandslideMap } from './LandslideMap';
import type { MapLayersState } from './LandslideMap';
import { IntelligencePanel } from './IntelligencePanel';
import { BottomInfoStrip } from './BottomInfoStrip';
import { LayerControl } from './LayerControl';
import type { GridCell } from '../../data/mockCells';
import { REGIONAL_SUMMARY } from '../../data/mockCells';
import { MapPin, ArrowLeft, Radio, Layers } from 'lucide-react';

interface DashboardPageProps {
  onNavigateToLanding: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigateToLanding }) => {
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [showLayerControl, setShowLayerControl] = useState<boolean>(false);

  const [layers, setLayers] = useState<MapLayersState>({
    riskOverlay: true,
    terrainContours: false,
    rainfallHeatmap: false,
    infrastructure: true,
    landslideEvents: true
  });

  const handleToggleLayer = (key: keyof MapLayersState) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="min-h-screen bg-[#0a0d12] text-slate-100 flex flex-col font-sans overflow-hidden">
      {/* Top Nav */}
      <header className="bg-[#0b0e14] border-b border-slate-800/80 px-5 py-3 flex items-center justify-between z-30 shrink-0">
        <div className="flex items-center gap-4">
          <button
            onClick={onNavigateToLanding}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-slate-300 transition-colors cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Overview
          </button>

          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-950 border border-emerald-500/50 flex items-center justify-center text-emerald-400 font-mono font-bold text-xs shadow-md">
              SIH
            </div>
            <div>
              <div className="font-bold tracking-tight text-white text-sm flex items-center gap-2">
                LANDSLIDE INTELLIGENCE
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-emerald-400 border border-slate-700">
                  SIH26001
                </span>
              </div>
              <div className="flex items-center gap-2 text-[11px] text-slate-400">
                <MapPin className="w-3 h-3 text-emerald-400" />
                <span>{REGIONAL_SUMMARY.regionName} ({REGIONAL_SUMMARY.boundsStr})</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Data Status Indicator */}
          <div className="px-3 py-1 rounded-full bg-slate-900 border border-amber-500/40 text-amber-300 text-xs flex items-center gap-2 font-mono">
            <Radio className="w-3 h-3 text-amber-400 animate-pulse" />
            <span className="hidden sm:inline">Satellite Rainfall:</span> UNAVAILABLE
          </div>

          <button
            onClick={() => setShowLayerControl(!showLayerControl)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-medium text-slate-200 transition-colors cursor-pointer"
          >
            <Layers className="w-3.5 h-3.5 text-emerald-400" />
            Layers
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 p-3 gap-3 overflow-hidden relative">
        {/* Layer Control Dropdown Panel */}
        {showLayerControl && (
          <div className="absolute top-5 left-5 z-[500] w-64 shadow-2xl">
            <LayerControl layers={layers} onToggleLayer={handleToggleLayer} />
          </div>
        )}

        {/* Main Map Area (Approx 65-70% workspace) */}
        <div className="lg:col-span-8 xl:col-span-8 flex flex-col h-full min-h-[450px]">
          <LandslideMap
            selectedCell={selectedCell}
            onSelectCell={setSelectedCell}
            layers={layers}
          />
        </div>

        {/* Contextual Intelligence Panel (Right Panel - Approx 30-35% workspace) */}
        <div className="lg:col-span-4 xl:col-span-4 h-full min-h-[450px]">
          <IntelligencePanel
            selectedCell={selectedCell}
            onClearSelection={() => setSelectedCell(null)}
            onSelectCell={setSelectedCell}
          />
        </div>
      </div>

      {/* Bottom Information Strip */}
      <BottomInfoStrip />
    </div>
  );
};
