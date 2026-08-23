import React, { useState } from 'react';
import { LandslideMap } from './LandslideMap';
import type { MapLayersState } from './LandslideMap';
import { IntelligencePanel } from './IntelligencePanel';
import { BottomInfoStrip } from './BottomInfoStrip';
import { LayerControl } from './LayerControl';
import type { PrimaryLayer, ForecastTime } from './LandslideMap';
import type { GridCell, NERState } from '../../data/mockCells';
import { NER_STATES } from '../../data/mockCells';
import { MapPin, ArrowLeft, Radio, Layers, ChevronDown } from 'lucide-react';

interface DashboardPageProps {
  onNavigateToLanding: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigateToLanding }) => {
  const [selectedState, setSelectedState] = useState<NERState | null>(null);
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [showLayerControl, setShowLayerControl] = useState<boolean>(false);

  const [layers, setLayers] = useState<MapLayersState>({
    primary: 'risk',
    forecastTime: '24h'
  });

  const handleChangeLayer = (layer: PrimaryLayer) => {
    setLayers((prev) => ({ ...prev, primary: layer }));
  };

  const handleChangeForecastTime = (time: ForecastTime) => {
    setLayers((prev) => ({ ...prev, forecastTime: time }));
  };

  const handleStateSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (val === 'all') {
      setSelectedState(null);
      setSelectedCell(null);
    } else {
      const st = NER_STATES.find((s) => s.id === val) || null;
      setSelectedState(st);
      setSelectedCell(null);
    }
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
              <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono">
                <span>System Scope: Northeast India — 8 States</span>
                <span>•</span>
                <span className="text-emerald-400 font-medium">Model Validation: East Sikkim — Validated Pilot</span>
              </div>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* State / Region Selector Dropdown containing all 8 NER States */}
          <div className="relative flex items-center">
            <MapPin className="w-3.5 h-3.5 text-emerald-400 absolute left-3 pointer-events-none" />
            <select
              value={selectedState ? selectedState.id : 'all'}
              onChange={handleStateSelectChange}
              className="bg-slate-900 border border-slate-700 rounded-lg pl-8 pr-8 py-1.5 text-xs font-medium text-slate-200 focus:outline-none focus:border-emerald-500 appearance-none cursor-pointer"
            >
              <option value="all">All Northeast India (NER — 8 States)</option>
              {NER_STATES.map((st) => (
                <option key={st.id} value={st.id}>
                  {st.name} {st.hasValidatedPilot ? '(Active Validated Pilot)' : '(Validation Pending)'}
                </option>
              ))}
            </select>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 pointer-events-none" />
          </div>

          {/* Data Status Indicator */}
          <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-amber-500/40 text-amber-300 text-xs flex items-center gap-2 font-mono">
            <Radio className="w-3 h-3 text-amber-400 animate-pulse" />
            <span className="hidden xl:inline">Satellite Rainfall Unavailable — NASA Earthdata authentication required</span>
            <span className="xl:hidden">LIVE DATA UNAVAILABLE</span>
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
            <LayerControl 
              layers={layers} 
              onChangeLayer={handleChangeLayer} 
              onChangeForecastTime={handleChangeForecastTime}
            />
          </div>
        )}

        {/* Main Map Area (Approx 65-70% workspace) */}
        <div className="lg:col-span-8 xl:col-span-8 flex flex-col h-full min-h-[450px]">
          <LandslideMap
            selectedState={selectedState}
            selectedCell={selectedCell}
            onSelectCell={setSelectedCell}
            onSelectState={setSelectedState}
            layers={layers}
          />
        </div>

        {/* Contextual Intelligence Panel (Right Panel - Approx 30-35% workspace) */}
        <div className="lg:col-span-4 xl:col-span-4 h-full min-h-[450px]">
          <IntelligencePanel
            selectedState={selectedState}
            selectedCell={selectedCell}
            onClearSelection={() => setSelectedCell(null)}
            onSelectCell={setSelectedCell}
            onSelectState={setSelectedState}
          />
        </div>
      </div>

      {/* Bottom Information Strip */}
      <BottomInfoStrip />
    </div>
  );
};
