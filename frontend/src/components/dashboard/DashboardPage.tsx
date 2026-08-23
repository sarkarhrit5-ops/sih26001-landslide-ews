import React, { useState, useEffect } from 'react';
import { LandslideMap } from './LandslideMap';
import type { MapLayersState } from './LandslideMap';
import { IntelligencePanel } from './IntelligencePanel';
import { BottomInfoStrip } from './BottomInfoStrip';
import { LayerControl } from './LayerControl';
import type { PrimaryLayer, ForecastTime } from './LandslideMap';
import type { GridCell, NERState } from '../../data/mockCells';
import { NER_STATES } from '../../data/mockCells';
import { MapPin, ArrowLeft, Radio, Layers, ChevronDown } from 'lucide-react';
import { apiService } from '../../services/api';
import logo from '../../logo.png';

interface DashboardPageProps {
  onNavigateToLanding: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({ onNavigateToLanding }) => {
  const [states, setStates] = useState<NERState[]>(NER_STATES);
  const [selectedStateId, setSelectedStateId] = useState<string | null>(null);
  const [selectedCell, setSelectedCell] = useState<GridCell | null>(null);
  const [showLayerControl, setShowLayerControl] = useState<boolean>(false);
  const [rainfallNote, setRainfallNote] = useState<string>('Satellite Rainfall Unavailable — NASA Earthdata authentication required');

  const selectedState = selectedStateId ? states.find((s) => s.id === selectedStateId) || null : null;

  useEffect(() => {
    apiService.getValidationStatus()
      .then((data) => {
        if (!data || data.length === 0) return;
        
        const updated = states.map((st) => {
          const report = data.find((r) => 
            (r.state_id === st.id) || 
            (r.id === st.id) || 
            (r.state && r.state.toLowerCase().replace(/\s+/g, '_') === st.id)
          );
          if (!report) return st;

          let status: NERState['status'] = 'VALIDATION_PENDING';
          let statusLabel = 'VALIDATION PENDING';
          let hasValidatedPilot = false;

          const rawStatus = (report.overall_status || report.validation_status || '').toUpperCase();
          switch (rawStatus) {
            case 'VALIDATED':
            case 'VALIDATED_PILOT':
              status = 'VALIDATED_PILOT';
              statusLabel = 'VALIDATED PILOT';
              hasValidatedPilot = true;
              break;
            case 'COMPLETED':
              status = 'COMPLETED';
              statusLabel = 'COMPLETED';
              hasValidatedPilot = true;
              break;
            case 'PROCESSING':
              status = 'PROCESSING';
              statusLabel = 'PROCESSING';
              break;
            case 'DATA UNAVAILABLE':
            case 'DATA_UNAVAILABLE':
              status = 'DATA_UNAVAILABLE';
              statusLabel = 'DATA UNAVAILABLE';
              break;
            case 'INSUFFICIENT DATA':
            case 'INSUFFICIENT_DATA':
              status = 'INSUFFICIENT_DATA';
              statusLabel = 'INSUFFICIENT DATA';
              break;
            case 'ERROR':
              status = 'ERROR';
              statusLabel = 'ERROR';
              break;
            case 'VALIDATION PENDING':
            case 'VALIDATION_PENDING':
            default:
              status = 'VALIDATION_PENDING';
              statusLabel = 'VALIDATION PENDING';
              break;
          }

          // Build dynamic checklist based on report
          const isSikkim = st.id === 'sikkim';
          const dynamicChecklist = [
            { label: 'NER geographic coverage', completed: true },
            { label: 'Landslide inventory validation', completed: report.usable_events >= 50 || isSikkim },
            { label: 'Terrain/data validation', completed: report.dem_status === 'Available' || isSikkim },
            { label: 'Model validation', completed: report.overall_status === 'VALIDATED' || report.overall_status === 'COMPLETED' }
          ];

          return {
            ...st,
            status,
            statusLabel,
            hasValidatedPilot,
            coverageArea: isSikkim ? 'East Sikkim' : `Entire State (${statusLabel})`,
            checklist: dynamicChecklist
          };
        });
        setStates(updated);
        
        // Find if there is any rainfall fallback note
        const anyRainReport = data.find(r => r.rainfall_status.includes('Fallback') || r.rainfall_status.includes('Unavailable'));
        if (anyRainReport) {
          setRainfallNote(anyRainReport.rainfall_status);
        } else {
          setRainfallNote('Satellite Rainfall Active');
        }
      })
      .catch((err) => {
        console.warn('[DashboardPage] Failed to load dynamic validation status:', err);
      });
  }, []);

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
      setSelectedStateId(null);
      setSelectedCell(null);
    } else {
      setSelectedStateId(val);
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
            <img src={logo} alt="SIH landslide intelligence logo" className="w-8 h-8 rounded-full object-cover border border-emerald-500/50 shadow-md" />
            <div>
              <div className="font-bold tracking-tight text-white text-sm flex items-center gap-2">
                LANDSLIDE INTELLIGENCE
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-slate-800 text-emerald-400 border border-slate-700">
                  SIH 2026
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
              {states.map((st) => (
                <option key={st.id} value={st.id}>
                  {st.name} ({st.statusLabel})
                </option>
              ))}
            </select>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 absolute right-2.5 pointer-events-none" />
          </div>

          {/* Data Status Indicator */}
          <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-amber-500/40 text-amber-300 text-xs flex items-center gap-2 font-mono">
            <Radio className="w-3 h-3 text-amber-400 animate-pulse" />
            <span className="hidden xl:inline">{rainfallNote}</span>
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
            states={states}
            onSelectCell={setSelectedCell}
            onSelectState={(st) => setSelectedStateId(st ? st.id : null)}
            layers={layers}
          />
        </div>

        {/* Contextual Intelligence Panel (Right Panel - Approx 30-35% workspace) */}
        <div className="lg:col-span-4 xl:col-span-4 h-full min-h-[450px]">
          <IntelligencePanel
            selectedState={selectedState}
            selectedCell={selectedCell}
            states={states}
            onClearSelection={() => setSelectedCell(null)}
            onSelectCell={setSelectedCell}
            onSelectState={(st) => setSelectedStateId(st ? st.id : null)}
          />
        </div>
      </div>

      {/* Bottom Information Strip */}
      <BottomInfoStrip />
    </div>
  );
};
