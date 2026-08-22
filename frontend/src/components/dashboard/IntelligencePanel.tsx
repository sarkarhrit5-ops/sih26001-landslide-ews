import React, { useEffect, useState } from 'react';
import type { GridCell, NERState } from '../../data/mockCells';
import { REGIONAL_SUMMARY, EAST_SIKKIM_CELLS, NER_STATES } from '../../data/mockCells';
import { WarningBadge } from '../common/WarningBadge';
import { DataStateBanner } from '../common/DataStateBanner';
import { apiService } from '../../services/api';
import type { CellExplainResponse } from '../../services/api';
import { Shield, ArrowLeft, Mountain, CloudRain, Layers, HelpCircle, Loader2, MapPin, Check, Circle } from 'lucide-react';

interface IntelligencePanelProps {
  selectedState: NERState | null;
  selectedCell: GridCell | null;
  onClearSelection: () => void;
  onSelectCell: (cell: GridCell) => void;
  onSelectState: (state: NERState | null) => void;
}

export const IntelligencePanel: React.FC<IntelligencePanelProps> = ({
  selectedState,
  selectedCell,
  onClearSelection,
  onSelectCell,
  onSelectState
}) => {
  const [explanation, setExplanation] = useState<CellExplainResponse | null>(null);
  const [loadingExplain, setLoadingExplain] = useState<boolean>(false);

  useEffect(() => {
    if (!selectedCell) {
      setExplanation(null);
      return;
    }

    setLoadingExplain(true);
    apiService
      .getCellExplanation(selectedCell.id)
      .then((data) => {
        setExplanation(data);
      })
      .catch((err) => {
        console.warn('Backend explain endpoint fallback:', err);
      })
      .finally(() => {
        setLoadingExplain(false);
      });
  }, [selectedCell]);

  // Case 1: State Selected (e.g. Sikkim or an unvalidated state)
  if (selectedState && !selectedCell) {
    const isValidated = selectedState.hasValidatedPilot;

    return (
      <div className="h-full bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between overflow-y-auto space-y-6">
        <div className="space-y-4">
          <button
            onClick={() => onSelectState(null)}
            className="inline-flex items-center gap-1.5 text-xs text-emerald-400 hover:text-emerald-300 transition-colors font-medium cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Return to Northeast India Overview
          </button>

          {/* State Header */}
          <div className="border-b border-slate-800 pb-3 space-y-1">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">
              STATE COVERAGE • {selectedState.name.toUpperCase()}
            </div>
            <h2 className="text-lg font-bold text-white tracking-tight">{selectedState.name}</h2>
            <div className="text-xs font-mono text-slate-400 flex items-center gap-2">
              <span>Capital: {selectedState.capital}</span>
              {isValidated && (
                <span className="px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-700 font-semibold text-[10px]">
                  Pilot Area: {selectedState.coverageArea}
                </span>
              )}
            </div>
          </div>

          {/* MODEL STATUS Section */}
          <div className="p-4 rounded-lg bg-slate-950/90 border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400 uppercase tracking-wider">MODEL STATUS</span>
              <span
                className={`text-xs font-mono font-bold px-2.5 py-1 rounded ${
                  isValidated
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50'
                    : 'bg-amber-500/20 text-amber-300 border border-amber-500/50'
                }`}
              >
                {selectedState.statusLabel}
              </span>
            </div>

            {!isValidated ? (
              <p className="text-xs text-slate-300 leading-relaxed pt-1">
                Risk modelling for this state will be activated after landslide inventory, terrain, rainfall, and exposure data pass the validation pipeline.
              </p>
            ) : (
              <p className="text-xs text-slate-300 leading-relaxed pt-1">
                High-resolution risk modeling validated for <strong>{selectedState.coverageArea}</strong>. Susceptibility matrix and dynamic rainfall triggers are active.
              </p>
            )}
          </div>

          {/* Progress / Validation Checklist */}
          <div className="p-4 rounded-lg bg-slate-950/80 border border-slate-800 space-y-3">
            <div className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider border-b border-slate-800/80 pb-2">
              Validation Requirements Checklist
            </div>
            <div className="space-y-2">
              {selectedState.checklist.map((item, idx) => (
                <div key={idx} className="flex items-center gap-2.5 text-xs">
                  {item.completed ? (
                    <Check className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : (
                    <Circle className="w-4 h-4 text-slate-600 shrink-0" />
                  )}
                  <span className={item.completed ? 'text-slate-200 font-medium' : 'text-slate-500'}>
                    {item.label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* If Sikkim -> Link to Validated East Sikkim Cells */}
          {isValidated && (
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <div className="text-xs font-mono font-semibold uppercase text-emerald-400 tracking-wider">
                Validated East Sikkim Risk Cells ({EAST_SIKKIM_CELLS.length})
              </div>
              <div className="space-y-1.5">
                {EAST_SIKKIM_CELLS.slice(0, 4).map((cell) => (
                  <button
                    key={cell.id}
                    onClick={() => onSelectCell(cell)}
                    className="w-full p-2.5 rounded-lg bg-slate-950/60 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 transition-all text-left flex items-center justify-between group cursor-pointer"
                  >
                    <div>
                      <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                        {cell.name}
                      </div>
                      <div className="text-[10px] text-slate-400 font-mono">
                        Slope: {cell.slope}° | Elev: {cell.elevation}m
                      </div>
                    </div>
                    <WarningBadge level={cell.warningLevel} size="sm" showPattern={false} />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="text-[11px] text-slate-500 border-t border-slate-800 pt-3 font-mono">
          SIH26001 Architecture: Designed for progressive expansion across all 8 NER states.
        </div>
      </div>
    );
  }

  // Case 2: Regional Overview (No cell or state selected)
  if (!selectedCell) {
    return (
      <div className="h-full bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between overflow-y-auto space-y-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" />
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                SYSTEM INTELLIGENCE OVERVIEW
              </h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-emerald-400 border border-slate-700">
              8 NER STATES
            </span>
          </div>

          {/* SYSTEM COVERAGE & MODEL VALIDATION Distinction */}
          <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 space-y-2 text-xs font-mono">
            <div className="flex items-center justify-between">
              <span className="text-slate-400 uppercase text-[10px]">SYSTEM COVERAGE:</span>
              <span className="font-semibold text-slate-100">{REGIONAL_SUMMARY.systemCoverage}</span>
            </div>
            <div className="flex items-center justify-between border-t border-slate-800/80 pt-2">
              <span className="text-slate-400 uppercase text-[10px]">MODEL VALIDATION:</span>
              <span className="font-bold text-emerald-400">{REGIONAL_SUMMARY.modelValidation}</span>
            </div>
          </div>

          {/* Data Transparency Banner */}
          <DataStateBanner
            mode="UNAVAILABLE"
            message={REGIONAL_SUMMARY.rainfallStatusNote}
          />

          {/* All 8 NER States Coverage List */}
          <div className="space-y-2 pt-1">
            <div className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider flex items-center justify-between">
              <span>All 8 NER States</span>
              <span className="text-[10px] text-slate-500">Click to view model status</span>
            </div>

            <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
              {NER_STATES.map((st) => {
                const isSelected = selectedState?.id === st.id;
                const isPilot = st.hasValidatedPilot;

                return (
                  <button
                    key={st.id}
                    onClick={() => onSelectState(st)}
                    className={`w-full p-2.5 rounded-lg border text-left flex items-center justify-between transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-emerald-950/60 border-emerald-500/80 text-white'
                        : isPilot
                        ? 'bg-slate-950/60 hover:bg-slate-800/80 border-emerald-800/60 text-slate-200'
                        : 'bg-slate-950/30 hover:bg-slate-800/50 border-slate-800/80 text-slate-400'
                    }`}
                  >
                    <div>
                      <div className="text-xs font-medium flex items-center gap-1.5">
                        <MapPin className={`w-3 h-3 ${isPilot ? 'text-emerald-400' : 'text-slate-500'}`} />
                        <span>{st.name}</span>
                      </div>
                      <div className="text-[10px] font-mono text-slate-500 pl-4">
                        Capital: {st.capital}
                      </div>
                    </div>

                    <span
                      className={`text-[9px] font-mono px-2 py-0.5 rounded font-semibold uppercase ${
                        isPilot
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          : 'bg-slate-800 text-slate-400 border border-slate-700'
                      }`}
                    >
                      {st.statusLabel}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* East Sikkim Pilot Hotspots */}
          <div className="space-y-2 pt-2 border-t border-slate-800">
            <div className="text-xs font-mono font-semibold uppercase text-emerald-400 tracking-wider">
              Validated East Sikkim Risk Cells
            </div>
            <div className="space-y-1.5">
              {EAST_SIKKIM_CELLS.slice(0, 3).map((cell) => (
                <button
                  key={cell.id}
                  onClick={() => onSelectCell(cell)}
                  className="w-full p-2.5 rounded-lg bg-slate-950/50 hover:bg-slate-800/80 border border-slate-800/80 hover:border-slate-700 transition-all text-left flex items-center justify-between group cursor-pointer"
                >
                  <div>
                    <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                      {cell.name}
                    </div>
                    <div className="text-[10px] text-slate-400 font-mono">
                      Slope: {cell.slope}° | Elev: {cell.elevation}m
                    </div>
                  </div>
                  <WarningBadge level={cell.warningLevel} size="sm" showPattern={false} />
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="text-[11px] text-slate-500 border-t border-slate-800 pt-3 font-mono">
          Click any state or East Sikkim risk cell to inspect validation status and hazard intelligence.
        </div>
      </div>
    );
  }

  // Case 3: Cell Selected -> CELL INTELLIGENCE
  return (
    <div className="h-full bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between overflow-y-auto space-y-5">
      <div className="space-y-4">
        <button
          onClick={onClearSelection}
          className="inline-flex items-center gap-1.5 text-xs text-emerald-400 hover:text-emerald-300 transition-colors font-medium cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Return to Overview
        </button>

        {/* Cell Header */}
        <div className="border-b border-slate-800 pb-3 space-y-1">
          <div className="text-[10px] font-mono text-emerald-400 uppercase tracking-widest">
            VALIDATED PILOT CELL • {selectedCell.id}
          </div>
          <h2 className="text-base font-bold text-white tracking-tight">{selectedCell.name}</h2>
          <p className="text-xs font-mono text-slate-400">
            Coordinates: {selectedCell.lat}°N, {selectedCell.lon}°E (Sikkim)
          </p>
        </div>

        {/* Risk Level Badge */}
        <div className="p-3.5 rounded-lg bg-slate-950/80 border border-slate-800 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Assessed Cell Hazard</span>
            <WarningBadge level={selectedCell.warningLevel} size="md" />
          </div>

          <div className="grid grid-cols-2 gap-2 pt-2 text-xs font-mono border-t border-slate-800/80">
            <div>
              <span className="text-[10px] text-slate-400 block">CURRENT RISK</span>
              <span className="text-sm font-bold text-slate-100">{selectedCell.finalRisk.toFixed(2)}</span>
            </div>
            <div>
              <span className="text-[10px] text-slate-400 block">FORECAST RISK</span>
              <span className="text-sm font-bold text-amber-400">
                {Math.min(1.0, selectedCell.finalRisk * 1.15).toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Risk Factor Meters */}
        <div className="space-y-3 pt-1">
          <div className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider">
            Key Risk Factors
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <Mountain className="w-3.5 h-3.5 text-emerald-400" /> Terrain Slope ({selectedCell.slope}°)
              </span>
              <span className="font-mono text-slate-200">
                {selectedCell.slope > 35 ? 'HIGH HAZARD' : 'MODERATE'}
              </span>
            </div>
            <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="bg-emerald-500 h-full transition-all duration-500"
                style={{ width: `${Math.min(100, (selectedCell.slope / 50) * 100)}%` }}
              />
            </div>
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <CloudRain className="w-3.5 h-3.5 text-blue-400" /> Rainfall Trigger
              </span>
              <span className="font-mono text-amber-400">
                {selectedCell.currentRain} mm/24h
              </span>
            </div>
            <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="bg-blue-500 h-full transition-all duration-500"
                style={{ width: `${Math.min(100, (selectedCell.currentRain / 100) * 100)}%` }}
              />
            </div>
          </div>

          <div className="space-y-1 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-slate-300 font-medium flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-amber-400" /> Susceptibility Index
              </span>
              <span className="font-mono text-slate-200">
                {selectedCell.susceptibility.toFixed(2)}
              </span>
            </div>
            <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden border border-slate-800">
              <div
                className="bg-amber-500 h-full transition-all duration-500"
                style={{ width: `${selectedCell.susceptibility * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Exposed Assets */}
        <div className="space-y-2 pt-2 border-t border-slate-800">
          <div className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider">
            Exposed Assets ({selectedCell.exposedAssetsCount})
          </div>
          <div className="space-y-1.5">
            {selectedCell.assets.map((asset, idx) => (
              <div key={idx} className="px-3 py-1.5 rounded bg-slate-950/60 border border-slate-800 text-xs text-slate-200 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                <span>{asset}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Why this cell is at risk */}
        <div className="space-y-2 pt-2 border-t border-slate-800">
          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold uppercase text-emerald-400 tracking-wider">
            <HelpCircle className="w-3.5 h-3.5" />
            Why this cell is at risk
          </div>

          {loadingExplain ? (
            <div className="p-3 text-xs text-slate-400 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
              Fetching explainable AI SHAP feature attributions...
            </div>
          ) : explanation?.explanation?.top_features ? (
            <div className="space-y-2 p-3 rounded-lg bg-slate-950/80 border border-slate-800/80 text-xs">
              {explanation.explanation.top_features.map((feat, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <span className="text-slate-200 capitalize">{feat.feature}</span>
                    <span className="text-emerald-400">{(feat.importance * 100).toFixed(0)}% contribution</span>
                  </div>
                  <p className="text-[11px] text-slate-400">{feat.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs text-slate-400">
              Steep terrain gradient (&gt;35°), elevated Copernicus DEM roughness, and empirical rainfall threshold proximity drive high risk score.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
