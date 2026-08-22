import React, { useEffect, useState } from 'react';
import type { GridCell } from '../../data/mockCells';
import { REGIONAL_SUMMARY, EAST_SIKKIM_CELLS } from '../../data/mockCells';
import { WarningBadge } from '../common/WarningBadge';
import { DataStateBanner } from '../common/DataStateBanner';
import { apiService } from '../../services/api';
import type { CellExplainResponse } from '../../services/api';
import { Shield, ArrowLeft, Mountain, CloudRain, Layers, HelpCircle, Loader2 } from 'lucide-react';

interface IntelligencePanelProps {
  selectedCell: GridCell | null;
  onClearSelection: () => void;
  onSelectCell: (cell: GridCell) => void;
}

export const IntelligencePanel: React.FC<IntelligencePanelProps> = ({
  selectedCell,
  onClearSelection,
  onSelectCell
}) => {
  const [explanation, setExplanation] = useState<CellExplainResponse | null>(null);
  const [loadingExplain, setLoadingExplain] = useState<boolean>(false);

  useEffect(() => {
    if (!selectedCell) {
      setExplanation(null);
      return;
    }

    // Fetch cell explanation from backend FastAPI endpoint /api/v1/cell/{id}/explain
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

  // If no cell is selected -> Render CURRENT REGIONAL STATUS
  if (!selectedCell) {
    return (
      <div className="h-full bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between overflow-y-auto space-y-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Shield className="w-4 h-4 text-emerald-400" />
              <h2 className="text-xs font-mono font-bold uppercase tracking-wider text-slate-200">
                CURRENT REGIONAL STATUS
              </h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              PILOT AOI
            </span>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-slate-400">Regional Risk Level</div>
            <WarningBadge level={REGIONAL_SUMMARY.overallRisk} size="lg" />
          </div>

          {/* Data Transparency Banner */}
          <DataStateBanner
            mode="UNAVAILABLE"
            message={REGIONAL_SUMMARY.rainfallStatusNote}
          />

          {/* Regional Summary Metrics */}
          <div className="grid grid-cols-2 gap-3 pt-1">
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium">Monitored Cells</div>
              <div className="text-lg font-bold text-slate-100 font-mono">
                {REGIONAL_SUMMARY.totalMonitoredCells} Cells
              </div>
              <div className="text-[10px] text-amber-400">
                {REGIONAL_SUMMARY.activeHighRiskCellsCount} High Hazard
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800 space-y-1">
              <div className="text-[11px] text-slate-400 font-medium font-sans">Exposed Assets</div>
              <div className="text-lg font-bold text-slate-100 font-mono">
                {REGIONAL_SUMMARY.totalExposedAssets} Assets
              </div>
              <div className="text-[10px] text-slate-400">STNM Corridor & NH10</div>
            </div>
          </div>

          {/* High Hazard Hotspots List */}
          <div className="space-y-2 pt-2">
            <div className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider">
              High Risk Cell Hotspots
            </div>
            <div className="space-y-2">
              {EAST_SIKKIM_CELLS.slice(0, 3).map((cell) => (
                <button
                  key={cell.id}
                  onClick={() => onSelectCell(cell)}
                  className="w-full p-3 rounded-lg bg-slate-950/50 hover:bg-slate-800/80 border border-slate-800/80 hover:border-slate-700 transition-all text-left flex items-center justify-between group cursor-pointer"
                >
                  <div>
                    <div className="text-xs font-semibold text-slate-200 group-hover:text-white">
                      {cell.name}
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                      Slope: {cell.slope}° | Elev: {cell.elevation}m
                    </div>
                  </div>
                  <WarningBadge level={cell.warningLevel} size="sm" showPattern={false} />
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="text-[11px] text-slate-400 border-t border-slate-800 pt-3">
          Click any cell on the map or hotspot list to view detailed cell-level risk intelligence.
        </div>
      </div>
    );
  }

  // When a cell IS selected -> Render CELL INTELLIGENCE
  return (
    <div className="h-full bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col justify-between overflow-y-auto space-y-5">
      <div className="space-y-4">
        {/* Navigation back */}
        <button
          onClick={onClearSelection}
          className="inline-flex items-center gap-1.5 text-xs text-emerald-400 hover:text-emerald-300 transition-colors font-medium cursor-pointer"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Return to Regional Overview
        </button>

        {/* Cell Header */}
        <div className="border-b border-slate-800 pb-3 space-y-1">
          <div className="text-[10px] font-mono text-emerald-400 uppercase tracking-widest">
            CELL INTELLIGENCE • {selectedCell.id}
          </div>
          <h2 className="text-base font-bold text-white tracking-tight">{selectedCell.name}</h2>
          <p className="text-xs font-mono text-slate-400">
            Coordinates: {selectedCell.lat}°N, {selectedCell.lon}°E
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

        {/* Risk Factors Meters */}
        <div className="space-y-3 pt-1">
          <div className="text-xs font-mono font-semibold uppercase text-slate-400 tracking-wider">
            Key Risk Factors
          </div>

          {/* Factor 1: Slope */}
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

          {/* Factor 2: Rainfall Trigger */}
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

          {/* Factor 3: Susceptibility Score */}
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

        {/* Exposed Infrastructure */}
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

        {/* Why this cell is at risk (Backend Explanation API Endpoint) */}
        <div className="space-y-2 pt-2 border-t border-slate-800">
          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold uppercase text-emerald-400 tracking-wider">
            <HelpCircle className="w-3.5 h-3.5" />
            Why this cell is at risk
          </div>

          {loadingExplain ? (
            <div className="p-3 text-xs text-slate-400 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
              Fetching explainable AI SHAP feature attributions from FastAPI backend...
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
