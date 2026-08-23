import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { GridCell, NERState } from '../../data/mockCells';
import { EAST_SIKKIM_CELLS, NER_STATES, NER_BOUNDS } from '../../data/mockCells';
import type { WarningLevel } from '../../services/api';
import { AlertTriangle, Info } from 'lucide-react';

export type PrimaryLayer = 'risk' | 'susceptibility' | 'rainfall' | 'forecast' | 'exposure' | 'events';
export type ForecastTime = '24h' | '48h' | '72h';

export interface MapLayersState {
  primary: PrimaryLayer;
  forecastTime?: ForecastTime;
}

interface LandslideMapProps {
  selectedState: NERState | null; // null means "All Northeast India"
  selectedCell: GridCell | null;
  onSelectCell: (cell: GridCell | null) => void;
  onSelectState: (state: NERState | null) => void;
  layers: MapLayersState;
}

export const LandslideMap: React.FC<LandslideMapProps> = ({
  selectedState,
  selectedCell,
  onSelectCell,
  onSelectState,
  layers
}) => {
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const layerGroupRef = useRef<L.LayerGroup | null>(null);

  // Helper for polygon colors by warning level
  const getCellColor = (level: WarningLevel) => {
    switch (level) {
      case 'EXTREME': return '#ef4444';
      case 'HIGH': return '#f97316';
      case 'MEDIUM': return '#f59e0b';
      case 'LOW': default: return '#10b981';
    }
  };

  const getSusceptibilityColor = (score: number) => {
    if (score >= 0.7) return '#ef4444'; // High
    if (score >= 0.4) return '#f59e0b'; // Moderate
    return '#10b981'; // Low
  };

  useEffect(() => {
    if (!mapContainerRef.current) return;

    // Initialize Leaflet map centered at Northeast India (NER)
    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [25.65, 92.6],
        zoom: 6.5,
        zoomControl: false,
        attributionControl: false
      });

      // CartoDB Dark Matter tile layer
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 18,
        subdomains: 'abcd'
      }).addTo(map);

      // Add custom zoom control
      L.control.zoom({ position: 'bottomright' }).addTo(map);

      mapInstanceRef.current = map;
      layerGroupRef.current = L.layerGroup().addTo(map);
    }

    const map = mapInstanceRef.current;
    const layerGroup = layerGroupRef.current;
    if (!map || !layerGroup) return;

    // Clear existing dynamic layers
    layerGroup.clearLayers();

    // 1. Render All 8 NER State Labels as compact rounded boxes
    NER_STATES.forEach((st) => {
      const isPilot = st.hasValidatedPilot;

      const iconHtml = isPilot
        ? `<div style="
                background:rgba(16,185,129,0.12);
                border:1.5px solid #10b981;
                border-radius:4px;
                padding:2px 6px;
                text-align:center;
                pointer-events:auto;
                cursor:pointer;
                white-space:nowrap;
                box-shadow:0 1px 4px rgba(0,0,0,0.4);
              ">
             <div style="font-family:ui-monospace,monospace;font-size:10px;font-weight:700;color:#34d399;
                         letter-spacing:0.3px;">${st.name}</div>
             <div style="font-family:ui-monospace,monospace;font-size:7px;color:#a7f3d0;text-transform:uppercase;
                         letter-spacing:0.6px;margin-top:1px;">validated pilot</div>
           </div>`
        : `<div style="
                background:rgba(100,116,139,0.12);
                border:1px solid #475569;
                border-radius:4px;
                padding:2px 6px;
                text-align:center;
                pointer-events:auto;
                cursor:pointer;
                white-space:nowrap;
                box-shadow:0 1px 3px rgba(0,0,0,0.3);
              ">
             <span style="font-family:ui-monospace,monospace;font-size:10px;color:#94a3b8;
                          letter-spacing:0.2px;">${st.name}</span>
           </div>`;

      const customIcon = L.divIcon({
        html: iconHtml,
        className: 'custom-state-label-icon',
        iconSize: isPilot ? [80, 26] : [70, 16],
        iconAnchor: isPilot ? [40, 13] : [35, 8]
      });

      const marker = L.marker([st.lat, st.lon], { icon: customIcon });

      const tooltipContent = isPilot
        ? `<div style="font-family:ui-monospace,monospace;font-size:11px;line-height:1.6;">
             <strong style="color:#f1f5f9;">Sikkim</strong><br/>
             <span style="color:#34d399;font-size:10px;">Status: Validated Pilot</span><br/>
             <span style="color:#94a3b8;font-size:10px;">Pilot: East Sikkim</span>
           </div>`
        : `<div style="font-family:ui-monospace,monospace;font-size:11px;line-height:1.6;">
             <strong style="color:#f1f5f9;">${st.name}</strong><br/>
             <span style="color:#94a3b8;font-size:10px;">Status: Validation Pending</span>
           </div>`;

      marker.bindTooltip(tooltipContent, { direction: 'top', offset: [0, -6], opacity: 0.95 });
      marker.on('click', () => onSelectState(st));
      marker.addTo(layerGroup);
    });

    // 2. Map Modes Rendering
    if (layers.primary === 'risk') {
      EAST_SIKKIM_CELLS.forEach((cell) => {
        const isSelected = selectedCell?.id === cell.id;
        const color = getCellColor(cell.warningLevel);

        const polygon = L.rectangle(cell.bounds, {
          color: isSelected ? '#ffffff' : color,
          weight: isSelected ? 3 : 1.5,
          opacity: isSelected ? 1.0 : 0.8,
          fillColor: color,
          fillOpacity: isSelected ? 0.45 : 0.25,
          dashArray: cell.warningLevel === 'EXTREME' ? '4, 4' : undefined
        });

        polygon.bindTooltip(
          `<div class="text-xs font-sans">
            <strong class="text-slate-100">${cell.name}</strong><br/>
            <span class="text-[10px] font-mono text-slate-300">East Sikkim Validated Cell</span><br/>
            <span class="text-[10px] font-mono text-slate-300">Risk Level: <b style="color:${color}">${cell.warningLevel}</b> (${cell.finalRisk})</span>
          </div>`,
          { sticky: true, opacity: 0.95 }
        );

        polygon.on('click', () => onSelectCell(cell));
        polygon.addTo(layerGroup);
      });
    } else if (layers.primary === 'susceptibility') {
      EAST_SIKKIM_CELLS.forEach((cell) => {
        const isSelected = selectedCell?.id === cell.id;
        const color = getSusceptibilityColor(cell.susceptibility);

        const polygon = L.rectangle(cell.bounds, {
          color: isSelected ? '#ffffff' : color,
          weight: isSelected ? 3 : 1.5,
          opacity: isSelected ? 1.0 : 0.8,
          fillColor: color,
          fillOpacity: isSelected ? 0.45 : 0.25
        });

        polygon.bindTooltip(
          `<div class="text-xs font-sans">
            <strong class="text-slate-100">${cell.name}</strong><br/>
            <span class="text-[10px] font-mono text-slate-300">Susceptibility Score: <b style="color:${color}">${cell.susceptibility.toFixed(2)}</b></span>
          </div>`,
          { sticky: true, opacity: 0.95 }
        );

        polygon.on('click', () => onSelectCell(cell));
        polygon.addTo(layerGroup);
      });
    } else if (layers.primary === 'forecast' && layers.forecastTime === '72h') {
      EAST_SIKKIM_CELLS.forEach((cell) => {
        const isSelected = selectedCell?.id === cell.id;
        const color = '#3b82f6'; // Blue gradient mapped to rain

        const polygon = L.rectangle(cell.bounds, {
          color: isSelected ? '#ffffff' : color,
          weight: isSelected ? 3 : 1.5,
          opacity: isSelected ? 1.0 : 0.8,
          fillColor: color,
          fillOpacity: Math.min(0.8, cell.forecastRain / 150) // Scale opacity by rain
        });

        polygon.bindTooltip(
          `<div class="text-xs font-sans">
            <strong class="text-slate-100">${cell.name}</strong><br/>
            <span class="text-[10px] font-mono text-slate-300">72h Forecast Rain: <b>${cell.forecastRain} mm</b></span>
          </div>`,
          { sticky: true, opacity: 0.95 }
        );

        polygon.on('click', () => onSelectCell(cell));
        polygon.addTo(layerGroup);
      });
    } else if (layers.primary === 'exposure') {
      const infraPoints = [
        { lat: 27.33, lon: 88.61, name: 'STNM Central Hospital Gangtok (Sikkim)', type: 'hospital' },
        { lat: 27.15, lon: 88.50, name: 'NH10 Teesta Valley Corridor (Sikkim-WB)', type: 'road' },
        { lat: 27.39, lon: 88.52, name: 'Dikchu Hydro Feeder Bridge (Sikkim)', type: 'bridge' },
        { lat: 27.24, lon: 88.59, name: 'Pakyong Airport Approach (Sikkim)', type: 'road' },
        { lat: 26.18, lon: 91.75, name: 'Guwahati Dispur Transport Axis (Assam)', type: 'road' },
        { lat: 25.57, lon: 91.88, name: 'Shillong Bypass Corridor (Meghalaya)', type: 'road' }
      ];

      infraPoints.forEach((pt) => {
        const iconHtml = pt.type === 'hospital'
          ? '<div style="background:#ef4444; width:10px; height:10px; border-radius:50%; border:2px solid #fff;"></div>'
          : '<div style="background:#3b82f6; width:8px; height:8px; border-radius:50%; border:1px solid #fff;"></div>';

        const customIcon = L.divIcon({
          html: iconHtml,
          className: 'custom-infra-div-icon',
          iconSize: [12, 12]
        });

        const marker = L.marker([pt.lat, pt.lon], { icon: customIcon });
        marker.bindTooltip(`<div class="text-xs"><b>Asset:</b> ${pt.name}</div>`, { sticky: true });
        marker.addTo(layerGroup);
      });
    } else if (layers.primary === 'events') {
      const glcEvents = [
        { lat: 27.478, lon: 88.527, date: '2009-07-01', trigger: 'downpour' },
        { lat: 27.593, lon: 88.492, date: '2016-08-16', trigger: 'rain' },
        { lat: 27.838, lon: 88.556, date: '2012-09-19', trigger: 'continuous_rain' },
        { lat: 27.021, lon: 88.259, date: '2017-07-06', trigger: 'downpour' }
      ];

      glcEvents.forEach((ev) => {
        const iconHtml = '<div style="background:#f59e0b; width:7px; height:7px; transform:rotate(45deg); border:1px solid #000;"></div>';
        const customIcon = L.divIcon({
          html: iconHtml,
          className: 'custom-glc-div-icon',
          iconSize: [10, 10]
        });
        const marker = L.marker([ev.lat, ev.lon], { icon: customIcon });
        marker.bindTooltip(
          `<div class="text-xs font-mono">
            <b>Historical Landslide Event</b><br/>
            Date: ${ev.date}<br/>
            Trigger: ${ev.trigger}
          </div>`,
          { sticky: true }
        );
        marker.addTo(layerGroup);
      });
    }

    // Smooth camera panning/zooming when selectedState or selectedCell changes
    if (selectedCell) {
      map.flyTo([selectedCell.lat, selectedCell.lon], 12, { duration: 1.2 });
    } else if (selectedState) {
      map.flyTo([selectedState.lat, selectedState.lon], selectedState.zoom, { duration: 1.2 });
    } else {
      map.flyToBounds(NER_BOUNDS, { duration: 1.2 });
    }

  }, [selectedState, selectedCell, layers, onSelectCell, onSelectState]);

  return (
    <div className="relative w-full h-full min-h-[450px] bg-[#0b0f17] rounded-xl overflow-hidden border border-slate-800 shadow-inner">
      <div ref={mapContainerRef} className="w-full h-full min-h-[450px]" />
      
      {/* Missing Data Overlays */}
      {layers.primary === 'rainfall' && (
        <div className="absolute inset-0 flex items-center justify-center z-[400] pointer-events-none">
          <div className="bg-slate-900/90 backdrop-blur border border-red-900/50 rounded-xl p-6 shadow-2xl flex flex-col items-center max-w-md text-center pointer-events-auto">
            <AlertTriangle className="w-10 h-10 text-red-500 mb-3" />
            <h3 className="text-lg font-bold text-slate-100 tracking-tight mb-1">UNAVAILABLE</h3>
            <p className="text-sm text-slate-400">NASA Earthdata authentication required</p>
            <div className="mt-4 text-xs font-mono text-slate-500 bg-slate-950 px-3 py-2 rounded border border-slate-800">
              ERR_IMERG_AUTH_MISSING
            </div>
          </div>
        </div>
      )}

      {layers.primary === 'forecast' && layers.forecastTime !== '72h' && (
        <div className="absolute inset-0 flex items-center justify-center z-[400] pointer-events-none">
          <div className="bg-slate-900/90 backdrop-blur border border-amber-900/50 rounded-xl p-6 shadow-2xl flex flex-col items-center max-w-md text-center pointer-events-auto">
            <Info className="w-10 h-10 text-amber-500 mb-3" />
            <h3 className="text-lg font-bold text-slate-100 tracking-tight mb-1">DATA UNAVAILABLE</h3>
            <p className="text-sm text-slate-400">Forecast model currently only active for 72h window.</p>
          </div>
        </div>
      )}

      {/* Map Legend Overlay */}
      <div className="absolute bottom-4 left-4 z-[400] bg-slate-950/90 backdrop-blur border border-slate-800 rounded-lg p-3 shadow-xl space-y-3 min-w-[200px]">
        {layers.primary === 'risk' && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Risk Overlay</div>
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-300">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#10b981]" /> LOW</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#f59e0b]" /> MEDIUM</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#f97316]" /> HIGH</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#ef4444]" /> EXTREME</span>
            </div>
          </div>
        )}
        
        {layers.primary === 'susceptibility' && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Susceptibility</div>
            <div className="flex items-center justify-between text-[10px] font-mono text-slate-300">
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#10b981]" /> Low</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#f59e0b]" /> Moderate</span>
              <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-[#ef4444]" /> High</span>
            </div>
          </div>
        )}

        {layers.primary === 'forecast' && layers.forecastTime === '72h' && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Forecast Rain (72h)</div>
            <div className="h-1.5 w-full rounded bg-gradient-to-r from-blue-900 via-blue-500 to-cyan-300" />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>
        )}

        {layers.primary === 'rainfall' && (
          <div className="space-y-1.5 opacity-50">
            <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Rainfall Intensity</div>
            <div className="h-1.5 w-full rounded bg-gradient-to-r from-blue-900 via-blue-500 to-cyan-300" />
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>Low precipitation</span>
              <span>High precipitation</span>
            </div>
          </div>
        )}

        {layers.primary === 'exposure' && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Exposure</div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-slate-300">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full border border-white bg-blue-500 inline-block" /> Road / Bridge
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full border border-white bg-red-500 inline-block" /> Critical Asset
              </span>
            </div>
          </div>
        )}

        {layers.primary === 'events' && (
          <div className="space-y-1.5">
            <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Historical Events</div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-slate-300">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 border border-black bg-amber-500 inline-block transform rotate-45" /> Verified Landslide
              </span>
            </div>
          </div>
        )}

        {/* Geographic Scope - always visible */}
        <div className="pt-2 border-t border-slate-800 space-y-1.5">
          <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Geographic Scope: All 8 NER States</div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-300">
              <span className="w-2 h-2 rounded-xs bg-emerald-500 inline-block" /> Active Pilot
            </div>
            <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-300">
              <span className="w-2 h-2 rounded-xs bg-slate-500 inline-block" /> Validation Pending
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
