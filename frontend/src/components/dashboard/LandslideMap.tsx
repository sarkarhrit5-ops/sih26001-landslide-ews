import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { GridCell, NERState } from '../../data/mockCells';
import { EAST_SIKKIM_CELLS, NER_STATES, NER_BOUNDS } from '../../data/mockCells';
import type { WarningLevel } from '../../services/api';

export interface MapLayersState {
  riskOverlay: boolean;
  terrainContours: boolean;
  rainfallHeatmap: boolean;
  infrastructure: boolean;
  landslideEvents: boolean;
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

      // Sikkim: green/teal border with "VALIDATED PILOT" sub-line
      // Other 7 states: muted neutral border, just state name
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

      // Compact hover tooltip
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

      marker.on('click', () => {
        onSelectState(st);
      });

      marker.addTo(layerGroup);
    });

    // 2. Render Risk Susceptibility Overlay Cells (East Sikkim Pilot)
    if (layers.riskOverlay) {
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

        polygon.on('click', () => {
          onSelectCell(cell);
        });

        polygon.addTo(layerGroup);
      });
    }

    // 3. Render Infrastructure Overlay Icons (East Sikkim Pilot & Regional Corridors)
    if (layers.infrastructure) {
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
    }

    // 4. Render Historical Landslide Events Markers
    if (layers.landslideEvents) {
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
      
      {/* Map Legend Overlay */}
      <div className="absolute bottom-4 left-4 z-[400] bg-slate-950/90 backdrop-blur border border-slate-800 rounded-lg p-2.5 shadow-xl text-[11px] font-mono text-slate-300 space-y-1.5">
        <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Geographic Scope: All 8 NER States</div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-xs bg-emerald-500 inline-block" /> Active Pilot
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-xs bg-slate-500 inline-block" /> Validation Pending
          </div>
        </div>
      </div>
    </div>
  );
};
