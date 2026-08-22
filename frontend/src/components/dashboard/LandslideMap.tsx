import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { GridCell } from '../../data/mockCells';
import { EAST_SIKKIM_CELLS } from '../../data/mockCells';
import type { WarningLevel } from '../../services/api';

export interface MapLayersState {
  riskOverlay: boolean;
  terrainContours: boolean;
  rainfallHeatmap: boolean;
  infrastructure: boolean;
  landslideEvents: boolean;
}

interface LandslideMapProps {
  selectedCell: GridCell | null;
  onSelectCell: (cell: GridCell | null) => void;
  layers: MapLayersState;
}

export const LandslideMap: React.FC<LandslideMapProps> = ({
  selectedCell,
  onSelectCell,
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

    // Initialize Leaflet map if not created
    if (!mapInstanceRef.current) {
      const map = L.map(mapContainerRef.current, {
        center: [27.33, 88.61],
        zoom: 10,
        zoomControl: false,
        attributionControl: false
      });

      // Add CartoDB Dark Matter tile layer
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

    // 1. Render Risk Susceptibility Overlay Cells
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

        // Tooltip on hover
        polygon.bindTooltip(
          `<div class="text-xs font-sans">
            <strong class="text-slate-100">${cell.name}</strong><br/>
            <span class="text-[10px] font-mono text-slate-300">Risk Level: <b style="color:${color}">${cell.warningLevel}</b> (Score: ${cell.finalRisk})</span>
          </div>`,
          { sticky: true, opacity: 0.95 }
        );

        polygon.on('click', () => {
          onSelectCell(cell);
        });

        polygon.addTo(layerGroup);
      });
    }

    // 2. Render Infrastructure Overlay Icons
    if (layers.infrastructure) {
      const infraPoints = [
        { lat: 27.33, lon: 88.61, name: 'STNM Central Hospital Gangtok', type: 'hospital' },
        { lat: 27.15, lon: 88.50, name: 'NH10 Teesta Corridor', type: 'road' },
        { lat: 27.39, lon: 88.52, name: 'Dikchu Hydro Feeder Bridge', type: 'bridge' },
        { lat: 27.24, lon: 88.59, name: 'Pakyong Airport Approach', type: 'road' },
        { lat: 27.50, lon: 88.53, name: 'North Sikkim Highway Segment', type: 'road' }
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

    // 3. Render Historical Landslide Events Markers
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

    // Invalidate map size to ensure fluid layout on resize
    setTimeout(() => {
      map.invalidateSize();
    }, 100);
  }, [selectedCell, layers, onSelectCell]);

  return (
    <div className="relative w-full h-full min-h-[450px] bg-[#0b0f17] rounded-xl overflow-hidden border border-slate-800 shadow-inner">
      <div ref={mapContainerRef} className="w-full h-full min-h-[450px]" />
      
      {/* Map Legend Overlay */}
      <div className="absolute bottom-4 left-4 z-[400] bg-slate-950/90 backdrop-blur border border-slate-800 rounded-lg p-2.5 shadow-xl text-[11px] font-mono text-slate-300 space-y-1.5">
        <div className="text-[10px] font-semibold uppercase text-slate-400 tracking-wider">Hazard Susceptibility Index</div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-xs bg-emerald-500 inline-block" /> LOW
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-xs bg-amber-500 inline-block" /> MED
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-xs bg-orange-500 inline-block" /> HIGH
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-xs bg-red-500 inline-block" /> EXTREME
          </div>
        </div>
      </div>
    </div>
  );
};
