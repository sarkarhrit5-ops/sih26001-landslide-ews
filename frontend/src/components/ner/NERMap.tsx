/**
 * NER overview map — the 8 state administrative boxes drawn from config_states,
 * coloured by their live validation tone, with the Sikkim canonical pilot AOI
 * highlighted. Uses vanilla Leaflet imperatively (the pattern already proven in
 * this codebase) with a clean mount/unmount lifecycle.
 *
 * Contains no risk/rainfall/susceptibility rendering — only real geography and
 * the validation status supplied by the parent from the live backend.
 */
import { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { NerStateMeta, StateValidationTone } from '../../data/nerStates';
import { EAST_SIKKIM_PILOT_AOI, NER_FIT_BOUNDS } from '../../data/nerStates';

export interface NerMapEntry extends NerStateMeta {
  tone: StateValidationTone;
  statusLabel: string;
}

interface NERMapProps {
  entries: NerMapEntry[];
  selectedStateId: string | null;
  onSelectState: (id: string | null) => void;
}

const TONE_COLOR: Record<StateValidationTone, string> = {
  pilot: '#10b981',
  pending: '#94a3b8',
};

export function NERMap({ entries, selectedStateId, onSelectState }: NERMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  // Keep the latest click handler without forcing the draw effect to re-run.
  const onSelectRef = useRef(onSelectState);
  onSelectRef.current = onSelectState;

  // Create / destroy the map exactly once.
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      zoomControl: false,
      attributionControl: true,
      minZoom: 5,
      maxZoom: 12,
    });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(map);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    map.fitBounds(NER_FIT_BOUNDS);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);
    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
    };
  }, []);

  // Redraw state geometry whenever entries / selection change.
  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer) return;
    layer.clearLayers();

    const drawState = (entry: NerMapEntry) => {
      const color = TONE_COLOR[entry.tone];
      const selected = entry.id === selectedStateId;
      const bounds: L.LatLngBoundsExpression = [
        [entry.adminBounds.minLat, entry.adminBounds.minLon],
        [entry.adminBounds.maxLat, entry.adminBounds.maxLon],
      ];
      const rect = L.rectangle(bounds, {
        color,
        weight: selected ? 2.5 : entry.isPilot ? 2 : 1,
        opacity: entry.tone === 'pilot' ? 0.9 : 0.5,
        fillColor: color,
        fillOpacity: selected ? 0.22 : entry.tone === 'pilot' ? 0.14 : 0.05,
      });
      rect.on('click', () => onSelectRef.current(entry.id));
      rect.addTo(layer);

      const labelColor = entry.tone === 'pilot' ? '#34d399' : '#cbd5e1';
      const sub = entry.tone === 'pilot' ? 'VALIDATED PILOT' : 'VALIDATION PENDING';
      const html = `
        <div style="transform:translate(-50%,-50%);pointer-events:auto;cursor:pointer;white-space:nowrap;
                    background:rgba(10,13,18,0.72);border:1px solid ${color}66;border-radius:5px;
                    padding:3px 7px;text-align:center;box-shadow:0 1px 6px rgba(0,0,0,0.5);">
          <div style="font-family:ui-monospace,monospace;font-size:11px;font-weight:700;color:${labelColor};letter-spacing:0.3px;">
            ${entry.name}
          </div>
          <div style="font-family:ui-monospace,monospace;font-size:7px;color:${labelColor};opacity:0.8;text-transform:uppercase;letter-spacing:0.7px;margin-top:1px;">
            ${sub}
          </div>
        </div>`;
      const icon = L.divIcon({ html, className: 'brk-state-label', iconSize: [0, 0] });
      const marker = L.marker(entry.center, { icon });
      marker.on('click', () => onSelectRef.current(entry.id));
      marker.addTo(layer);
    };

    // Pending first, pilot on top so its emerald reads clearly.
    entries.filter((e) => e.tone !== 'pilot').forEach(drawState);
    entries.filter((e) => e.tone === 'pilot').forEach(drawState);

    // Canonical pilot AOI (dashed emerald) — the exact modelled rectangle.
    const aoi = EAST_SIKKIM_PILOT_AOI;
    L.rectangle(
      [
        [aoi.minLat, aoi.minLon],
        [aoi.maxLat, aoi.maxLon],
      ],
      { color: '#22d3ee', weight: 1.5, dashArray: '5,4', fill: false, opacity: 0.9 },
    )
      .bindTooltip('East Sikkim pilot AOI — modelled extent', {
        direction: 'top',
        opacity: 0.95,
      })
      .addTo(layer);

    if (selectedStateId) {
      const sel = entries.find((e) => e.id === selectedStateId);
      if (sel) map.flyTo(sel.center, sel.zoom, { duration: 0.9 });
    } else {
      map.flyToBounds(NER_FIT_BOUNDS, { duration: 0.9 });
    }
  }, [entries, selectedStateId]);

  return (
    <div className="relative h-full w-full min-h-[420px] overflow-hidden rounded-xl border border-slate-800 bg-[#0b0f17]">
      <div ref={containerRef} className="h-full w-full min-h-[420px]" />
      {/* Legend */}
      <div className="pointer-events-none absolute bottom-4 left-4 z-[400] space-y-2 rounded-lg border border-slate-800 bg-slate-950/85 p-3 backdrop-blur">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          8 NER states
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-emerald-500" /> Validated pilot
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-slate-400/70" /> Validation pending
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
          <span className="inline-block h-2.5 w-3 border border-dashed border-cyan-400" /> Pilot AOI
        </div>
      </div>
    </div>
  );
}
