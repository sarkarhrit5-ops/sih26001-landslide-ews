/**
 * Assam pilot map — the exact modelled AOI rectangle, the real NASA GLC landslide
 * positives inside it, and (when available) the model's predicted per-grid-cell
 * susceptibility zones. Event points and predicted cells are both supplied by the
 * parent; this component invents nothing. Points are coloured only by their honest
 * spatial-uncertainty class (precise <5 km vs approximate ≥5 km); predicted cells
 * are coloured by the model's warning class and rendered only for cells the backend
 * actually scored (UNAVAILABLE cells — terrain or ESA WorldCover land cover missing
 * — are drawn as hollow, never as a fabricated low-risk zone).
 *
 * This is a sibling of SikkimMap: same rendering contract, only the AOI constant and
 * two labels differ. SikkimMap is left untouched.
 */
import { useEffect, useRef } from 'react';
import L from 'leaflet';
import type { LandslideEvent, AssamPredictionCell } from '../../services/api';
import { ASSAM_PILOT_AOI } from '../../data/nerStates';

interface AssamMapProps {
  events: LandslideEvent[];
  /** null while loading; empty array only after a successful empty response. */
  loaded: boolean;
  /** Predicted grid cells; omitted/empty when no prediction is available. */
  predictedCells?: AssamPredictionCell[];
}

/** Fill colour per model warning class (matches the app's risk palette). */
const RISK_FILL: Record<string, string> = {
  LOW: '#22c55e',
  MEDIUM: '#eab308',
  HIGH: '#f97316',
  EXTREME: '#ef4444',
};

const AOI = ASSAM_PILOT_AOI;
const AOI_CENTER: [number, number] = [(AOI.minLat + AOI.maxLat) / 2, (AOI.minLon + AOI.maxLon) / 2];

export function AssamMap({ events, loaded, predictedCells }: AssamMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, {
      zoomControl: false,
      attributionControl: false,
      minZoom: 7,
      maxZoom: 14,
    });
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      maxZoom: 18,
      subdomains: 'abcd',
    }).addTo(map);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    // The Assam AOI is wider in longitude (~2.4°) than Sikkim's, so it frames at a
    // slightly lower zoom before the event-driven fitBounds takes over.
    map.setView(AOI_CENTER, 8);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);
    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer) return;
    layer.clearLayers();

    // The modelled AOI rectangle.
    const aoiRect = L.rectangle(
      [
        [AOI.minLat, AOI.minLon],
        [AOI.maxLat, AOI.maxLon],
      ],
      { color: '#22d3ee', weight: 1.5, dashArray: '5,4', fill: true, fillColor: '#22d3ee', fillOpacity: 0.04, opacity: 0.9 },
    ).bindTooltip('Assam pilot AOI — modelled extent', { direction: 'top', opacity: 0.95 });
    aoiRect.addTo(layer);

    // Predicted susceptibility zones (drawn under the GLC points). Only cells the
    // backend actually scored get a filled colour; UNAVAILABLE cells are drawn as
    // a faint hollow outline so the gap is visible rather than implied safe.
    (predictedCells ?? []).forEach((cell) => {
      const b = cell.bbox;
      if (cell.status === 'OK' && cell.risk_class && cell.susceptibility_probability != null) {
        const fill = RISK_FILL[cell.risk_class] ?? '#64748b';
        const rect = L.rectangle(
          [
            [b.min_lat, b.min_lon],
            [b.max_lat, b.max_lon],
          ],
          { color: fill, weight: 0.5, opacity: 0.35, fill: true, fillColor: fill, fillOpacity: 0.32 },
        );
        const prob = (cell.susceptibility_probability * 100).toFixed(1);
        const parts = [
          `<div style="font-family:ui-monospace,monospace;font-size:11px;color:#e2e8f0;">`,
          `<div style="font-weight:700;color:${fill};text-transform:uppercase;letter-spacing:0.5px;">${cell.risk_class}</div>`,
          `<div style="margin-top:2px;">susceptibility ${prob}%</div>`,
          `<div style="margin-top:2px;color:#94a3b8;">${cell.latitude.toFixed(3)}, ${cell.longitude.toFixed(3)}</div>`,
          cell.exceeds_decision_threshold
            ? `<div style="margin-top:2px;color:${fill};">≥ decision threshold</div>`
            : '',
          `</div>`,
        ];
        rect.bindPopup(parts.join(''));
        rect.addTo(layer);
      } else {
        const rect = L.rectangle(
          [
            [b.min_lat, b.min_lon],
            [b.max_lat, b.max_lon],
          ],
          { color: '#475569', weight: 0.4, opacity: 0.25, fill: false, dashArray: '2,3' },
        );
        rect.bindTooltip('No prediction (terrain or land cover unavailable at this cell)', { direction: 'top', opacity: 0.9 });
        rect.addTo(layer);
      }
    });

    // Real GLC positives.
    events.forEach((ev) => {
      const precise = ev.spatial_uncertainty === 'precise_lt_5km';
      const color = precise ? '#34d399' : '#f59e0b';
      const marker = L.circleMarker([ev.latitude, ev.longitude], {
        radius: precise ? 5 : 4,
        color,
        weight: 1.2,
        fillColor: color,
        fillOpacity: 0.55,
      });
      const parts = [
        `<div style="font-family:ui-monospace,monospace;font-size:11px;color:#e2e8f0;">`,
        `<div style="font-weight:700;color:${color};">${ev.event_date || 'undated'}</div>`,
        ev.event_title ? `<div style="margin-top:2px;">${ev.event_title}</div>` : '',
        `<div style="margin-top:3px;color:#94a3b8;">`,
        `${ev.latitude.toFixed(3)}, ${ev.longitude.toFixed(3)}`,
        `</div>`,
        ev.landslide_trigger ? `<div style="color:#94a3b8;">trigger: ${ev.landslide_trigger}</div>` : '',
        ev.location_accuracy ? `<div style="color:#94a3b8;">accuracy: ${ev.location_accuracy}</div>` : '',
        `<div style="margin-top:3px;color:${color};text-transform:uppercase;font-size:9px;letter-spacing:0.5px;">`,
        precise ? 'precise <5 km' : 'approximate ≥5 km',
        `</div></div>`,
      ];
      marker.bindPopup(parts.join(''));
      marker.addTo(layer);
    });

    if (loaded && events.length > 0) {
      const grp = L.featureGroup(
        events.map((e) => L.marker([e.latitude, e.longitude])),
      );
      map.fitBounds(grp.getBounds().pad(0.15), { animate: false });
    }
  }, [events, loaded, predictedCells]);

  // Only advertise the predicted-risk swatches when the backend actually scored
  // at least one cell, so the legend never implies a prediction we don't have.
  const scoredClasses = new Set(
    (predictedCells ?? [])
      .filter((c) => c.status === 'OK' && c.risk_class)
      .map((c) => c.risk_class as string),
  );
  const hasUnavailable = (predictedCells ?? []).some((c) => c.status !== 'OK');
  const RISK_ORDER: Array<'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME'> = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME'];

  return (
    <div className="relative h-full w-full min-h-[440px] overflow-hidden rounded-xl border border-slate-800 bg-[#0b0f17]">
      <div ref={containerRef} className="h-full w-full min-h-[440px]" />
      <div className="pointer-events-none absolute bottom-4 left-4 z-[400] space-y-2 rounded-lg border border-slate-800 bg-slate-950/85 p-3 backdrop-blur">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-400">
          NASA GLC positives
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" /> Precise &lt;5 km
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-amber-400" /> Approximate ≥5 km
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
          <span className="inline-block h-2.5 w-3 border border-dashed border-cyan-400" /> Pilot AOI
        </div>
        {scoredClasses.size > 0 && (
          <>
            <div className="mt-1 border-t border-slate-800 pt-2 font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-400">
              Predicted risk class
            </div>
            {RISK_ORDER.filter((rc) => scoredClasses.has(rc)).map((rc) => (
              <div key={rc} className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
                <span
                  className="inline-block h-2.5 w-3 rounded-sm"
                  style={{ backgroundColor: RISK_FILL[rc], opacity: 0.7 }}
                />{' '}
                {rc.charAt(0) + rc.slice(1).toLowerCase()}
              </div>
            ))}
            {hasUnavailable && (
              <div className="flex items-center gap-2 text-[10px] font-mono text-slate-300">
                <span className="inline-block h-2.5 w-3 border border-dashed border-slate-500" /> No prediction
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
