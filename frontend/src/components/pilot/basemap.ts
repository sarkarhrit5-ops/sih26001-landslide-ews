/**
 * Leaflet glue for the shared basemap. Split from mapTheme.ts so the theme logic
 * stays importable without Leaflet (and therefore unit-testable in node).
 *
 * Creates the ONE key-free tile layer used by every map and switches its theme
 * class in place. Switching does not recreate the layer, so tiles already in the
 * browser cache are reused and the change is a CSS cross-fade rather than a
 * reload — and no map geometry, AOI rectangle, marker, label or control is
 * touched in the process.
 */
import L from 'leaflet';
import {
  BASEMAP_ATTRIBUTION,
  BASEMAP_MAX_ZOOM,
  BASEMAP_TILE_URL,
  applyBasemapThemeClass,
  basemapClassName,
} from './mapTheme';
import type { MapTheme } from './mapTheme';

/** The basemap layer, already themed and attributed. Caller adds it to the map. */
export function createBasemapLayer(theme: MapTheme): L.TileLayer {
  return L.tileLayer(BASEMAP_TILE_URL, {
    maxZoom: BASEMAP_MAX_ZOOM,
    attribution: BASEMAP_ATTRIBUTION,
    className: basemapClassName(theme),
  });
}

/**
 * Re-theme an existing layer. A no-op when the layer is not mounted yet, so an
 * early theme change during mount cannot throw.
 */
export function setBasemapTheme(layer: L.TileLayer | null | undefined, theme: MapTheme): void {
  const container = layer?.getContainer?.();
  if (!container) return;
  applyBasemapThemeClass(container.classList, theme);
}
