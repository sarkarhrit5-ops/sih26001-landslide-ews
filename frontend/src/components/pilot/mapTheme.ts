/**
 * Basemap THEME logic, shared by the NER overview map and all four pilot maps.
 *
 * Why this module exists
 *   The five Leaflet maps each built their own tile layer inline. The theme
 *   toggle has to behave identically on all of them, so the decisions live here
 *   once: which tile endpoint, which attribution, and which CSS class carries
 *   the dark treatment. Everything here is pure — no Leaflet, no React, no DOM
 *   globals — so it is testable under `node --test` like pilotMapCells.ts.
 *
 * Why one tile endpoint and a CSS filter rather than two providers
 *   Dark Mode is produced by filtering the SAME OpenStreetMap tiles that Light
 *   Mode draws. That keeps the basemap key-free (the previous Carto dark style
 *   started demanding a client key and stamped "API KEY REQUIRED" across every
 *   map), keeps attribution honest and identical in both modes, and guarantees
 *   the dark mode cannot silently fail to load tiles that the light mode can.
 *   No map data, geometry, marker, label or overlay is affected: the filter is
 *   scoped to the tile layer's own container, not to Leaflet's overlay panes.
 */

export type MapTheme = 'dark' | 'light';

/** Dark is the default so the existing BhūRaksha console design is preserved. */
export const DEFAULT_MAP_THEME: MapTheme = 'dark';

export const MAP_THEMES: readonly MapTheme[] = ['dark', 'light'] as const;

/** Key-free tiles. Both themes read from here; only the CSS treatment differs. */
export const BASEMAP_TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

/** Required attribution for the tiles above — rendered in both themes. */
export const BASEMAP_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export const BASEMAP_MAX_ZOOM = 19;

/** Marker class on the tile container; carries the cross-fade transition. */
export const BASEMAP_CLASS = 'brk-basemap';

export const BASEMAP_THEME_CLASS: Record<MapTheme, string> = {
  dark: 'brk-basemap--dark',
  light: 'brk-basemap--light',
};

export const MAP_THEME_LABEL: Record<MapTheme, string> = {
  dark: 'Dark',
  light: 'Light',
};

/** Class string for a freshly created tile layer. */
export function basemapClassName(theme: MapTheme): string {
  return `${BASEMAP_CLASS} ${BASEMAP_THEME_CLASS[theme]}`;
}

/** The theme a toggle would switch to from `theme`. */
export function otherMapTheme(theme: MapTheme): MapTheme {
  return theme === 'dark' ? 'light' : 'dark';
}

export function isMapTheme(value: unknown): value is MapTheme {
  return value === 'dark' || value === 'light';
}

/** Minimal shape of DOMTokenList that the swap needs (keeps this module DOM-free). */
export interface ClassListLike {
  add(...tokens: string[]): void;
  remove(...tokens: string[]): void;
}

/**
 * Put the container into exactly one theme class. Both classes are removed
 * first, so repeated calls can never leave two treatments stacked.
 */
export function applyBasemapThemeClass(classList: ClassListLike, theme: MapTheme): void {
  classList.remove(BASEMAP_THEME_CLASS.dark, BASEMAP_THEME_CLASS.light);
  classList.add(BASEMAP_CLASS, BASEMAP_THEME_CLASS[theme]);
}
