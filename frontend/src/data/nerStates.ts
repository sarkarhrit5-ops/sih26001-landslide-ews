/**
 * BhūRaksha — North Eastern Region (NER) state metadata.
 *
 * IMPORTANT — provenance of the numbers in this file:
 *   * Administrative bounding boxes are copied verbatim from the backend single
 *     source of truth, app/core/config_states.py -> NER_STATES_CONFIG. They are
 *     deliberately loose over-approximations of each state's extent (used by the
 *     backend for its 8-state sweep), NOT precise borders.
 *   * EAST_SIKKIM_PILOT_AOI matches app/core/config_states.EAST_SIKKIM_PILOT_AOI
 *     — the single canonical AOI the pilot model was actually trained on.
 *   * State capitals are real, well-known administrative facts.
 *   * `center` is the geometric centroid of the administrative box (a derived
 *     display coordinate for map labelling), and `zoom` is a UI display setting.
 *
 * This file contains NO risk, rainfall, susceptibility or model-output numbers.
 * All such values are fetched live from the backend, which refuses rather than
 * fabricates when an input is missing.
 */

export type StateValidationTone = 'pilot' | 'pending';

export interface AdminBounds {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
}

export interface NerStateMeta {
  /** Matches the backend state_id (e.g. "sikkim", "arunachal_pradesh"). */
  id: string;
  name: string;
  capital: string;
  /** Loose administrative bbox from config_states.py (over-approximation). */
  adminBounds: AdminBounds;
  /** Centroid of adminBounds, [lat, lon] — a derived display coordinate. */
  center: [number, number];
  /** UI display zoom for fly-to. */
  zoom: number;
  isPilot: boolean;
}

/** Canonical pilot AOI — mirrors config_states.EAST_SIKKIM_PILOT_AOI exactly. */
export const EAST_SIKKIM_PILOT_AOI: AdminBounds = {
  minLat: 27.0,
  maxLat: 28.1,
  minLon: 88.0,
  maxLon: 88.9,
};

/**
 * Canonical Assam pilot AOI — mirrors config_states.ASSAM_PILOT_AOI exactly
 * ("Assam pilot AOI (Guwahati-Kamrup + western Karbi Anglong)"). This is the
 * single AOI the Assam pilot model was actually trained on; it is deliberately
 * narrower than Assam's loose administrative bbox above.
 */
export const ASSAM_PILOT_AOI: AdminBounds = {
  minLat: 25.6,
  maxLat: 26.6,
  minLon: 91.3,
  maxLon: 93.7,
};

/**
 * Canonical Arunachal Pradesh pilot AOI — mirrors config_states.ARUNACHAL_PILOT_AOI
 * exactly ("Arunachal Pradesh pilot AOI (central Subansiri-Siang belt)"). This is the
 * single AOI the Arunachal pilot model was actually trained on; it is deliberately
 * narrower than Arunachal's loose administrative bbox above.
 */
export const ARUNACHAL_PILOT_AOI: AdminBounds = {
  minLat: 26.5,
  maxLat: 27.99,
  minLon: 92.0,
  maxLon: 94.5,
};

/**
 * Canonical Meghalaya pilot AOI — mirrors config_states.MEGHALAYA_PILOT_AOI exactly
 * ("Meghalaya pilot AOI (East Khasi + Jaintia Hills belt)"). This is the single AOI
 * the Meghalaya pilot model was actually trained on; it is deliberately narrower than
 * Meghalaya's loose administrative bbox above.
 */
export const MEGHALAYA_PILOT_AOI: AdminBounds = {
  minLat: 25.0,
  maxLat: 25.99,
  minLon: 91.0,
  maxLon: 92.8,
};

function centroid(b: AdminBounds): [number, number] {
  return [(b.minLat + b.maxLat) / 2, (b.minLon + b.maxLon) / 2];
}

const RAW_STATES: Omit<NerStateMeta, 'center'>[] = [
  {
    id: 'sikkim',
    name: 'Sikkim',
    capital: 'Gangtok',
    adminBounds: { minLat: 27.0, maxLat: 28.2, minLon: 88.0, maxLon: 89.0 },
    zoom: 9,
    isPilot: true,
  },
  {
    id: 'arunachal_pradesh',
    name: 'Arunachal Pradesh',
    capital: 'Itanagar',
    adminBounds: { minLat: 26.5, maxLat: 29.5, minLon: 91.5, maxLon: 97.5 },
    zoom: 7,
    isPilot: false,
  },
  {
    id: 'assam',
    name: 'Assam',
    capital: 'Dispur',
    adminBounds: { minLat: 24.0, maxLat: 28.0, minLon: 89.5, maxLon: 96.0 },
    zoom: 7,
    isPilot: false,
  },
  {
    id: 'manipur',
    name: 'Manipur',
    capital: 'Imphal',
    adminBounds: { minLat: 23.8, maxLat: 25.7, minLon: 93.0, maxLon: 94.8 },
    zoom: 8,
    isPilot: false,
  },
  {
    id: 'meghalaya',
    name: 'Meghalaya',
    capital: 'Shillong',
    adminBounds: { minLat: 25.0, maxLat: 26.1, minLon: 89.8, maxLon: 92.8 },
    zoom: 8,
    isPilot: false,
  },
  {
    id: 'mizoram',
    name: 'Mizoram',
    capital: 'Aizawl',
    adminBounds: { minLat: 21.9, maxLat: 24.5, minLon: 92.2, maxLon: 93.4 },
    zoom: 8,
    isPilot: false,
  },
  {
    id: 'nagaland',
    name: 'Nagaland',
    capital: 'Kohima',
    adminBounds: { minLat: 25.2, maxLat: 27.0, minLon: 93.3, maxLon: 95.3 },
    zoom: 8,
    isPilot: false,
  },
  {
    id: 'tripura',
    name: 'Tripura',
    capital: 'Agartala',
    adminBounds: { minLat: 22.9, maxLat: 24.5, minLon: 91.1, maxLon: 92.4 },
    zoom: 8,
    isPilot: false,
  },
];

export const NER_STATES: NerStateMeta[] = RAW_STATES.map((s) => ({
  ...s,
  center: centroid(s.adminBounds),
}));

/** Padded bounds enclosing all 8 states, for initial map framing. */
export const NER_FIT_BOUNDS: [[number, number], [number, number]] = [
  [21.6, 87.7],
  [29.7, 97.7],
];

export function getStateMeta(stateId: string | undefined): NerStateMeta | undefined {
  if (!stateId) return undefined;
  return NER_STATES.find((s) => s.id === stateId);
}

/**
 * Collapses the backend's overall_status vocabulary into the two display tones
 * this product surfaces: the Sikkim validated pilot vs. everything still pending.
 */
export function validationTone(overallStatus: string | undefined): StateValidationTone {
  return (overallStatus || '').toUpperCase().includes('VALIDATED_PILOT') ? 'pilot' : 'pending';
}

/**
 * The four pilot consoles, keyed by the SAME token the backend uses in its route
 * paths (/api/v1/predict/<key>/map, /api/v1/validation/<key>/evidence). Note the
 * key is the route token, which for Arunachal is "arunachal" — deliberately not
 * the backend state_id "arunachal_pradesh" (that is carried separately as
 * `stateId`, and the two must not be conflated).
 */
export type PilotStateKey = 'sikkim' | 'assam' | 'arunachal' | 'meghalaya';

export interface PilotConsoleConfig {
  /** Route token used to build the backend endpoint paths. */
  key: PilotStateKey;
  /** Backend state_id (config_states) — NOT always equal to `key`. */
  stateId: string;
  name: string;
  /**
   * Canonical pilot AOI, mirroring the corresponding config_states constant.
   * This is the modelled extent and is deliberately NOT the full state.
   */
  aoi: AdminBounds;
  /** Initial Leaflet zoom for this console — preserves each map's current framing. */
  zoom: number;
  /** Tooltip on the AOI rectangle — preserves each map's current wording. */
  aoiTooltip: string;
  /**
   * Tooltip on a cell the backend returned as UNAVAILABLE. Sikkim needs only
   * terrain; the three WorldCover pilots additionally need land cover, so their
   * wording names both inputs. Neither version claims a risk value.
   */
  unavailableCellTooltip: string;
}

/**
 * Single source of truth for pilot endpoint + AOI selection. Every console reads
 * its AOI, framing and endpoint token from here rather than hard-coding them, so
 * a new pilot is one entry and no per-state branch. Contains no risk, rainfall or
 * model numbers.
 */
export const PILOT_REGISTRY: Record<PilotStateKey, PilotConsoleConfig> = {
  sikkim: {
    key: 'sikkim',
    stateId: 'sikkim',
    name: 'Sikkim',
    aoi: EAST_SIKKIM_PILOT_AOI,
    zoom: 9,
    aoiTooltip: 'East Sikkim pilot AOI — modelled extent',
    unavailableCellTooltip: 'No prediction (terrain unavailable at this cell)',
  },
  assam: {
    key: 'assam',
    stateId: 'assam',
    name: 'Assam',
    aoi: ASSAM_PILOT_AOI,
    zoom: 8,
    aoiTooltip: 'Assam pilot AOI — modelled extent',
    unavailableCellTooltip:
      'No prediction (terrain or land cover unavailable at this cell)',
  },
  arunachal: {
    key: 'arunachal',
    stateId: 'arunachal_pradesh',
    name: 'Arunachal Pradesh',
    aoi: ARUNACHAL_PILOT_AOI,
    zoom: 8,
    aoiTooltip: 'Arunachal Pradesh pilot AOI — modelled extent',
    unavailableCellTooltip:
      'No prediction (terrain or land cover unavailable at this cell)',
  },
  meghalaya: {
    key: 'meghalaya',
    stateId: 'meghalaya',
    name: 'Meghalaya',
    aoi: MEGHALAYA_PILOT_AOI,
    zoom: 8,
    aoiTooltip: 'Meghalaya pilot AOI — modelled extent',
    unavailableCellTooltip:
      'No prediction (terrain or land cover unavailable at this cell)',
  },
};

export const PILOT_STATE_KEYS: PilotStateKey[] = ['sikkim', 'assam', 'arunachal', 'meghalaya'];

/** Throws on an unknown key rather than silently falling back to another state's AOI. */
export function getPilotConfig(key: string): PilotConsoleConfig {
  const config = (PILOT_REGISTRY as Record<string, PilotConsoleConfig | undefined>)[key];
  if (!config) {
    throw new Error(
      `Unknown pilot state key "${key}" — expected one of ${PILOT_STATE_KEYS.join(', ')}`,
    );
  }
  return config;
}
