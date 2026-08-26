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
