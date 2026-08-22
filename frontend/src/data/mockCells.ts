import type { WarningLevel } from '../services/api';

export interface ValidationCheckitem {
  label: string;
  completed: boolean;
}

export interface NERState {
  id: string;
  name: string;
  capital: string;
  lat: number;
  lon: number;
  zoom: number;
  bounds: [[number, number], [number, number]];
  status: 'VALIDATED_PILOT' | 'VALIDATION_PENDING';
  statusLabel: 'VALIDATED PILOT' | 'VALIDATION PENDING';
  coverageArea: string;
  hasValidatedPilot: boolean;
  checklist: ValidationCheckitem[];
}

export interface GridCell {
  id: string;
  name: string;
  stateId: string;
  lat: number;
  lon: number;
  bounds: [[number, number], [number, number]];
  susceptibility: number;
  currentRain: number; // mm/24h
  forecastRain: number; // mm/72h
  slope: number; // deg
  elevation: number; // m
  roughness: number; // m
  aspect: number; // deg
  warningLevel: WarningLevel;
  finalRisk: number;
  exposedAssetsCount: number;
  assets: string[];
}

const PENDING_CHECKLIST: ValidationCheckitem[] = [
  { label: 'NER geographic coverage', completed: true },
  { label: 'Landslide inventory validation', completed: false },
  { label: 'Terrain/data validation', completed: false },
  { label: 'Rainfall integration', completed: false },
  { label: 'Model validation', completed: false }
];

const VALIDATED_CHECKLIST: ValidationCheckitem[] = [
  { label: 'NER geographic coverage', completed: true },
  { label: 'Landslide inventory validated', completed: true },
  { label: 'Terrain pipeline validated', completed: true },
  { label: 'Susceptibility model validated', completed: true },
  { label: 'Risk pipeline validated', completed: true }
];

// All 8 North Eastern Region (NER) States
export const NER_STATES: NERState[] = [
  {
    id: 'sikkim',
    name: 'Sikkim',
    capital: 'Gangtok',
    lat: 27.4,
    lon: 88.5,
    zoom: 9.5,
    bounds: [[27.0, 88.0], [28.1, 88.9]],
    status: 'VALIDATED_PILOT',
    statusLabel: 'VALIDATED PILOT',
    coverageArea: 'East Sikkim',
    hasValidatedPilot: true,
    checklist: VALIDATED_CHECKLIST
  },
  {
    id: 'arunachal_pradesh',
    name: 'Arunachal Pradesh',
    capital: 'Itanagar',
    lat: 28.2,
    lon: 94.5,
    zoom: 7.5,
    bounds: [[26.5, 91.5], [29.5, 97.5]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  },
  {
    id: 'assam',
    name: 'Assam',
    capital: 'Dispur / Guwahati',
    lat: 26.2,
    lon: 92.5,
    zoom: 7.5,
    bounds: [[24.1, 89.7], [28.0, 96.0]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  },
  {
    id: 'manipur',
    name: 'Manipur',
    capital: 'Imphal',
    lat: 24.8,
    lon: 93.9,
    zoom: 8.5,
    bounds: [[23.8, 93.0], [25.7, 94.8]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  },
  {
    id: 'meghalaya',
    name: 'Meghalaya',
    capital: 'Shillong',
    lat: 25.5,
    lon: 91.5,
    zoom: 8.5,
    bounds: [[25.0, 89.8], [26.1, 92.8]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  },
  {
    id: 'mizoram',
    name: 'Mizoram',
    capital: 'Aizawl',
    lat: 23.1,
    lon: 92.8,
    zoom: 8.5,
    bounds: [[21.9, 92.2], [24.5, 93.4]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  },
  {
    id: 'nagaland',
    name: 'Nagaland',
    capital: 'Kohima',
    lat: 26.1,
    lon: 94.5,
    zoom: 8.5,
    bounds: [[25.2, 93.3], [27.0, 95.3]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  },
  {
    id: 'tripura',
    name: 'Tripura',
    capital: 'Agartala',
    lat: 23.8,
    lon: 91.3,
    zoom: 8.5,
    bounds: [[22.9, 91.1], [24.5, 92.4]],
    status: 'VALIDATION_PENDING',
    statusLabel: 'VALIDATION PENDING',
    coverageArea: 'Entire State (Validation Pending)',
    hasValidatedPilot: false,
    checklist: PENDING_CHECKLIST
  }
];

export const NER_BOUNDS: [[number, number], [number, number]] = [[21.8, 87.8], [29.5, 97.4]];

// Validated East Sikkim Pilot Cells
export const EAST_SIKKIM_CELLS: GridCell[] = [
  {
    id: 'cell_gangtok_central',
    name: 'Gangtok Central & STNM Corridor',
    stateId: 'sikkim',
    lat: 27.33,
    lon: 88.61,
    bounds: [[27.30, 88.58], [27.36, 88.64]],
    susceptibility: 0.72,
    currentRain: 55.0,
    forecastRain: 110.0,
    slope: 38.5,
    elevation: 1650,
    roughness: 68.2,
    aspect: 195.0,
    warningLevel: 'HIGH',
    finalRisk: 0.78,
    exposedAssetsCount: 4,
    assets: ['NH10 Transport Highway', 'STNM Central Hospital', 'Indira Bypass Road', 'Gangtok Urban Settlement']
  },
  {
    id: 'cell_dikchu_gorge',
    name: 'Dikchu River Basin & Slope Zone',
    stateId: 'sikkim',
    lat: 27.39,
    lon: 88.52,
    bounds: [[27.36, 88.49], [27.42, 88.55]],
    susceptibility: 0.88,
    currentRain: 78.0,
    forecastRain: 145.0,
    slope: 44.2,
    elevation: 850,
    roughness: 94.5,
    aspect: 220.0,
    warningLevel: 'EXTREME',
    finalRisk: 0.92,
    exposedAssetsCount: 3,
    assets: ['Dikchu Hydroelectric Feeder Bridge', 'North Sikkim Highway Segment', 'Local Power Substation']
  },
  {
    id: 'cell_pakyong_slope',
    name: 'Pakyong Airport Approach Slopes',
    stateId: 'sikkim',
    lat: 27.24,
    lon: 88.59,
    bounds: [[27.21, 88.56], [27.27, 88.62]],
    susceptibility: 0.61,
    currentRain: 32.0,
    forecastRain: 65.0,
    slope: 29.0,
    elevation: 1420,
    roughness: 42.1,
    aspect: 140.0,
    warningLevel: 'MEDIUM',
    finalRisk: 0.54,
    exposedAssetsCount: 2,
    assets: ['Pakyong Airport Access Road', 'Pakyong School Complex']
  },
  {
    id: 'cell_singtam_rangpo',
    name: 'Singtam - Rangpo Teesta Corridor',
    stateId: 'sikkim',
    lat: 27.15,
    lon: 88.50,
    bounds: [[27.12, 88.47], [27.18, 88.53]],
    susceptibility: 0.52,
    currentRain: 28.0,
    forecastRain: 50.0,
    slope: 24.5,
    elevation: 350,
    roughness: 28.4,
    aspect: 175.0,
    warningLevel: 'MEDIUM',
    finalRisk: 0.46,
    exposedAssetsCount: 5,
    assets: ['NH10 Teesta Valley Corridor', 'Rangpo Border Checkpost', 'Teesta Suspension Bridge', 'Singtam Market Road']
  },
  {
    id: 'cell_mangan_south',
    name: 'Mangan South Ridge Approach',
    stateId: 'sikkim',
    lat: 27.50,
    lon: 88.53,
    bounds: [[27.47, 88.50], [27.53, 88.56]],
    susceptibility: 0.79,
    currentRain: 64.0,
    forecastRain: 125.0,
    slope: 41.0,
    elevation: 1280,
    roughness: 82.0,
    aspect: 210.0,
    warningLevel: 'HIGH',
    finalRisk: 0.81,
    exposedAssetsCount: 3,
    assets: ['North Sikkim Highway Corridor', 'Mangan District Hospital', 'Zilla Panchayat Building']
  },
  {
    id: 'cell_nathula_pass',
    name: 'Nathula Alpine Pass Corridor',
    stateId: 'sikkim',
    lat: 27.38,
    lon: 88.82,
    bounds: [[27.35, 88.79], [27.41, 88.85]],
    susceptibility: 0.35,
    currentRain: 15.0,
    forecastRain: 30.0,
    slope: 18.0,
    elevation: 4310,
    roughness: 35.0,
    aspect: 90.0,
    warningLevel: 'LOW',
    finalRisk: 0.28,
    exposedAssetsCount: 1,
    assets: ['Jawaharlal Nehru Road']
  }
];

export const REGIONAL_SUMMARY = {
  systemCoverage: 'Northeast India — 8 States',
  modelValidation: 'East Sikkim — Validated Pilot',
  overallRisk: 'HIGH' as WarningLevel,
  activeHighRiskCellsCount: 3,
  totalMonitoredCells: 6,
  totalExposedAssets: 18,
  demSource: 'Copernicus 30m GLO-30 DEM',
  rainfallStatusNote: 'Satellite Rainfall Unavailable — NASA Earthdata authentication required'
};
