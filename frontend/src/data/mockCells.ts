import type { WarningLevel } from '../services/api';

export interface GridCell {
  id: string;
  name: string;
  lat: number;
  lon: number;
  bounds: [[number, number], [number, number]]; // [[minLat, minLon], [maxLat, maxLon]]
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

// East Sikkim Pilot AOI Cells (27.0 N to 28.1 N, 88.0 E to 88.9 E)
export const EAST_SIKKIM_CELLS: GridCell[] = [
  {
    id: 'cell_gangtok_central',
    name: 'Gangtok Central & STNM Corridor',
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
  regionName: 'East Sikkim Pilot AOI',
  boundsStr: '27.0°N - 28.1°N, 88.0°E - 88.9°E',
  overallRisk: 'HIGH' as WarningLevel,
  activeHighRiskCellsCount: 3,
  totalMonitoredCells: 6,
  totalExposedAssets: 18,
  demSource: 'Copernicus 30m GLO-30 DEM',
  rainfallStatusNote: 'Satellite rainfall unavailable: NASA Earthdata authentication is required for live precipitation retrieval.'
};
