/**
 * API Service Layer for SIH 2026 Landslide Early Warning System
 * Interfaces cleanly with FastAPI backend endpoints.
 */

export type WarningLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'EXTREME';
export type ConfidenceLevel = 'LOW' | 'MEDIUM' | 'HIGH';

export interface TriggerDetail {
  trigger_exceeded: boolean;
  trigger_score: number;
  observed_intensity_mm_h: number;
  critical_intensity_mm_h: number;
  threshold_ratio: number;
  status: 'valid' | 'missing_data' | 'out_of_bounds';
}

export interface RiskData {
  susceptibility_score: number;
  current_trigger_score: number;
  forecast_trigger_score: number;
  exposure_score: number;
  final_risk_score: number;
  warning_level: WarningLevel;
  confidence: ConfidenceLevel;
  trigger_details: {
    current: TriggerDetail;
    forecast: TriggerDetail;
  };
}

export interface CurrentRiskResponse {
  location: [number, number];
  risk: RiskData;
}

export interface ForecastRiskResponse {
  location: [number, number];
  forecast_accumulation_mm: number;
  risk_forecast: RiskData;
}

export interface FeatureExplanation {
  feature: string;
  importance: number;
  description: string;
}

export interface CellExplainResponse {
  cell_id: string;
  explanation: {
    status: string;
    top_features: FeatureExplanation[];
    shap_values?: number[];
  };
}

export interface ExposedAsset {
  id: string;
  type: string;
  name: string;
  latitude: number;
  longitude: number;
  geometry?: string;
  risk_level?: WarningLevel;
}

export interface ExposureAlertsResponse {
  exposed_assets: ExposedAsset[];
}

export interface SystemHealthResponse {
  status: string;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export interface StateValidationReport {
  id?: string;
  state_id?: string;
  state: string;
  state_name?: string;
  processing_status?: string;
  validation_status?: string;
  rainfall_source?: string;
  rainfall_status: string;
  inventory_events: number;
  usable_events: number;
  spatial_quality: string;
  temporal_quality: string;
  dem_status: string;
  exposure_status: string;
  model_status: string;
  validation_metrics: Record<string, any>;
  risk_result?: any;
  overall_status: string;
  blocking_reasons: string[];
  error?: string | null;
}

class ApiService {
  private async fetchJson<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
      }
      return (await response.json()) as T;
    } catch (err: any) {
      console.warn(`[ApiService] Request to ${endpoint} failed:`, err.message);
      throw err;
    }
  }

  async checkHealth(): Promise<SystemHealthResponse> {
    return this.fetchJson<SystemHealthResponse>('/health');
  }

  async getCurrentRisk(lat: number, lon: number): Promise<CurrentRiskResponse> {
    return this.fetchJson<CurrentRiskResponse>(`/api/v1/risk/current?lat=${lat}&lon=${lon}`);
  }

  async getForecastRisk(lat: number, lon: number): Promise<ForecastRiskResponse> {
    return this.fetchJson<ForecastRiskResponse>(`/api/v1/risk/forecast?lat=${lat}&lon=${lon}`);
  }

  async getCellExplanation(cellId: string): Promise<CellExplainResponse> {
    return this.fetchJson<CellExplainResponse>(`/api/v1/cell/${cellId}/explain`);
  }

  async getExposureAlerts(): Promise<ExposureAlertsResponse> {
    return this.fetchJson<ExposureAlertsResponse>('/api/v1/exposure/alerts');
  }

  async getValidationStatus(): Promise<StateValidationReport[]> {
    return this.fetchJson<StateValidationReport[]>('/api/v1/validation/status');
  }
}

export const apiService = new ApiService();
