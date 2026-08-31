/**
 * API Service Layer for BhūRaksha — NER Landslide Early Warning System (SIH 2026)
 * Interfaces cleanly with FastAPI backend endpoints.
 *
 * Data-integrity contract: this layer never fabricates or defaults values. If an
 * endpoint refuses (e.g. HTTP 503 DATA_UNAVAILABLE), fetchJson throws and the
 * caller is expected to surface an explicit "unavailable" state to the user
 * rather than substituting placeholder data.
 */

// Endpoint construction for the four pilot consoles lives in ONE pure module so
// the paths are unit-testable without a browser. The reverse edge is type-only
// (pilotMapCells imports only `import type` from here), so there is no runtime
// import cycle.
import { pilotGridEndpoint, pilotMapEndpoint } from '../components/pilot/pilotMapCells';

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

/**
 * Base URL that every backend request below is prefixed with (see fetchJson:
 * `${API_BASE_URL}${endpoint}`). Each endpoint string already carries its own
 * "/api/v1/..." (or "/health") path, so this only controls the origin.
 *
 * - Deployed / cross-origin: set VITE_API_URL (e.g. "https://api.example.com") and
 *   all requests become VITE_API_URL + "/api/v1/..." — no route strings change.
 * - Local development: leave VITE_API_URL unset. The fallback is an empty string, so
 *   requests stay relative ("/api/v1/...", "/health") and the Vite dev server proxies
 *   them to the FastAPI backend (vite.config.ts → server.proxy for "/api" and
 *   "/health"). This is the existing local "/api" development path, unchanged.
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

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

/**
 * Geographic bounding box, as returned verbatim by the backend
 * (config_states.get_pilot_aoi_bounds).
 */
export interface AoiBounds {
  min_lat: number;
  max_lat: number;
  min_lon: number;
  max_lon: number;
}

/** The six validation metrics computed by ml_pipeline.compute_metrics(). */
export interface ValidationMetricSet {
  'PR-AUC': number;
  'ROC-AUC': number;
  Precision: number;
  Recall: number;
  F1: number;
  'False Alarm Rate': number;
}

/**
 * model_comparison is nested: holdout -> feature_set -> model_name -> metrics.
 * e.g. model_comparison['temporal_holdout']['static_plus_rainfall']['LightGBM'].
 */
export type ModelComparison = Record<
  string,
  Record<string, Record<string, ValidationMetricSet>>
>;

export interface SikkimEvidenceMetrics {
  validation_metrics: ValidationMetricSet;
  metrics_source?: string;
  status?: string;
  primary_model?: string;
  primary_evaluation?: string;
  feature_set?: string;
  holdout_details?: {
    spatial_holdout?: string;
    temporal_holdout?: string;
    decision_threshold?: number;
  };
  sample_counts?: {
    total_samples: number;
    positive_samples: number;
    negative_samples: number;
    primary_train_samples: number;
    primary_test_samples: number;
    primary_train_positives: number;
    primary_test_positives: number;
  };
  model_comparison?: ModelComparison;
  model_decision?: {
    final_recommendation: string;
    justification_reasons: string[];
  };
  generated_at?: string;
}

export interface SikkimFeatureSchema {
  feature_set_name?: string;
  feature_names: string[];
  feature_order?: number[];
  n_features: number;
  dtype?: Record<string, string>;
  meaning?: Record<string, string>;
  target_column?: string;
}

export interface SikkimProvenance {
  aoi: AoiBounds;
  glc_source?: string;
  glc_event_count: number;
  sample_counts?: {
    raw_catalog_events_in_aoi: number;
    deduplicated_positive_events: number;
    negative_samples: number;
    total_samples: number;
    independent_event_dates: number;
    pct_events_spatial_uncertainty_ge_5km: number;
  };
  rainfall_source?: string;
  dem_source?: string;
  terrain_derivative_method?: string;
  exposure_source?: string;
  model_type?: string;
  model_hyperparameters?: Record<string, any>;
  model_serialization?: string;
  feature_list?: string[];
  spatial_split?: string;
  temporal_split?: string;
  negative_sampling?: string;
  leakage_controls?: Record<string, string>;
  random_seed?: number;
  code_version?: string;
  software_versions?: Record<string, string>;
  input_status?: Record<string, string>;
  generation_timestamp?: string;
  additional_context?: Record<string, any>;
}

/** GET /api/v1/validation/sikkim/evidence */
export interface SikkimEvidenceResponse {
  state: string;
  pilot_area: string;
  /** Honest verdict from model_artifacts.verify_artifact_set: VALID | MISSING | INVALID. */
  status: string;
  gate_compatible: boolean;
  problems: string[];
  metrics: SikkimEvidenceMetrics | null;
  feature_schema: SikkimFeatureSchema | null;
  provenance: SikkimProvenance | null;
}

/** A single real NASA GLC landslide record inside the pilot AOI. */
export interface LandslideEvent {
  latitude: number;
  longitude: number;
  event_date: string;
  event_title: string | null;
  landslide_category: string | null;
  landslide_trigger: string | null;
  landslide_size: string | null;
  location_accuracy: string | null;
  fatality_count: number | null;
  spatial_uncertainty: 'precise_lt_5km' | 'approximate_ge_5km';
  source_name: string | null;
}

/** GET /api/v1/validation/sikkim/events */
export interface SikkimEventsResponse {
  state: string;
  pilot_area: string;
  aoi: AoiBounds;
  count: number;
  source: string;
  spatial_uncertainty_summary: {
    precise_lt_5km: number;
    approximate_ge_5km: number;
    pct_approximate_ge_5km: number;
  };
  events: LandslideEvent[];
}

/** One coarse grid cell scored by the persisted Sikkim model. */
export interface SikkimPredictionCell {
  cell_id: string;
  row: number;
  col: number;
  latitude: number;
  longitude: number;
  bbox: AoiBounds;
  /** 'OK' when scored; 'UNAVAILABLE' when terrain was missing/nodata (no probability). */
  status: 'OK' | 'UNAVAILABLE';
  susceptibility_probability: number | null;
  risk_class: WarningLevel | null;
  exceeds_decision_threshold: boolean | null;
  /** Present only for scored cells; the exact 11-feature vector fed to the model. */
  features?: Record<string, number>;
  reasons: string[];
}

export interface SikkimPredictionGrid {
  step_deg: number;
  n_lat: number;
  n_lon: number;
  cell_count: number;
  cell_height_deg: number;
  cell_width_deg: number;
}

export interface SikkimPredictionRainfall {
  source: string | null;
  run_type: string | null;
  aoi_uniform: boolean;
  window_days: number;
  daily_series_mm: number[] | null;
  features: Record<string, number>;
  note: string;
}

export interface SikkimPredictionSummary {
  cells_total: number;
  cells_scored: number;
  cells_unavailable: number;
  risk_class_counts: Record<WarningLevel, number>;
  cells_exceeding_threshold: number;
  max_probability: number | null;
  mean_probability: number | null;
}

/** GET /api/v1/predict/sikkim/grid */
export interface SikkimPredictionResponse {
  state: string;
  pilot_area: string;
  generated_from: string;
  target_date: string;
  aoi: AoiBounds;
  grid: SikkimPredictionGrid;
  decision_threshold: number;
  model: {
    feature_order: string[];
    n_features: number;
    decision_threshold: number;
    artifact_status: string;
    validation_metrics?: Record<string, any>;
  };
  rainfall: SikkimPredictionRainfall;
  summary: SikkimPredictionSummary;
  /** Honesty caveats (raw-probability vs Option-C, ERA5->IMERG shift, AOI-uniform rainfall, proxy land cover, etc.). */
  disclosures: string[];
  cells: SikkimPredictionCell[];
}

/**
 * Assam pilot response contracts.
 *
 * The three Assam endpoints (/validation/assam/evidence, /validation/assam/events,
 * /predict/assam/grid) return byte-for-byte the SAME JSON shapes as their Sikkim
 * counterparts — the backend shares the identical field projections (_pick) and the
 * same prediction builder — so the Assam console reuses these types via aliases
 * rather than duplicating the definitions. The differences between the two pilots
 * are in the *values*, not the schema: Assam's pilot_area label differs, and
 * land_cover_class is carried as REAL ESA WorldCover (a categorical feature) rather
 * than the Sikkim elevation-derived proxy. Those honest distinctions are surfaced by
 * the Assam console from the live feature_schema / provenance / disclosures fields.
 */
export type AssamEvidenceResponse = SikkimEvidenceResponse;
export type AssamEventsResponse = SikkimEventsResponse;
export type AssamPredictionResponse = SikkimPredictionResponse;
export type AssamPredictionCell = SikkimPredictionCell;

/**
 * Arunachal Pradesh pilot response contracts.
 *
 * The three Arunachal endpoints (/validation/arunachal/evidence,
 * /validation/arunachal/events, /predict/arunachal/grid) return byte-for-byte the SAME
 * JSON shapes as their Sikkim counterparts — the backend shares the identical field
 * projections (_pick) and the same prediction builder — so the Arunachal console reuses
 * these types via aliases rather than duplicating the definitions. As with Assam, the
 * differences are in the *values*, not the schema: Arunachal's pilot_area label differs
 * ("central Subansiri-Siang belt"), and land_cover_class is carried as REAL ESA
 * WorldCover (a categorical feature) rather than the Sikkim elevation-derived proxy —
 * the same methodological treatment as the Assam pilot. Those honest distinctions are
 * surfaced by the Arunachal console from the live feature_schema / provenance /
 * disclosures fields.
 */
export type ArunachalEvidenceResponse = SikkimEvidenceResponse;
export type ArunachalEventsResponse = SikkimEventsResponse;
export type ArunachalPredictionResponse = SikkimPredictionResponse;
export type ArunachalPredictionCell = SikkimPredictionCell;

/**
 * Meghalaya pilot response contracts.
 *
 * The three Meghalaya endpoints (/validation/meghalaya/evidence,
 * /validation/meghalaya/events, /predict/meghalaya/grid) return byte-for-byte the SAME
 * JSON shapes as their Sikkim counterparts — the backend shares the identical field
 * projections (_pick) and the same prediction builder — so the Meghalaya console reuses
 * these types via aliases rather than duplicating the definitions. As with Assam and
 * Arunachal, the differences are in the *values*, not the schema: Meghalaya's pilot_area
 * label differs ("East Khasi + Jaintia Hills belt"), and land_cover_class is carried as
 * REAL ESA WorldCover (a categorical feature) rather than the Sikkim elevation-derived
 * proxy — the same methodological treatment as the Assam and Arunachal pilots. Those
 * honest distinctions are surfaced by the Meghalaya console from the live feature_schema
 * / provenance / disclosures fields.
 */
export type MeghalayaEvidenceResponse = SikkimEvidenceResponse;
export type MeghalayaEventsResponse = SikkimEventsResponse;
export type MeghalayaPredictionResponse = SikkimPredictionResponse;
export type MeghalayaPredictionCell = SikkimPredictionCell;

/* ------------------------------------------------------------------------- *
 * Lightweight pilot MAP view: GET /api/v1/predict/<state>/map
 *
 * A pure projection of the SAME grid prediction (backend
 * app/services/pilot_map_view.py), shaped as a GeoJSON FeatureCollection so raw
 * Leaflet can consume it directly. It carries one Point feature per grid cell at
 * the cell CENTER with only five properties; the per-cell 11-feature vectors,
 * cell bboxes and per-cell reasons are deliberately omitted and remain available
 * on /predict/<state>/grid.
 *
 * Two consequences the consoles must respect:
 *   * `bbox` is gone, so a cell rectangle is reconstructed from the copied
 *     `grid.cell_height_deg` / `grid.cell_width_deg` around the feature's center
 *     (see components/pilot/pilotMapCells.ts). Nothing is guessed.
 *   * `disclosures` is NOT in this payload; it stays on /grid, which the consoles
 *     fetch lazily only when an operator opens the audit panel.
 * ------------------------------------------------------------------------- */

/** Route token used by the pilot endpoints (NOT always the backend state_id). */
export type PilotStateKey = 'sikkim' | 'assam' | 'arunachal' | 'meghalaya';

/** rainfall_service quality vocabulary, reported verbatim. */
export type RainfallQualityStatus = 'REAL' | 'FALLBACK' | 'UNAVAILABLE';
/** rainfall_service source_kind vocabulary, reported verbatim. */
export type RainfallSourceKind = 'IMERG' | 'OPEN_METEO_FALLBACK';

/**
 * Cache/observation-age block from rainfall_service. Every field is optional
 * because a legacy or injected provider may not supply it; absence must render as
 * "not reported", never as a fabricated zero.
 */
export interface RainfallFreshness {
  cache_hit?: boolean | null;
  age_seconds?: number | null;
  ttl_seconds?: number | null;
  expires_in_seconds?: number | null;
  observation_lag_days?: number | null;
  probe_days_walked?: number | null;
  probe_first_offset?: number | null;
  probe_reach?: string | null;
  probe_mode?: string | null;
}

export interface RainfallCoverage {
  state?: string | null;
  aoi_uniform?: boolean | null;
  window_days?: number | null;
  window_semantics?: string | null;
}

/**
 * The O(1) rainfall provenance subset copied into the map document
 * (pilot_map_view.RAINFALL_VIEW_KEYS). The heavy `daily_series_mm` and the five
 * rainfall model features are NOT here — they are not rewritten, just dropped.
 * `null` for the whole block means the producer reported no rainfall metadata at
 * all, which must be surfaced as unreported provenance.
 */
export interface PilotMapRainfallView {
  source?: string | null;
  source_kind?: RainfallSourceKind | string | null;
  run_type?: string | null;
  is_fallback?: boolean | null;
  data_quality_status?: RainfallQualityStatus | string | null;
  units?: string | null;
  requested_date?: string | null;
  rainfall_observation_date?: string | null;
  fetched_at_utc?: string | null;
  freshness?: RainfallFreshness | null;
  window_days?: number | null;
  aoi_uniform?: boolean | null;
  note?: string | null;
  caveats?: string[] | null;
  coverage?: RainfallCoverage | null;
}

/**
 * The route-level normalised provenance block (routes._rainfall_provenance).
 * Present only when the producer supplied rainfall metadata; `fallback_warning`
 * appears verbatim whenever is_fallback is true.
 */
export interface RainfallProvenanceBlock {
  source?: string | null;
  source_kind?: RainfallSourceKind | string | null;
  is_fallback?: boolean | null;
  data_quality_status?: RainfallQualityStatus | string | null;
  requested_date?: string | null;
  rainfall_observation_date?: string | null;
  fetched_at_utc?: string | null;
  freshness?: RainfallFreshness | null;
  units?: string | null;
  fallback_warning?: string;
  caveats?: string[];
}

/**
 * Exactly the five properties pilot_map_view emits per cell. An UNAVAILABLE cell
 * is PRESENT with probability/risk_class/exceeds_decision_threshold all null —
 * it must stay visible on the map rather than be dropped and read as safe.
 */
export interface PilotMapCellProperties {
  cell_id: string | null;
  status: 'OK' | 'UNAVAILABLE' | string | null;
  probability: number | null;
  risk_class: WarningLevel | null;
  exceeds_decision_threshold: boolean | null;
}

/** GeoJSON Point at the cell center. Coordinates are [lon, lat] (not Leaflet order). */
export interface PilotMapFeature {
  type: 'Feature';
  id?: string | null;
  geometry: {
    type: 'Point';
    coordinates: [number, number];
  };
  properties: PilotMapCellProperties;
}

/** Self-describing metadata about what this projection kept and dropped. */
export interface PilotMapViewMeta {
  kind: string;
  geometry: string;
  cell_property_keys: string[];
  omitted_cell_keys: string[];
  full_response_endpoint_note?: string;
  note?: string;
}

/** GET /api/v1/predict/<state>/map */
export interface PilotMapResponse {
  type: 'FeatureCollection';
  state: string | null;
  pilot_area: string | null;
  target_date: string | null;
  /** Derived from the rainfall series actually used — never a hard-coded IMERG claim. */
  generated_from: string | null;
  decision_threshold: number | null;
  aoi: AoiBounds | null;
  /** Copied verbatim; cell_height_deg / cell_width_deg let the map rebuild rectangles. */
  grid: SikkimPredictionGrid | null;
  rainfall: PilotMapRainfallView | null;
  summary: SikkimPredictionSummary | null;
  view: PilotMapViewMeta;
  rainfall_provenance?: RainfallProvenanceBlock;
  features: PilotMapFeature[];
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

  /**
   * Persisted Sikkim model-evidence bundle (metrics + feature schema + provenance).
   * Returns an explicit status (VALID/MISSING/INVALID); never fabricated.
   */
  async getSikkimEvidence(): Promise<SikkimEvidenceResponse> {
    return this.fetchJson<SikkimEvidenceResponse>('/api/v1/validation/sikkim/evidence');
  }

  /**
   * Real NASA GLC landslide positives inside the East Sikkim pilot AOI.
   * Throws if the backend refuses (HTTP 503 DATA_UNAVAILABLE) rather than
   * returning a synthesised list.
   */
  async getSikkimEvents(): Promise<SikkimEventsResponse> {
    return this.fetchJson<SikkimEventsResponse>('/api/v1/validation/sikkim/events');
  }

  /**
   * Real per-grid-cell landslide susceptibility for the East Sikkim pilot AOI,
   * produced by running the persisted 11-feature LightGBM over a coarse grid with
   * real IMERG antecedent rainfall. This is the model's RAW probability, not the
   * Option-C fused /risk/current score (see response.disclosures). Throws if the
   * backend refuses (HTTP 503 DATA_UNAVAILABLE) rather than returning fabricated
   * risk zones.
   *
   * @param date optional 'YYYY-MM-DD' prediction date (default: backend uses today, UTC)
   * @param step optional grid cell size in degrees (default: backend coarse grid)
   */
  async getSikkimPrediction(date?: string, step?: number): Promise<SikkimPredictionResponse> {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    if (step != null) params.set('step', String(step));
    const query = params.toString();
    return this.fetchJson<SikkimPredictionResponse>(
      `/api/v1/predict/sikkim/grid${query ? `?${query}` : ''}`,
    );
  }

  /**
   * Persisted Assam model-evidence bundle (metrics + feature schema + provenance).
   * Same contract as getSikkimEvidence — returns an explicit status
   * (VALID/MISSING/INVALID), reads every number from the persisted Assam artifacts,
   * and never fabricates. The Assam-specific truth (land_cover_class is REAL ESA
   * WorldCover, not an elevation proxy) is carried faithfully in feature_schema /
   * provenance.
   */
  async getAssamEvidence(): Promise<AssamEvidenceResponse> {
    return this.fetchJson<AssamEvidenceResponse>('/api/v1/validation/assam/evidence');
  }

  /**
   * Real NASA GLC landslide positives inside the canonical Assam pilot AOI
   * (Guwahati-Kamrup + western Karbi Anglong). Throws if the backend refuses
   * (HTTP 503 DATA_UNAVAILABLE) rather than returning a synthesised list.
   */
  async getAssamEvents(): Promise<AssamEventsResponse> {
    return this.fetchJson<AssamEventsResponse>('/api/v1/validation/assam/events');
  }

  /**
   * Real per-grid-cell landslide susceptibility for the Assam pilot AOI, produced by
   * running the persisted 11-feature Assam LightGBM over a coarse grid with real ESA
   * WorldCover land cover (categorical) and real IMERG antecedent rainfall. This is
   * the model's RAW probability, not the Option-C fused score (see
   * response.disclosures, which also record the ERA5-trained → IMERG-served rainfall
   * shift). Throws if the backend refuses (HTTP 503 DATA_UNAVAILABLE) rather than
   * returning fabricated risk zones.
   *
   * @param date optional 'YYYY-MM-DD' prediction date (default: backend uses today, UTC)
   * @param step optional grid cell size in degrees (default: backend coarse grid)
   */
  async getAssamPrediction(date?: string, step?: number): Promise<AssamPredictionResponse> {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    if (step != null) params.set('step', String(step));
    const query = params.toString();
    return this.fetchJson<AssamPredictionResponse>(
      `/api/v1/predict/assam/grid${query ? `?${query}` : ''}`,
    );
  }

  /**
   * Persisted Arunachal Pradesh model-evidence bundle (metrics + feature schema +
   * provenance). Same contract as getSikkimEvidence — returns an explicit status
   * (VALID/MISSING/INVALID), reads every number from the persisted Arunachal artifacts,
   * and never fabricates. As with Assam, the pilot-specific truth (land_cover_class is
   * REAL ESA WorldCover, not an elevation proxy) is carried faithfully in
   * feature_schema / provenance.
   */
  async getArunachalEvidence(): Promise<ArunachalEvidenceResponse> {
    return this.fetchJson<ArunachalEvidenceResponse>('/api/v1/validation/arunachal/evidence');
  }

  /**
   * Real NASA GLC landslide positives inside the canonical Arunachal Pradesh pilot AOI
   * (central Subansiri-Siang belt). Throws if the backend refuses
   * (HTTP 503 DATA_UNAVAILABLE) rather than returning a synthesised list.
   */
  async getArunachalEvents(): Promise<ArunachalEventsResponse> {
    return this.fetchJson<ArunachalEventsResponse>('/api/v1/validation/arunachal/events');
  }

  /**
   * Real per-grid-cell landslide susceptibility for the Arunachal Pradesh pilot AOI,
   * produced by running the persisted 11-feature Arunachal LightGBM over a coarse grid
   * with real ESA WorldCover land cover (categorical) and real IMERG antecedent
   * rainfall. This is the model's RAW probability, not the Option-C fused score (see
   * response.disclosures, which also record the ERA5-trained → IMERG-served rainfall
   * shift). Throws if the backend refuses (HTTP 503 DATA_UNAVAILABLE) rather than
   * returning fabricated risk zones.
   *
   * @param date optional 'YYYY-MM-DD' prediction date (default: backend uses today, UTC)
   * @param step optional grid cell size in degrees (default: backend coarse grid)
   */
  async getArunachalPrediction(date?: string, step?: number): Promise<ArunachalPredictionResponse> {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    if (step != null) params.set('step', String(step));
    const query = params.toString();
    return this.fetchJson<ArunachalPredictionResponse>(
      `/api/v1/predict/arunachal/grid${query ? `?${query}` : ''}`,
    );
  }

  /**
   * Persisted Meghalaya model-evidence bundle (metrics + feature schema + provenance).
   * Same contract as getSikkimEvidence — returns an explicit status
   * (VALID/MISSING/INVALID), reads every number from the persisted Meghalaya artifacts,
   * and never fabricates. As with Assam and Arunachal, the pilot-specific truth
   * (land_cover_class is REAL ESA WorldCover, not an elevation proxy) is carried
   * faithfully in feature_schema / provenance.
   */
  async getMeghalayaEvidence(): Promise<MeghalayaEvidenceResponse> {
    return this.fetchJson<MeghalayaEvidenceResponse>('/api/v1/validation/meghalaya/evidence');
  }

  /**
   * Real NASA GLC landslide positives inside the canonical Meghalaya pilot AOI
   * (East Khasi + Jaintia Hills belt). Throws if the backend refuses
   * (HTTP 503 DATA_UNAVAILABLE) rather than returning a synthesised list.
   */
  async getMeghalayaEvents(): Promise<MeghalayaEventsResponse> {
    return this.fetchJson<MeghalayaEventsResponse>('/api/v1/validation/meghalaya/events');
  }

  /**
   * Real per-grid-cell landslide susceptibility for the Meghalaya pilot AOI, produced by
   * running the persisted 11-feature Meghalaya LightGBM over a coarse grid with real ESA
   * WorldCover land cover (categorical) and real IMERG antecedent rainfall. This is the
   * model's RAW probability, not the Option-C fused score (see response.disclosures,
   * which also record the ERA5-trained → IMERG-served rainfall shift). Throws if the
   * backend refuses (HTTP 503 DATA_UNAVAILABLE) rather than returning fabricated risk
   * zones.
   *
   * @param date optional 'YYYY-MM-DD' prediction date (default: backend uses today, UTC)
   * @param step optional grid cell size in degrees (default: backend coarse grid)
   */
  async getMeghalayaPrediction(date?: string, step?: number): Promise<MeghalayaPredictionResponse> {
    const params = new URLSearchParams();
    if (date) params.set('date', date);
    if (step != null) params.set('step', String(step));
    const query = params.toString();
    return this.fetchJson<MeghalayaPredictionResponse>(
      `/api/v1/predict/meghalaya/grid${query ? `?${query}` : ''}`,
    );
  }

  /**
   * COMPACT map payload for any of the four pilot consoles: the same prediction
   * as /predict/<state>/grid, projected to a GeoJSON FeatureCollection of
   * cell-centre Points with five properties each. This is what the consoles use
   * to draw their maps.
   *
   * The endpoint is selected from the pilot registry by `state`, so there is one
   * code path for all four states and an unknown key throws instead of silently
   * resolving to another state's data. ONE request draws the whole grid — there
   * is no per-cell request anywhere in this layer.
   *
   * Throws when the backend refuses (HTTP 503 DATA_UNAVAILABLE, e.g. a missing
   * land-cover raster or model artifact) rather than returning fabricated risk
   * zones; the caller must render an explicit unavailable state carrying the
   * backend's own reason.
   *
   * @param state pilot route token: 'sikkim' | 'assam' | 'arunachal' | 'meghalaya'
   * @param date optional 'YYYY-MM-DD' prediction date (default: backend uses today, UTC)
   * @param step optional grid cell size in degrees (default: backend coarse grid)
   */
  async getPilotMap(
    state: PilotStateKey,
    date?: string,
    step?: number,
  ): Promise<PilotMapResponse> {
    return this.fetchJson<PilotMapResponse>(pilotMapEndpoint(state, { date, step }));
  }

  /**
   * FULL grid response for a pilot, including the per-cell 11-feature vectors and
   * the `disclosures` list. Deliberately NOT used to draw the map: the consoles
   * call this only when an operator opens "Audit this prediction", so the initial
   * map load issues no /grid request.
   */
  async getPilotGrid(
    state: PilotStateKey,
    date?: string,
    step?: number,
  ): Promise<SikkimPredictionResponse> {
    return this.fetchJson<SikkimPredictionResponse>(pilotGridEndpoint(state, { date, step }));
  }
}

export const apiService = new ApiService();
