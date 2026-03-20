export type EmotionQuadrant = 0 | 1 | 2 | 3;

export interface EmotionInfo {
  label: string;
  description: string;
  color: string;
  icon: string;
  gradient: string;
}

export interface EmotionResult {
  emotion_label: EmotionQuadrant;
  confidence: number;
  arousal: number;
  valence: number;
  probabilities?: number[];
}

export interface AudioFeatures {
  acousticness: number;
  danceability: number;
  energy: number;
  instrumentalness: number;
  liveness: number;
  loudness: number;
  speechiness: number;
  tempo: number;
  valence: number;
}

export interface Track {
  id: string;
  title: string;
  artist: string;
  duration: string;
}

export interface MusicRecommendation {
  styles: string[];
  blurb?: string;
  tracks: Track[];
  isAiGenerated: boolean;
}

export interface ModelPrediction {
  model: string;
  emotion: string;
  confidence: number;
}

export interface TherapeuticInsight {
  text: string;
  audioUrl?: string;
}

export interface SystemStatus {
  azureOpenAI: boolean;
  azureSpeech: boolean;
  lstmModel: boolean;
  avRegressor: boolean;
  baselineModels: boolean;
}

export type InputMethod = "manual" | "features" | "upload";

export interface AnalyzeRequest {
  method: InputMethod;
  arousal?: number;
  valence?: number;
  features?: AudioFeatures;
  audioFile?: File;
}

export interface AnalyzeResponse {
  emotion: EmotionResult;
  recommendations: MusicRecommendation;
  insight?: TherapeuticInsight;
  modelComparison?: ModelPrediction[];
}
