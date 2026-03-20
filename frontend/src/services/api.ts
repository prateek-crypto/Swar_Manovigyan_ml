import type {
  EmotionQuadrant,
  AudioFeatures,
  EmotionResult,
  MusicRecommendation,
  TherapeuticInsight,
  ModelPrediction,
  SystemStatus,
} from "@/types";
import {
  FALLBACK_RECOMMENDATIONS,
  FALLBACK_TRACKS,
} from "@/lib/constants";
import { deriveEmotionQuadrant, estimateArousalValence } from "@/lib/utils";

const API_BASE = "";

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function analyzeManual(
  arousal: number,
  valence: number
): Promise<EmotionResult> {
  try {
    return await fetchJSON<EmotionResult>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ method: "manual", arousal, valence }),
    });
  } catch {
    const emotionLabel = deriveEmotionQuadrant(arousal, valence);
    return {
      emotion_label: emotionLabel,
      confidence: 1.0,
      arousal,
      valence,
    };
  }
}

export async function analyzeFeatures(
  features: AudioFeatures
): Promise<EmotionResult> {
  try {
    return await fetchJSON<EmotionResult>("/api/analyze", {
      method: "POST",
      body: JSON.stringify({ method: "features", features }),
    });
  } catch {
    const { arousal, valence } = estimateArousalValence(features);
    const emotionLabel = deriveEmotionQuadrant(arousal, valence);
    return {
      emotion_label: emotionLabel,
      confidence: 1.0,
      arousal,
      valence,
    };
  }
}

export async function analyzeAudio(file: File): Promise<EmotionResult> {
  const formData = new FormData();
  formData.append("audio", file);

  const res = await fetch(`${API_BASE}/api/analyze/audio`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`Audio analysis failed: ${res.status}`);
  }

  return res.json() as Promise<EmotionResult>;
}

export async function getRecommendations(
  emotionLabel: EmotionQuadrant,
  arousal: number,
  valence: number
): Promise<MusicRecommendation> {
  try {
    return await fetchJSON<MusicRecommendation>("/api/recommend", {
      method: "POST",
      body: JSON.stringify({
        emotion_label: emotionLabel,
        arousal,
        valence,
      }),
    });
  } catch {
    return {
      styles: FALLBACK_RECOMMENDATIONS[emotionLabel],
      tracks: FALLBACK_TRACKS[emotionLabel],
      isAiGenerated: false,
    };
  }
}

export async function getInsight(
  emotionLabel: EmotionQuadrant,
  arousal: number,
  valence: number
): Promise<TherapeuticInsight | null> {
  try {
    return await fetchJSON<TherapeuticInsight>("/api/analyze/insight", {
      method: "POST",
      body: JSON.stringify({
        emotion_label: emotionLabel,
        arousal,
        valence,
      }),
    });
  } catch {
    return null;
  }
}

export async function getModelComparison(
  features: AudioFeatures
): Promise<ModelPrediction[]> {
  try {
    return await fetchJSON<ModelPrediction[]>("/api/analyze/models", {
      method: "POST",
      body: JSON.stringify({ features }),
    });
  } catch {
    return [];
  }
}

export async function getSystemStatus(): Promise<SystemStatus> {
  try {
    return await fetchJSON<SystemStatus>("/api/status");
  } catch {
    return {
      azureOpenAI: false,
      azureSpeech: false,
      lstmModel: false,
      avRegressor: false,
      baselineModels: false,
    };
  }
}

export async function synthesizeSpeech(text: string): Promise<Blob | null> {
  try {
    const res = await fetch(`${API_BASE}/api/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) return null;
    return await res.blob();
  } catch {
    return null;
  }
}
