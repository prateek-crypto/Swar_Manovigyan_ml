import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { EmotionQuadrant, AudioFeatures } from "@/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function deriveEmotionQuadrant(
  arousal: number,
  valence: number
): EmotionQuadrant {
  const arousalBin = arousal > 0.5 ? 1 : 0;
  const valenceBin = valence > 0.5 ? 1 : 0;
  return (arousalBin * 2 + valenceBin) as EmotionQuadrant;
}

const LOUDNESS_MIN = -60;
const LOUDNESS_MAX = 0;
const TEMPO_MIN = 50;
const TEMPO_MAX = 200;

export function estimateArousalValence(features: AudioFeatures): {
  arousal: number;
  valence: number;
} {
  const loudnessNorm =
    (features.loudness - LOUDNESS_MIN) / (LOUDNESS_MAX - LOUDNESS_MIN + 1e-8);
  const tempoNorm =
    (features.tempo - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN + 1e-8);

  const arousal =
    0.4 * features.energy + 0.3 * loudnessNorm + 0.3 * tempoNorm;
  const enhancedValence =
    0.6 * features.valence +
    0.3 * features.danceability +
    0.1 * (1.0 - features.acousticness);

  return {
    arousal: Math.max(0, Math.min(1, arousal)),
    valence: Math.max(0, Math.min(1, enhancedValence)),
  };
}

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

export function formatDuration(duration: string): string {
  return duration;
}
