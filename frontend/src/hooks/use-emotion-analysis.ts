"use client";

import { useState, useCallback } from "react";
import type {
  EmotionResult,
  MusicRecommendation,
  TherapeuticInsight,
  ModelPrediction,
  AudioFeatures,
  InputMethod,
} from "@/types";
import {
  analyzeManual,
  analyzeFeatures,
  analyzeAudio,
  getRecommendations,
  getInsight,
  getModelComparison,
} from "@/services/api";

interface AnalysisState {
  isLoading: boolean;
  error: string | null;
  emotion: EmotionResult | null;
  recommendations: MusicRecommendation | null;
  insight: TherapeuticInsight | null;
  modelComparison: ModelPrediction[];
}

export function useEmotionAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    isLoading: false,
    error: null,
    emotion: null,
    recommendations: null,
    insight: null,
    modelComparison: [],
  });

  const analyze = useCallback(
    async (
      method: InputMethod,
      params: {
        arousal?: number;
        valence?: number;
        features?: AudioFeatures;
        file?: File;
      }
    ) => {
      setState((prev) => ({ ...prev, isLoading: true, error: null }));

      try {
        let emotion: EmotionResult;

        switch (method) {
          case "manual":
            emotion = await analyzeManual(
              params.arousal ?? 0.5,
              params.valence ?? 0.5
            );
            break;
          case "features":
            if (!params.features) throw new Error("Features are required");
            emotion = await analyzeFeatures(params.features);
            break;
          case "upload":
            if (!params.file) throw new Error("Audio file is required");
            emotion = await analyzeAudio(params.file);
            break;
        }

        const [recommendations, insight, modelComparison] = await Promise.all([
          getRecommendations(
            emotion.emotion_label,
            emotion.arousal,
            emotion.valence
          ),
          getInsight(emotion.emotion_label, emotion.arousal, emotion.valence),
          method === "features" && params.features
            ? getModelComparison(params.features)
            : Promise.resolve([]),
        ]);

        setState({
          isLoading: false,
          error: null,
          emotion,
          recommendations,
          insight,
          modelComparison,
        });
      } catch (err) {
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: err instanceof Error ? err.message : "Analysis failed",
        }));
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState({
      isLoading: false,
      error: null,
      emotion: null,
      recommendations: null,
      insight: null,
      modelComparison: [],
    });
  }, []);

  return { ...state, analyze, reset };
}
