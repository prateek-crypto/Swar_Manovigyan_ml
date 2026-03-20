"use client";

import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Slider } from "@/components/ui/slider";
import { EMOTION_MAP } from "@/lib/constants";
import { deriveEmotionQuadrant } from "@/lib/utils";
import type { EmotionQuadrant } from "@/types";

interface EmotionSelectorProps {
  arousal: number;
  valence: number;
  onArousalChange: (value: number) => void;
  onValenceChange: (value: number) => void;
  onAnalyze?: () => void;
}

export function EmotionSelector({
  arousal,
  valence,
  onArousalChange,
  onValenceChange,
}: EmotionSelectorProps) {
  const quadrant = deriveEmotionQuadrant(arousal, valence);
  const emotionInfo = EMOTION_MAP[quadrant];

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-lg font-semibold">How are you feeling?</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Adjust the sliders to match your current emotional state
        </p>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Arousal Slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">
              Energy Level (Arousal)
            </label>
            <span className="text-sm tabular-nums text-muted-foreground">
              {arousal.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[arousal]}
            onValueChange={([v]) => onArousalChange(v)}
            min={0}
            max={1}
            step={0.01}
            className="w-full"
          />
          <div className="flex justify-between text-[11px] text-muted-foreground">
            <span>Calm, Relaxed</span>
            <span>Energetic, Excited</span>
          </div>
        </div>

        {/* Valence Slider */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">
              Mood Positivity (Valence)
            </label>
            <span className="text-sm tabular-nums text-muted-foreground">
              {valence.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[valence]}
            onValueChange={([v]) => onValenceChange(v)}
            min={0}
            max={1}
            step={0.01}
            className="w-full"
          />
          <div className="flex justify-between text-[11px] text-muted-foreground">
            <span>Negative</span>
            <span>Positive</span>
          </div>
        </div>
      </div>

      {/* AV Space Preview */}
      <AVSpacePreview arousal={arousal} valence={valence} quadrant={quadrant} />

      {/* Current Emotion Preview */}
      <motion.div
        key={quadrant}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn(
          "rounded-lg border p-4 bg-gradient-to-r",
          emotionInfo.gradient
        )}
      >
        <div className="flex items-center gap-3">
          <div
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: emotionInfo.color }}
          />
          <div>
            <p className="text-sm font-medium">{emotionInfo.label}</p>
            <p className="text-xs text-muted-foreground">
              {emotionInfo.description}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

function AVSpacePreview({
  arousal,
  valence,
  quadrant,
}: {
  arousal: number;
  valence: number;
  quadrant: EmotionQuadrant;
}) {
  const info = EMOTION_MAP[quadrant];

  return (
    <div className="relative mx-auto aspect-square w-full max-w-[280px] rounded-lg border bg-muted/30">
      {/* Grid lines */}
      <div className="absolute inset-0 flex items-center">
        <div className="h-px w-full bg-border" />
      </div>
      <div className="absolute inset-0 flex justify-center">
        <div className="h-full w-px bg-border" />
      </div>

      {/* Quadrant labels */}
      <span className="absolute left-2 top-2 text-[10px] text-muted-foreground">
        Angry
      </span>
      <span className="absolute right-2 top-2 text-[10px] text-muted-foreground">
        Happy
      </span>
      <span className="absolute bottom-2 left-2 text-[10px] text-muted-foreground">
        Sad
      </span>
      <span className="absolute bottom-2 right-2 text-[10px] text-muted-foreground">
        Calm
      </span>

      {/* Axis labels */}
      <span className="absolute -bottom-5 left-1/2 -translate-x-1/2 text-[10px] text-muted-foreground">
        Valence →
      </span>
      <span className="absolute -left-5 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-muted-foreground">
        Arousal →
      </span>

      {/* Position dot */}
      <motion.div
        className="absolute h-4 w-4 -translate-x-1/2 translate-y-1/2 rounded-full shadow-lg"
        style={{ backgroundColor: info.color }}
        animate={{
          left: `${valence * 100}%`,
          bottom: `${arousal * 100}%`,
        }}
        transition={{ type: "spring", stiffness: 300, damping: 25 }}
      >
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{ backgroundColor: info.color }}
          animate={{ scale: [1, 1.8, 1], opacity: [0.6, 0, 0.6] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
      </motion.div>
    </div>
  );
}
