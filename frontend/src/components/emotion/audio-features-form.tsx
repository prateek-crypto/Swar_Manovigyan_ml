"use client";

import { motion } from "framer-motion";
import { Slider } from "@/components/ui/slider";
import {
  AUDIO_FEATURE_RANGES,
  FEATURE_LABELS,
} from "@/lib/constants";
import type { AudioFeatures } from "@/types";

interface AudioFeaturesFormProps {
  features: AudioFeatures;
  onFeaturesChange: (features: AudioFeatures) => void;
}

const FEATURE_GROUPS = [
  {
    title: "Rhythm & Energy",
    features: ["energy", "danceability", "tempo"] as const,
  },
  {
    title: "Tonality & Character",
    features: ["valence", "acousticness", "instrumentalness"] as const,
  },
  {
    title: "Production & Presence",
    features: ["loudness", "speechiness", "liveness"] as const,
  },
];

export function AudioFeaturesForm({
  features,
  onFeaturesChange,
}: AudioFeaturesFormProps) {
  const handleChange = (key: keyof AudioFeatures, value: number) => {
    onFeaturesChange({ ...features, [key]: value });
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h2 className="text-lg font-semibold">Audio Features</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Describe the musical qualities to detect the emotional tone
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {FEATURE_GROUPS.map((group, groupIdx) => (
          <motion.div
            key={group.title}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: groupIdx * 0.1 }}
            className="rounded-lg border bg-card p-4"
          >
            <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              {group.title}
            </h3>
            <div className="space-y-5">
              {group.features.map((key) => {
                const range = AUDIO_FEATURE_RANGES[key];
                const label = FEATURE_LABELS[key];
                const value = features[key];

                return (
                  <div key={key} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-sm">{label}</label>
                      <span className="text-xs tabular-nums text-muted-foreground">
                        {value.toFixed(range.step < 1 ? 2 : 0)}
                      </span>
                    </div>
                    <Slider
                      value={[value]}
                      onValueChange={([v]) => handleChange(key, v)}
                      min={range.min}
                      max={range.max}
                      step={range.step}
                    />
                  </div>
                );
              })}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
