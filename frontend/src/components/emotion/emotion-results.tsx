"use client";

import { motion } from "framer-motion";
import {
  CloudRain,
  Leaf,
  Flame,
  Sun,
  Sparkles,
  Volume2,
  Loader2,
} from "lucide-react";
import { cn, formatConfidence } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { EMOTION_MAP } from "@/lib/constants";
import { AVPlot } from "./av-plot";
import type { EmotionResult, TherapeuticInsight } from "@/types";
import { useState } from "react";
import { synthesizeSpeech } from "@/services/api";

const ICON_MAP = {
  CloudRain,
  Leaf,
  Flame,
  Sun,
} as const;

interface EmotionResultsProps {
  result: EmotionResult;
  insight: TherapeuticInsight | null;
}

export function EmotionResults({ result, insight }: EmotionResultsProps) {
  const emotionInfo = EMOTION_MAP[result.emotion_label];
  const Icon =
    ICON_MAP[emotionInfo.icon as keyof typeof ICON_MAP] ?? Sparkles;

  return (
    <div className="space-y-6">
      {/* Primary Result Card */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Card
          className={cn(
            "overflow-hidden border-0 bg-gradient-to-br",
            emotionInfo.gradient
          )}
        >
          <CardContent className="p-6">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
              {/* Emotion Info */}
              <div className="flex-1 space-y-4">
                <div className="flex items-center gap-3">
                  <div
                    className="flex h-10 w-10 items-center justify-center rounded-xl"
                    style={{ backgroundColor: emotionInfo.color + "20" }}
                  >
                    <Icon
                      className="h-5 w-5"
                      style={{ color: emotionInfo.color }}
                    />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold">
                      {emotionInfo.label}
                    </h3>
                    <Badge variant="secondary" className="mt-0.5">
                      Confidence: {formatConfidence(result.confidence)}
                    </Badge>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground leading-relaxed">
                  {emotionInfo.description}
                </p>

                {/* A/V Values */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Arousal</span>
                      <span className="font-medium tabular-nums">
                        {result.arousal.toFixed(2)}
                      </span>
                    </div>
                    <Progress
                      value={result.arousal * 100}
                      className="h-1.5"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">Valence</span>
                      <span className="font-medium tabular-nums">
                        {result.valence.toFixed(2)}
                      </span>
                    </div>
                    <Progress
                      value={result.valence * 100}
                      className="h-1.5"
                    />
                  </div>
                </div>
              </div>

              {/* A/V Plot */}
              <div className="w-full lg:w-[240px]">
                <AVPlot
                  arousal={result.arousal}
                  valence={result.valence}
                  emotionLabel={result.emotion_label}
                />
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* AI Insight */}
      {insight && (
        <InsightCard insight={insight} />
      )}
    </div>
  );
}

function InsightCard({ insight }: { insight: TherapeuticInsight }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSynthesizing, setIsSynthesizing] = useState(false);

  const handleSpeak = async () => {
    setIsSynthesizing(true);
    try {
      const blob = await synthesizeSpeech(insight.text);
      if (blob) {
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        setIsPlaying(true);
        audio.onended = () => {
          setIsPlaying(false);
          URL.revokeObjectURL(url);
        };
        audio.play();
      }
    } finally {
      setIsSynthesizing(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <Card className="border-primary/20 bg-primary/5">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Sparkles className="h-4 w-4 text-primary" />
            AI Therapeutic Insight
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-muted-foreground">
            {insight.text}
          </p>
          <div className="mt-3">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSpeak}
              disabled={isSynthesizing || isPlaying}
            >
              {isSynthesizing ? (
                <Loader2 className="mr-2 h-3 w-3 animate-spin" />
              ) : (
                <Volume2 className="mr-2 h-3 w-3" />
              )}
              {isPlaying ? "Playing..." : "Listen"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
