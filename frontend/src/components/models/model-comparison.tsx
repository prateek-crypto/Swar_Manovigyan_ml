"use client";

import { motion } from "framer-motion";
import { Brain, Trophy, BarChart3 } from "lucide-react";
import { cn, formatConfidence } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { ModelPrediction } from "@/types";

interface ModelComparisonProps {
  predictions: ModelPrediction[];
}

export function ModelComparison({ predictions }: ModelComparisonProps) {
  if (predictions.length === 0) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <BarChart3 className="mb-3 h-8 w-8 text-muted-foreground/50" />
          <p className="text-sm font-medium text-muted-foreground">
            No model predictions available
          </p>
          <p className="mt-1 text-xs text-muted-foreground/70">
            Use the Audio Features input to see predictions from all models
          </p>
        </CardContent>
      </Card>
    );
  }

  const topModel = predictions.reduce((best, p) =>
    p.confidence > best.confidence ? p : best
  );

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Brain className="h-4 w-4" />
            Model Comparison
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {predictions.map((prediction, i) => {
              const isTop = prediction.model === topModel.model;

              return (
                <motion.div
                  key={prediction.model}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.08 }}
                  className={cn(
                    "rounded-lg border p-4 transition-colors",
                    isTop && "border-primary/30 bg-primary/5"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">
                        {prediction.model}
                      </span>
                      {isTop && (
                        <Badge
                          variant="default"
                          className="gap-1 text-[10px]"
                        >
                          <Trophy className="h-3 w-3" />
                          Top
                        </Badge>
                      )}
                    </div>
                    <Badge variant="outline">
                      {prediction.emotion}
                    </Badge>
                  </div>
                  <div className="mt-3 space-y-1">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Confidence</span>
                      <span className="tabular-nums">
                        {formatConfidence(prediction.confidence)}
                      </span>
                    </div>
                    <Progress
                      value={prediction.confidence * 100}
                      className="h-1.5"
                    />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
