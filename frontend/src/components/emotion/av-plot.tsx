"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceLine,
  Label,
} from "recharts";
import { EMOTION_MAP } from "@/lib/constants";
import type { EmotionQuadrant } from "@/types";

interface AVPlotProps {
  arousal: number;
  valence: number;
  emotionLabel: EmotionQuadrant;
}

export function AVPlot({ arousal, valence, emotionLabel }: AVPlotProps) {
  const info = EMOTION_MAP[emotionLabel];
  const data = [{ x: valence, y: arousal }];

  return (
    <div className="relative aspect-square w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 20 }}>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="hsl(var(--border))"
            opacity={0.5}
          />
          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            tick={{ fontSize: 10 }}
            stroke="hsl(var(--muted-foreground))"
          >
            <Label
              value="Valence"
              position="bottom"
              offset={0}
              style={{
                fontSize: 10,
                fill: "hsl(var(--muted-foreground))",
              }}
            />
          </XAxis>
          <YAxis
            type="number"
            dataKey="y"
            domain={[0, 1]}
            tick={{ fontSize: 10 }}
            stroke="hsl(var(--muted-foreground))"
          >
            <Label
              value="Arousal"
              position="left"
              angle={-90}
              offset={0}
              style={{
                fontSize: 10,
                fill: "hsl(var(--muted-foreground))",
              }}
            />
          </YAxis>
          <ReferenceLine x={0.5} stroke="hsl(var(--border))" />
          <ReferenceLine y={0.5} stroke="hsl(var(--border))" />
          <Scatter data={data} fill={info.color} r={8} />
        </ScatterChart>
      </ResponsiveContainer>

      {/* Quadrant labels overlay */}
      <div className="pointer-events-none absolute inset-0">
        <span className="absolute left-[15%] top-[15%] text-[9px] text-muted-foreground/60">
          Angry
        </span>
        <span className="absolute right-[15%] top-[15%] text-[9px] text-muted-foreground/60">
          Happy
        </span>
        <span className="absolute bottom-[20%] left-[15%] text-[9px] text-muted-foreground/60">
          Sad
        </span>
        <span className="absolute bottom-[20%] right-[15%] text-[9px] text-muted-foreground/60">
          Calm
        </span>
      </div>
    </div>
  );
}
