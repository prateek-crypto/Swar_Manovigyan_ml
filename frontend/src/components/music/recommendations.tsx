"use client";

import { motion } from "framer-motion";
import { Music, Sparkles, ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { MusicRecommendation } from "@/types";
import { PlaylistView } from "./playlist";

interface RecommendationsProps {
  data: MusicRecommendation;
}

export function Recommendations({ data }: RecommendationsProps) {
  return (
    <div className="space-y-6">
      {/* Blurb */}
      {data.blurb && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-lg border border-primary/20 bg-primary/5 p-4"
        >
          <div className="flex items-start gap-2">
            <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
            <p className="text-sm leading-relaxed text-muted-foreground">
              {data.blurb}
            </p>
          </div>
        </motion.div>
      )}

      {/* Style Suggestions */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Music className="h-4 w-4" />
            Recommended Styles
            {data.isAiGenerated && (
              <Badge variant="secondary" className="text-[10px]">
                AI Generated
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {data.styles.map((style, i) => (
              <motion.div
                key={style}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.05 }}
              >
                <Badge
                  variant="outline"
                  className="cursor-default py-1.5 px-3 text-xs"
                >
                  {style}
                </Badge>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Playlist */}
      <PlaylistView
        tracks={data.tracks}
        isAiGenerated={data.isAiGenerated}
      />
    </div>
  );
}
