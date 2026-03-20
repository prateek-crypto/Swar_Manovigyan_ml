"use client";

import { motion } from "framer-motion";
import { Music2, Clock, User2, Headphones } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { Track } from "@/types";

interface PlaylistViewProps {
  tracks: Track[];
  isAiGenerated: boolean;
}

export function PlaylistView({ tracks, isAiGenerated }: PlaylistViewProps) {
  if (tracks.length === 0) return null;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Headphones className="h-4 w-4" />
            Sample Playlist
          </CardTitle>
          {isAiGenerated && (
            <p className="text-[11px] text-muted-foreground">
              AI-suggested — search on your streaming platform
            </p>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y">
          {tracks.map((track, i) => (
            <motion.div
              key={track.id}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.06 }}
              className="flex items-center gap-4 px-6 py-3 transition-colors hover:bg-muted/50"
            >
              <span className="w-6 text-center text-xs tabular-nums text-muted-foreground">
                {i + 1}
              </span>
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-muted">
                <Music2 className="h-3.5 w-3.5 text-muted-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{track.title}</p>
                <p className="flex items-center gap-1 text-xs text-muted-foreground">
                  <User2 className="h-3 w-3" />
                  {track.artist}
                </p>
              </div>
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                {track.duration}
              </div>
            </motion.div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
