"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  X,
  Cloud,
  Volume2,
  Brain,
  Activity,
  CheckCircle2,
  XCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { SystemStatus } from "@/types";

interface SystemStatusPanelProps {
  status: SystemStatus;
  isOpen: boolean;
  onClose: () => void;
}

const STATUS_ITEMS = [
  {
    key: "azureOpenAI" as const,
    label: "Azure OpenAI",
    description: "GPT-4o for therapeutic insights & music suggestions",
    icon: Cloud,
  },
  {
    key: "azureSpeech" as const,
    label: "Azure Speech",
    description: "Text-to-speech for AI insight narration",
    icon: Volume2,
  },
  {
    key: "lstmModel" as const,
    label: "LSTM Classifier",
    description: "BiLSTM emotion classification model",
    icon: Brain,
  },
  {
    key: "avRegressor" as const,
    label: "A/V Regressor",
    description: "Arousal-Valence prediction from audio features",
    icon: Activity,
  },
];

export function SystemStatusPanel({
  status,
  isOpen,
  onClose,
}: SystemStatusPanelProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm"
            onClick={onClose}
          />

          {/* Panel */}
          <motion.div
            initial={{ opacity: 0, x: -320 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -320 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="fixed left-0 top-0 z-50 h-screen w-80 border-r bg-background shadow-xl"
          >
            <div className="flex h-16 items-center justify-between border-b px-6">
              <h2 className="text-sm font-semibold">System Status</h2>
              <Button variant="ghost" size="icon" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-1 p-4">
              {STATUS_ITEMS.map((item) => {
                const isActive = status[item.key];
                const Icon = item.icon;

                return (
                  <div
                    key={item.key}
                    className="flex items-start gap-3 rounded-lg p-3 transition-colors hover:bg-muted/50"
                  >
                    <div className="mt-0.5">
                      <Icon className="h-4 w-4 text-muted-foreground" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">
                          {item.label}
                        </span>
                        {isActive ? (
                          <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                        ) : (
                          <XCircle className="h-3.5 w-3.5 text-muted-foreground/50" />
                        )}
                      </div>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {item.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            <Separator />

            <div className="p-4">
              <p className="text-xs text-muted-foreground leading-relaxed">
                Services marked as inactive may require environment variables or
                trained model files. See the project README for setup
                instructions.
              </p>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
