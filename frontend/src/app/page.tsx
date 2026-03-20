"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Loader2,
  Sparkles,
  AlertCircle,
  Music,
  BarChart3,
  Target,
  Hand,
  Sliders,
  Upload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import { MobileNav } from "@/components/layout/mobile-nav";
import { EmotionSelector } from "@/components/emotion/emotion-selector";
import { AudioFeaturesForm } from "@/components/emotion/audio-features-form";
import { AudioUpload } from "@/components/emotion/audio-upload";
import { EmotionResults } from "@/components/emotion/emotion-results";
import { AnalysisLoading } from "@/components/emotion/analysis-loading";
import { Recommendations } from "@/components/music/recommendations";
import { ModelComparison } from "@/components/models/model-comparison";
import { SystemStatusPanel } from "@/components/models/system-status";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useEmotionAnalysis } from "@/hooks/use-emotion-analysis";
import { getSystemStatus } from "@/services/api";
import { AUDIO_FEATURE_RANGES } from "@/lib/constants";
import type { InputMethod, AudioFeatures, SystemStatus } from "@/types";

const DEFAULT_FEATURES: AudioFeatures = Object.fromEntries(
  Object.entries(AUDIO_FEATURE_RANGES).map(([k, v]) => [k, v.default])
) as unknown as AudioFeatures;

export default function HomePage() {
  const [inputMethod, setInputMethod] = useState<InputMethod>("manual");
  const [arousal, setArousal] = useState(0.5);
  const [valence, setValence] = useState(0.5);
  const [features, setFeatures] = useState<AudioFeatures>(DEFAULT_FEATURES);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [statusOpen, setStatusOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    azureOpenAI: false,
    azureSpeech: false,
    lstmModel: false,
    avRegressor: false,
    baselineModels: false,
  });

  const {
    isLoading,
    error,
    emotion,
    recommendations,
    insight,
    modelComparison,
    analyze,
    reset,
  } = useEmotionAnalysis();

  useEffect(() => {
    getSystemStatus().then(setSystemStatus);
  }, []);

  const handleAnalyze = useCallback(() => {
    switch (inputMethod) {
      case "manual":
        analyze("manual", { arousal, valence });
        break;
      case "features":
        analyze("features", { features });
        break;
      case "upload":
        if (audioFile) analyze("upload", { file: audioFile });
        break;
    }
  }, [inputMethod, arousal, valence, features, audioFile, analyze]);

  const handleMethodChange = useCallback(
    (method: InputMethod) => {
      setInputMethod(method);
      reset();
    },
    [reset]
  );

  useEffect(() => {
    if (inputMethod === "manual") {
      const timer = setTimeout(() => {
        analyze("manual", { arousal, valence });
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [arousal, valence, inputMethod, analyze]);

  const canAnalyze =
    (inputMethod === "features") ||
    (inputMethod === "upload" && audioFile !== null);

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Desktop sidebar - hidden on mobile */}
      <div className="hidden md:block">
        <Sidebar
          inputMethod={inputMethod}
          onInputMethodChange={handleMethodChange}
          status={systemStatus}
          onStatusClick={() => setStatusOpen(true)}
        />
      </div>

      {/* Main content */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />

        <main className="flex-1 overflow-y-auto pb-20 md:pb-0">
          <div className="mx-auto max-w-5xl px-4 py-6 sm:px-6 sm:py-8">
            {/* Mobile input method selector */}
            <MobileInputSelector
              inputMethod={inputMethod}
              onInputMethodChange={handleMethodChange}
            />

            {/* Input Section */}
            <AnimatePresence mode="wait">
              <motion.div
                key={inputMethod}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -12 }}
                transition={{ duration: 0.2 }}
              >
                {inputMethod === "manual" && (
                  <EmotionSelector
                    arousal={arousal}
                    valence={valence}
                    onArousalChange={setArousal}
                    onValenceChange={setValence}
                  />
                )}

                {inputMethod === "features" && (
                  <AudioFeaturesForm
                    features={features}
                    onFeaturesChange={setFeatures}
                  />
                )}

                {inputMethod === "upload" && (
                  <AudioUpload
                    file={audioFile}
                    onFileChange={setAudioFile}
                  />
                )}
              </motion.div>
            </AnimatePresence>

            {/* Analyze Button (for non-manual modes) */}
            {inputMethod !== "manual" && (
              <div className="mt-6 flex justify-center">
                <Button
                  size="lg"
                  onClick={handleAnalyze}
                  disabled={isLoading || !canAnalyze}
                  className="gap-2 px-8"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      Analyze Emotion
                    </>
                  )}
                </Button>
              </div>
            )}

            {/* Error State */}
            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-6 flex items-center gap-3 rounded-lg border border-destructive/50 bg-destructive/10 p-4"
              >
                <AlertCircle className="h-5 w-5 shrink-0 text-destructive" />
                <div>
                  <p className="text-sm font-medium text-destructive">
                    Analysis Error
                  </p>
                  <p className="text-xs text-muted-foreground">{error}</p>
                </div>
              </motion.div>
            )}

            {/* Loading State */}
            {isLoading && !emotion && <AnalysisLoading />}

            {/* Results Section */}
            {emotion && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="mt-8"
              >
                <Tabs defaultValue="results" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="results" className="gap-2">
                      <Target className="h-3.5 w-3.5" />
                      Results
                    </TabsTrigger>
                    <TabsTrigger value="recommendations" className="gap-2">
                      <Music className="h-3.5 w-3.5" />
                      Music
                    </TabsTrigger>
                    <TabsTrigger value="models" className="gap-2">
                      <BarChart3 className="h-3.5 w-3.5" />
                      Models
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="results" className="mt-4">
                    <EmotionResults result={emotion} insight={insight} />
                  </TabsContent>

                  <TabsContent value="recommendations" className="mt-4">
                    {recommendations ? (
                      <Recommendations data={recommendations} />
                    ) : (
                      <EmptyState message="Recommendations will appear here after analysis" />
                    )}
                  </TabsContent>

                  <TabsContent value="models" className="mt-4">
                    <ModelComparison predictions={modelComparison} />
                  </TabsContent>
                </Tabs>
              </motion.div>
            )}

            {/* Footer */}
            <footer className="mt-12 border-t pt-6 pb-8">
              <p className="text-center text-xs text-muted-foreground">
                Built by Adit Jain, Mehir Singh and Ayush Pandey
              </p>
            </footer>
          </div>
        </main>
      </div>

      {/* Mobile bottom nav */}
      <MobileNav
        inputMethod={inputMethod}
        onInputMethodChange={handleMethodChange}
        status={systemStatus}
        onStatusClick={() => setStatusOpen(true)}
      />

      {/* System Status Panel */}
      <SystemStatusPanel
        status={systemStatus}
        isOpen={statusOpen}
        onClose={() => setStatusOpen(false)}
      />
    </div>
  );
}

const MOBILE_INPUT_OPTIONS = [
  { id: "manual" as InputMethod, label: "Manual", icon: Hand },
  { id: "features" as InputMethod, label: "Features", icon: Sliders },
  { id: "upload" as InputMethod, label: "Upload", icon: Upload },
] as const;

function MobileInputSelector({
  inputMethod,
  onInputMethodChange,
}: {
  inputMethod: InputMethod;
  onInputMethodChange: (method: InputMethod) => void;
}) {
  return (
    <div className="mb-6 flex gap-2 overflow-x-auto md:hidden">
      {MOBILE_INPUT_OPTIONS.map((opt) => {
        const isActive = inputMethod === opt.id;
        const Icon = opt.icon;
        return (
          <button
            key={opt.id}
            onClick={() => onInputMethodChange(opt.id)}
            className={cn(
              "flex shrink-0 items-center gap-1.5 rounded-full px-4 py-2 text-sm font-medium transition-colors",
              isActive
                ? "bg-primary text-primary-foreground"
                : "bg-muted text-muted-foreground hover:bg-accent"
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <Music className="mb-3 h-8 w-8 text-muted-foreground/40" />
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
