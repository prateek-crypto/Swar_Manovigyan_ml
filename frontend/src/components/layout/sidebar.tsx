"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Music,
  Brain,
  Upload,
  Sliders,
  Hand,
  ChevronLeft,
  ChevronRight,
  Activity,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { InputMethod, SystemStatus } from "@/types";

interface SidebarProps {
  inputMethod: InputMethod;
  onInputMethodChange: (method: InputMethod) => void;
  status: SystemStatus;
  onStatusClick: () => void;
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
}

const INPUT_METHODS = [
  {
    id: "manual" as InputMethod,
    label: "Manual Selection",
    description: "Set arousal & valence",
    icon: Hand,
  },
  {
    id: "features" as InputMethod,
    label: "Audio Features",
    description: "Adjust feature sliders",
    icon: Sliders,
  },
  {
    id: "upload" as InputMethod,
    label: "Audio Upload",
    description: "Upload audio file",
    icon: Upload,
  },
];

export function Sidebar({
  inputMethod,
  onInputMethodChange,
  status,
  onStatusClick,
  collapsed: controlledCollapsed,
  onCollapsedChange,
}: SidebarProps) {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const collapsed = controlledCollapsed ?? internalCollapsed;
  const setCollapsed = onCollapsedChange ?? setInternalCollapsed;

  const activeServices = [
    status.azureOpenAI,
    status.azureSpeech,
    status.lstmModel,
    status.avRegressor,
  ].filter(Boolean).length;

  return (
    <TooltipProvider delayDuration={0}>
      <motion.aside
        className={cn(
          "flex h-screen flex-col border-r bg-card/50 backdrop-blur-sm",
          "transition-all duration-300 ease-in-out"
        )}
        animate={{ width: collapsed ? 68 : 264 }}
      >
        {/* Logo */}
        <div className="flex h-16 items-center gap-3 border-b px-4">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Music className="h-5 w-5" />
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: "auto" }}
                exit={{ opacity: 0, width: 0 }}
                className="overflow-hidden whitespace-nowrap"
              >
                <p className="text-sm font-semibold tracking-tight">
                  Swar Manovigyan
                </p>
                <p className="text-[10px] text-muted-foreground">
                  Emotion-Aware Music
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Input Methods */}
        <div className="flex-1 overflow-y-auto px-3 py-4">
          <AnimatePresence>
            {!collapsed && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="mb-2 px-2 text-[11px] font-medium uppercase tracking-wider text-muted-foreground"
              >
                Input Method
              </motion.p>
            )}
          </AnimatePresence>

          <nav className="space-y-1">
            {INPUT_METHODS.map((method) => {
              const isActive = inputMethod === method.id;
              const Icon = method.icon;

              const button = (
                <button
                  key={method.id}
                  onClick={() => onInputMethodChange(method.id)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all",
                    isActive
                      ? "bg-primary/10 text-primary font-medium"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground"
                  )}
                >
                  <Icon
                    className={cn(
                      "h-4 w-4 shrink-0",
                      isActive && "text-primary"
                    )}
                  />
                  {!collapsed && (
                    <div className="text-left">
                      <p className="leading-none">{method.label}</p>
                      <p className="mt-0.5 text-[11px] text-muted-foreground">
                        {method.description}
                      </p>
                    </div>
                  )}
                </button>
              );

              if (collapsed) {
                return (
                  <Tooltip key={method.id}>
                    <TooltipTrigger asChild>{button}</TooltipTrigger>
                    <TooltipContent side="right">
                      <p>{method.label}</p>
                    </TooltipContent>
                  </Tooltip>
                );
              }

              return button;
            })}
          </nav>
        </div>

        {/* Bottom Section */}
        <div className="border-t px-3 py-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                onClick={onStatusClick}
                className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                <Activity className="h-4 w-4 shrink-0" />
                {!collapsed && (
                  <div className="flex flex-1 items-center justify-between">
                    <span>System Status</span>
                    <Badge
                      variant={activeServices > 2 ? "success" : "warning"}
                      className="text-[10px]"
                    >
                      {activeServices}/4
                    </Badge>
                  </div>
                )}
              </button>
            </TooltipTrigger>
            {collapsed && (
              <TooltipContent side="right">
                <p>System Status ({activeServices}/4 active)</p>
              </TooltipContent>
            )}
          </Tooltip>

          <Separator className="my-2" />

          <Button
            variant="ghost"
            size="icon"
            className="w-full"
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>
      </motion.aside>
    </TooltipProvider>
  );
}
