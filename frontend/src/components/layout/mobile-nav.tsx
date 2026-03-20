"use client";

import { Activity, Hand, Music, Sliders, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { InputMethod, SystemStatus } from "@/types";

interface MobileNavProps {
  inputMethod: InputMethod;
  onInputMethodChange: (method: InputMethod) => void;
  status: SystemStatus;
  onStatusClick: () => void;
}

const NAV_ITEMS = [
  { id: "manual" as InputMethod, label: "Manual", icon: Hand },
  { id: "features" as InputMethod, label: "Features", icon: Sliders },
  { id: "upload" as InputMethod, label: "Upload", icon: Upload },
] as const;

export function MobileNav({
  inputMethod,
  onInputMethodChange,
  status,
  onStatusClick,
}: MobileNavProps) {
  const activeServices = [
    status.azureOpenAI,
    status.azureSpeech,
    status.lstmModel,
    status.avRegressor,
  ].filter(Boolean).length;

  return (
    <nav className="fixed inset-x-0 bottom-0 z-50 border-t bg-background/95 backdrop-blur-md md:hidden">
      <div className="flex items-center justify-around px-2 py-1.5">
        {NAV_ITEMS.map((item) => {
          const isActive = inputMethod === item.id;
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              onClick={() => onInputMethodChange(item.id)}
              className={cn(
                "flex flex-1 flex-col items-center gap-0.5 rounded-lg px-2 py-1.5 text-[11px] transition-colors",
                isActive
                  ? "text-primary font-medium"
                  : "text-muted-foreground"
              )}
            >
              <Icon className={cn("h-5 w-5", isActive && "text-primary")} />
              {item.label}
            </button>
          );
        })}
        <button
          onClick={onStatusClick}
          className="flex flex-1 flex-col items-center gap-0.5 rounded-lg px-2 py-1.5 text-[11px] text-muted-foreground transition-colors"
        >
          <div className="relative">
            <Activity className="h-5 w-5" />
            <Badge
              variant={activeServices > 2 ? "success" : "warning"}
              className="absolute -right-2.5 -top-1.5 h-3.5 min-w-[14px] px-0.5 text-[8px]"
            >
              {activeServices}
            </Badge>
          </div>
          Status
        </button>
      </div>
      {/* Safe area for notched phones */}
      <div className="h-[env(safe-area-inset-bottom)]" />
    </nav>
  );
}
