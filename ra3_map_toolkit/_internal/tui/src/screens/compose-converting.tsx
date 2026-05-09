import { Box, Text } from "ink";

import { Badge } from "@/components/ui/badge";
import { Divider } from "@/components/ui/divider";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Spinner } from "@/components/ui/spinner";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";
import {
  COMPOSE_STEP_LABEL,
  type ComposePreset,
  type ComposeStep,
} from "@/lib/events";

const STEP_ORDER: ComposeStep[] = ["parse", "layout", "compose", "save", "tga"];

export interface ComposeConvertingState {
  preset: ComposePreset;
  maps: string[];           // basenames of all source maps
  outputDir: string;
  step: ComposeStep | null;
  detail: string | null;
  done: boolean;
  success: boolean | null;  // null while running
  error?: string;
}

export interface ComposeCompletedItem {
  preset: ComposePreset;
  maps: string[];
  success: boolean;
  error?: string;
}

export function ComposeConvertingScreen({
  state,
  recent,
}: {
  state: ComposeConvertingState | null;
  recent: ComposeCompletedItem[];
}) {
  const theme = useTheme();

  if (!state) {
    return (
      <Box marginTop={1}>
        <StatusMessage variant="loading">Preparing…</StatusMessage>
      </Box>
    );
  }

  const stepIdx = state.step ? STEP_ORDER.indexOf(state.step) : -1;
  const overallTotal = STEP_ORDER.length;
  const overallValue = state.done
    ? overallTotal
    : Math.max(0, stepIdx);

  return (
    <Box flexDirection="column">
      <Box marginTop={1}>
        <Divider label="Composing" />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{state.preset}</Badge>
        {state.maps.map((name, i) => (
          <Box key={`${name}-${i}`} gap={1}>
            <Text color={theme.colors.primary} bold>
              {(["A", "B", "C"][i] ?? "?")}
            </Text>
            <Text>{name}</Text>
          </Box>
        ))}
        <Text dimColor>→</Text>
        <Text color={theme.colors.accent} bold>
          {state.outputDir}
        </Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Text>Pipeline progress</Text>
        <ProgressBar value={overallValue} total={overallTotal} width={40} />
      </Box>

      <Box marginTop={1} flexDirection="column">
        {STEP_ORDER.map((step, idx) => {
          const isCurrent = !state.done && state.step === step;
          const isDone = state.done || idx < stepIdx;
          const isPending = !isCurrent && !isDone;
          const icon = isDone ? "✓" : isCurrent ? "•" : "·";
          const color = isDone
            ? theme.colors.success
            : isCurrent
            ? theme.colors.primary
            : theme.colors.mutedForeground;
          return (
            <Box key={step} gap={1}>
              <Text color={color} bold>
                {icon}
              </Text>
              <Text color={isPending ? theme.colors.mutedForeground : color}
                    bold={isCurrent}>
                {COMPOSE_STEP_LABEL[step]}
              </Text>
              {isCurrent && state.detail && (
                <Text dimColor>· {state.detail}</Text>
              )}
            </Box>
          );
        })}
      </Box>

      {!state.done && (
        <Box marginTop={1} flexDirection="row" gap={1}>
          <Spinner type="dots" />
          <Text>
            {state.step ? COMPOSE_STEP_LABEL[state.step] : "Starting…"}
            {state.detail ? ` · ${state.detail}` : ""}
          </Text>
        </Box>
      )}

      {recent.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Divider label="Recent" />
          <Box marginTop={1} flexDirection="column">
            {recent.map((c, idx) => (
              <StatusMessage
                key={`${c.preset}-${idx}`}
                variant={c.success ? "success" : "error"}
              >
                {c.preset} · {c.maps.join(" + ")}
                {!c.success && c.error ? ` — ${c.error}` : ""}
              </StatusMessage>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}
