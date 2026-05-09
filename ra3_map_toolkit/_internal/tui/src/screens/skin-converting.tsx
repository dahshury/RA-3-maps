import { Box, Text } from "ink";

import { Badge } from "@/components/ui/badge";
import { Divider } from "@/components/ui/divider";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Spinner } from "@/components/ui/spinner";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";
import {
  SKIN_VARIANT_LABELS,
  SKIN_VARIANT_ORDER,
  type SkinVariantName,
} from "@/lib/events";

export interface SkinConvertingState {
  sourceName: string;
  outputDir: string;
  current: SkinVariantName | null;
  step: string | null;
  index: number;
  total: number;
  completed: Set<SkinVariantName>;
  failed: Set<SkinVariantName>;
  successCount: number;
  failCount: number;
}

export interface SkinCompletedItem {
  name: SkinVariantName;
  success: boolean;
  error?: string;
}

export function SkinConvertingScreen({
  state,
  recent,
}: {
  state: SkinConvertingState | null;
  recent: SkinCompletedItem[];
}) {
  const theme = useTheme();

  if (!state) {
    return (
      <Box marginTop={1}>
        <StatusMessage variant="loading">Preparing…</StatusMessage>
      </Box>
    );
  }

  const overallTotal = state.total;
  const overallValue = state.successCount + state.failCount;

  return (
    <Box flexDirection="column">
      <Box marginTop={1}>
        <Divider label="Skinning" />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{`${state.index}/${state.total}`}</Badge>
        <Text bold>{state.sourceName}</Text>
        <Text dimColor>→</Text>
        <Text color={theme.colors.accent} bold>
          {state.outputDir}
        </Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Text>Overall progress</Text>
        <ProgressBar value={overallValue} total={overallTotal} width={40} />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Spinner type="dots" />
        <Text>
          {state.current
            ? `${SKIN_VARIANT_LABELS[state.current]}${
                state.step ? ` — ${state.step}` : ""
              }`
            : "Starting…"}
        </Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        {SKIN_VARIANT_ORDER.map((v) => {
          const done = state.completed.has(v);
          const failed = state.failed.has(v);
          const active = state.current === v;
          let icon = "○";
          let color: string | undefined;
          if (failed) {
            icon = "✗";
            color = theme.colors.error;
          } else if (done) {
            icon = "✓";
            color = theme.colors.success;
          } else if (active) {
            icon = "◐";
            color = theme.colors.primary;
          }
          return (
            <Box key={v} flexDirection="row" gap={1}>
              <Text color={color}>{icon}</Text>
              <Text
                color={
                  active
                    ? theme.colors.primary
                    : done || failed
                    ? undefined
                    : theme.colors.mutedForeground
                }
                dimColor={!active && !done && !failed}
              >
                {SKIN_VARIANT_LABELS[v]}
              </Text>
            </Box>
          );
        })}
      </Box>

      {recent.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Divider label="Recent" />
          <Box marginTop={1} flexDirection="column">
            {recent.map((c, idx) => (
              <StatusMessage
                key={`${c.name}-${idx}`}
                variant={c.success ? "success" : "error"}
              >
                {SKIN_VARIANT_LABELS[c.name]}
                {!c.success && c.error ? ` — ${c.error}` : ""}
              </StatusMessage>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}
