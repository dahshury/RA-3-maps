import { Box, Text } from "ink";

import { Badge } from "@/components/ui/badge";
import { Divider } from "@/components/ui/divider";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Spinner } from "@/components/ui/spinner";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";
import {
  STEP_LABEL,
  STEP_ORDER,
  type ConvertStep,
} from "@/lib/events";

export interface ConvertingState {
  name: string;
  archonName: string;
  index: number;
  total: number;
  playerCount: number;
  applyArchon: boolean;
  variation: string;
  step: ConvertStep | null;
  completedSteps: Set<ConvertStep>;
}

export interface CompletedItem {
  name: string;
  archonName: string;
  success: boolean;
  error?: string;
}

export function ConvertingScreen({
  current,
  successCount,
  failCount,
  recent,
}: {
  current: ConvertingState | null;
  successCount: number;
  failCount: number;
  recent: CompletedItem[];
}) {
  return (
    <Box flexDirection="column">
      {current ? (
        <CurrentJob
          current={current}
          successCount={successCount}
          failCount={failCount}
        />
      ) : (
        <Box marginTop={1}>
          <StatusMessage variant="loading">Preparing…</StatusMessage>
        </Box>
      )}

      {recent.length > 0 && <RecentCompletions completed={recent} />}
    </Box>
  );
}

function CurrentJob({
  current,
  successCount,
  failCount,
}: {
  current: ConvertingState;
  successCount: number;
  failCount: number;
}) {
  const theme = useTheme();
  const overallTotal = current.total;
  const overallValue = successCount + failCount;
  const stepIdx = current.step ? STEP_ORDER.indexOf(current.step) : -1;
  const stepCount = STEP_ORDER.length;
  const stepValue = stepIdx === -1 ? 0 : stepIdx + 1;

  return (
    <Box marginTop={1} flexDirection="column">
      <Divider label="Converting" />

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{`${current.index}/${current.total}`}</Badge>
        <Text bold>{current.name}</Text>
        <Text dimColor>→</Text>
        <Text color={theme.colors.accent} bold>
          {current.archonName}
        </Text>
      </Box>

      <Box flexDirection="row" gap={2} marginTop={0}>
        <Text dimColor>Players: {current.playerCount}</Text>
        <Text dimColor>
          Pipeline:{" "}
          {[
            current.applyArchon ? "archon" : null,
            current.variation || null,
          ]
            .filter(Boolean)
            .join(" + ") || "(none)"}
        </Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Text>Overall progress</Text>
        <ProgressBar value={overallValue} total={overallTotal} width={40} />
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Box flexDirection="row" gap={1}>
          <Spinner type="dots" />
          <Text>{current.step ? STEP_LABEL[current.step] : "Starting…"}</Text>
        </Box>
        <ProgressBar
          value={stepValue}
          total={stepCount}
          width={40}
          color={theme.colors.success}
        />
      </Box>

      <Box marginTop={1} flexDirection="column">
        {STEP_ORDER.map((s) => {
          const done = current.completedSteps.has(s);
          const active = current.step === s;
          let icon = "○";
          let color: string | undefined;
          if (done) {
            icon = "✓";
            color = theme.colors.success;
          } else if (active) {
            icon = "◐";
            color = theme.colors.primary;
          }
          return (
            <Box key={s} flexDirection="row" gap={1}>
              <Text color={color}>{icon}</Text>
              <Text
                color={active ? theme.colors.primary : done ? undefined : theme.colors.mutedForeground}
                dimColor={!active && !done}
              >
                {STEP_LABEL[s]}
              </Text>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}

function RecentCompletions({ completed }: { completed: CompletedItem[] }) {
  return (
    <Box marginTop={1} flexDirection="column">
      <Divider label="Recent" />
      <Box marginTop={1} flexDirection="column">
        {completed.map((c, idx) => (
          <StatusMessage
            key={`${c.archonName}-${idx}`}
            variant={c.success ? "success" : "error"}
          >
            {c.archonName}
            {!c.success && c.error ? ` — ${c.error}` : ""}
          </StatusMessage>
        ))}
      </Box>
    </Box>
  );
}
