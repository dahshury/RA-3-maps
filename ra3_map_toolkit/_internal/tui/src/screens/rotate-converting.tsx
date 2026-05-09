import { Box, Text } from "ink";

import { Badge } from "@/components/ui/badge";
import { Divider } from "@/components/ui/divider";
import { ProgressBar } from "@/components/ui/progress-bar";
import { Spinner } from "@/components/ui/spinner";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";
import {
  ROTATE_OPS,
  ROTATE_STEP_LABEL,
  type RotateOp,
  type RotateStep,
} from "@/lib/events";

export interface RotateConvertingState {
  sourceName: string;     // for single-source flow; the *current* one for batch
  outputDir: string;
  currentOp: RotateOp | null;
  currentSource: string;
  step: RotateStep | null;
  index: number;
  total: number;
  successCount: number;
  failCount: number;
}

export interface RotateCompletedItem {
  source: string;
  op: RotateOp;
  success: boolean;
  error?: string;
}

export function RotateConvertingScreen({
  state,
  recent,
}: {
  state: RotateConvertingState | null;
  recent: RotateCompletedItem[];
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

  const opLabel = (k: RotateOp) =>
    ROTATE_OPS.find((r) => r.key === k)?.label ?? k;

  return (
    <Box flexDirection="column">
      <Box marginTop={1}>
        <Divider label="Rotating / Flipping" />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{`${state.index}/${state.total}`}</Badge>
        <Text bold>{state.currentSource || state.sourceName}</Text>
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
          {state.currentOp
            ? `${opLabel(state.currentOp)}${
                state.step ? ` — ${ROTATE_STEP_LABEL[state.step]}` : ""
              }`
            : "Starting…"}
        </Text>
      </Box>

      {recent.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Divider label="Recent" />
          <Box marginTop={1} flexDirection="column">
            {recent.map((c, idx) => (
              <StatusMessage
                key={`${c.source}-${c.op}-${idx}`}
                variant={c.success ? "success" : "error"}
              >
                {c.source} · {opLabel(c.op)}
                {!c.success && c.error ? ` — ${c.error}` : ""}
              </StatusMessage>
            ))}
          </Box>
        </Box>
      )}
    </Box>
  );
}
