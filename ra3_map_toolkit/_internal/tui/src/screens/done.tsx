import { Box, Text, useStdin } from "ink";

import { useInput } from "@/hooks/use-input";

import { Badge } from "@/components/ui/badge";
import { Banner } from "@/components/ui/banner";
import { Divider } from "@/components/ui/divider";
import { useTheme } from "@/components/ui/theme-provider";
import type { CompletedItem } from "./converting";

export function DoneScreen({
  success,
  fail,
  skipped,
  completed,
  inputDir,
  outputDir,
  onAgain,
  onQuit,
}: {
  success: number;
  fail: number;
  skipped: number;
  completed: CompletedItem[];
  inputDir: string;
  outputDir: string;
  onAgain: () => void;
  onQuit: () => void;
}) {
  const theme = useTheme();
  const { isRawModeSupported } = useStdin();

  // Nothing was attempted: no successes, no failures, no skipped.
  if (success === 0 && fail === 0 && skipped === 0) {
    return (
      <Box marginTop={1} flexDirection="column">
        <Divider label="Done" />
        <Box marginTop={1}>
          <Banner variant="warning" title="No maps found">
            Place .map files (or folders containing them) in {inputDir} and run
            again.
          </Banner>
        </Box>
        <KeysFooter onAgainLabel="rescan" />
        {isRawModeSupported && (
          <DoneKeyHandler onAgain={onAgain} onQuit={onQuit} />
        )}
      </Box>
    );
  }

  const variant = fail === 0 ? "success" : success === 0 ? "error" : "warning";
  return (
    <Box marginTop={1} flexDirection="column">
      <Divider label="Done" />
      <Box marginTop={1} flexDirection="row" gap={2}>
        <Badge variant="success">{`✓ ${success} succeeded`}</Badge>
        {fail > 0 && <Badge variant="error">{`✗ ${fail} failed`}</Badge>}
        {skipped > 0 && (
          <Badge variant="warning">{`⚠ ${skipped} skipped`}</Badge>
        )}
      </Box>
      <Box marginTop={1}>
        <Banner variant={variant} title="Conversion complete">
          {success} map(s) ready in {outputDir || "the output folder"}.
        </Banner>
      </Box>
      {fail > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Text bold color={theme.colors.error}>
            Failures:
          </Text>
          {completed
            .filter((c) => !c.success)
            .map((c, idx) => (
              <Text key={`${c.archonName}-${idx}`} color={theme.colors.error}>
                {"  "}• {c.name} — {c.error ?? "unknown error"}
              </Text>
            ))}
        </Box>
      )}
      <KeysFooter onAgainLabel="convert more" />
      {isRawModeSupported && (
        <DoneKeyHandler onAgain={onAgain} onQuit={onQuit} />
      )}
    </Box>
  );
}

function DoneKeyHandler({
  onAgain,
  onQuit,
}: {
  onAgain: () => void;
  onQuit: () => void;
}) {
  useInput((input) => {
    if (input === "a") onAgain();
    if (input === "q" || input === "\r") onQuit();
  });
  return null;
}

function KeysFooter({ onAgainLabel }: { onAgainLabel: string }) {
  const theme = useTheme();
  return (
    <Box marginTop={1} flexDirection="row" gap={2}>
      <Box gap={1}>
        <Text bold color={theme.colors.primary}>
          a
        </Text>
        <Text dimColor>{onAgainLabel}</Text>
      </Box>
      <Box gap={1}>
        <Text bold color={theme.colors.primary}>
          enter / q
        </Text>
        <Text dimColor>quit</Text>
      </Box>
    </Box>
  );
}
