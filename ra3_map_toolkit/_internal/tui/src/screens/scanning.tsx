import { Box, Text } from "ink";

import { StatusMessage } from "@/components/ui/status-message";

export function ScanningScreen({
  scannedCount,
  lastFile,
}: {
  scannedCount: number;
  lastFile?: string;
}) {
  return (
    <Box marginTop={1} flexDirection="column">
      <StatusMessage variant="loading">
        Scanning input folder for .map files…
      </StatusMessage>
      {scannedCount > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Text dimColor>
            Analyzed {scannedCount} file(s)
            {lastFile ? `: ${lastFile}` : ""}
          </Text>
        </Box>
      )}
    </Box>
  );
}
