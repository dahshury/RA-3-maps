import { useState } from "react";
import { Box, Text, useStdin } from "ink";

import { useInput } from "@/hooks/use-input";

import { Divider } from "@/components/ui/divider";
import { useTheme } from "@/components/ui/theme-provider";

export type ToolkitMode = "convert" | "skin" | "rotate" | "compose";

export interface ModeSelectScreenProps {
  onPick: (mode: ToolkitMode) => void;
  onQuit: () => void;
}

interface ModeRow {
  key: ToolkitMode;
  label: string;
  hint: string;
  details: string[];
}

const MODES: ModeRow[] = [
  {
    key: "convert",
    label: "Convert Maps",
    hint: "Batch Archon and/or match-restriction transforms",
    details: [
      "Pair builders with controllers to build Archon-style maps.",
      "Or apply No-Superweapons / No-Air / Inf-Only / Tanks-Only restrictions.",
      "Combine both for a single pipeline pass.",
    ],
  },
  {
    key: "skin",
    label: "Skin / Decompose Map",
    hint: "Split one source map into 8 layered strip variants (4 isolated + 4 cumulative)",
    details: [
      "Useful for diagnosing what each visual layer contributes.",
      "Outputs blends-off, textures-off, objects-off, flat, plus a cumulative skeleton.",
      "Each variant is also rendered to a PNG minimap.",
    ],
  },
  {
    key: "rotate",
    label: "Rotate / Flip Maps",
    hint: "Apply 90°/180°/270° rotations and X/Y flips to one map or a whole folder",
    details: [
      "Rotates heightmap, blends, passability, objects (positions + angles), water/triggers.",
      "Flips include a +180° fixup for asymmetric meshes (cliff walls / sea cliff walls).",
      "Pick one or more ops; each writes its own output map.",
    ],
  },
  {
    key: "compose",
    label: "Compose / Stitch Maps",
    hint: "Compose 1–3 source maps into one output via a layout preset (row, col, triangle, duplicate)",
    details: [
      "Six layout presets: row · col · triangle_top/bottom/left/right · duplicate (Nx×Ny tile of one source).",
      "Each source must have ≥ 2 players; total players capped at 6 (RA3 lobby cap).",
      "Stitches terrain, objects, players, water, triggers; merges texture tables across sources.",
    ],
  },
];

export function ModeSelectScreen({ onPick, onQuit }: ModeSelectScreenProps) {
  const theme = useTheme();
  const [cursor, setCursor] = useState(0);
  const { isRawModeSupported } = useStdin();

  return (
    <Box flexDirection="column">
      <Box marginTop={1}>
        <Divider label="Choose pipeline" />
      </Box>

      <Box marginTop={1} flexDirection="column">
        {MODES.map((m, idx) => {
          const isCursor = idx === cursor;
          return (
            <Box key={m.key} flexDirection="column" marginTop={idx > 0 ? 1 : 0}>
              <Box gap={1}>
                <Text color={isCursor ? theme.colors.primary : undefined}>
                  {isCursor ? "›" : " "}
                </Text>
                <Text
                  bold={isCursor}
                  color={isCursor ? theme.colors.primary : undefined}
                >
                  {m.label}
                </Text>
                <Text dimColor> · {m.hint}</Text>
              </Box>
              {isCursor && (
                <Box marginLeft={4} flexDirection="column">
                  {m.details.map((d) => (
                    <Text key={d} dimColor>
                      {d}
                    </Text>
                  ))}
                </Box>
              )}
            </Box>
          );
        })}
      </Box>

      <Box marginTop={1}>
        <Divider />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={2}>
        <Box gap={1}>
          <Text color={theme.colors.primary} bold>
            ↑↓ / j k
          </Text>
          <Text dimColor>navigate</Text>
        </Box>
        <Box gap={1}>
          <Text color={theme.colors.primary} bold>
            enter
          </Text>
          <Text dimColor>select</Text>
        </Box>
        <Box gap={1}>
          <Text color={theme.colors.primary} bold>
            q
          </Text>
          <Text dimColor>quit</Text>
        </Box>
      </Box>

      {isRawModeSupported && (
        <KeyHandler
          cursor={cursor}
          setCursor={setCursor}
          onPick={() => onPick(MODES[cursor]!.key)}
          onQuit={onQuit}
        />
      )}
    </Box>
  );
}

function KeyHandler({
  cursor,
  setCursor,
  onPick,
  onQuit,
}: {
  cursor: number;
  setCursor: (n: number | ((prev: number) => number)) => void;
  onPick: () => void;
  onQuit: () => void;
}) {
  useInput((input, key) => {
    if (input === "q" || (key.ctrl && input === "c")) {
      onQuit();
      return;
    }
    if (key.return) {
      onPick();
      return;
    }
    if (key.upArrow || input === "k") {
      setCursor((c) => Math.max(0, c - 1));
    } else if (key.downArrow || input === "j") {
      setCursor((c) => Math.min(MODES.length - 1, c + 1));
    } else if (input === "1") {
      setCursor(0);
    } else if (input === "2") {
      setCursor(1);
    } else if (input === "3") {
      setCursor(2);
    } else if (input === "4") {
      setCursor(3);
    }
  });
  return null;
}
