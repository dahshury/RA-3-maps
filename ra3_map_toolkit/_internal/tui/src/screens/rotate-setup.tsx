import { useMemo, useState } from "react";
import { Box, Text, useStdin } from "ink";

import { useInput } from "@/hooks/use-input";

import { Badge } from "@/components/ui/badge";
import { Banner } from "@/components/ui/banner";
import { Divider } from "@/components/ui/divider";
import { KeyValue } from "@/components/ui/key-value";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";

import { ROTATE_OPS, type RotateOp } from "@/lib/events";
import type { ScannedMap } from "./setup";

export interface RotateSetupOptions {
  source: ScannedMap | null;
  batchAll: boolean;
  ops: RotateOp[];
  compress: boolean;
}

export const defaultRotateSetupOptions = (): RotateSetupOptions => ({
  source: null,
  batchAll: false,
  ops: ["rot90cw"],
  compress: true,
});

export interface RotateSetupScreenProps {
  inputDir: string;
  outputDir: string;
  maps: ScannedMap[];
  skipped: { name: string; reason: string }[];
  options: RotateSetupOptions;
  onChange: (opts: RotateSetupOptions) => void;
  onStart: () => void;
  onBack: () => void;
  onRescan: () => void;
  onQuit: () => void;
}

type FocusZone = "maps" | "ops" | "options";

export function RotateSetupScreen({
  inputDir,
  outputDir,
  maps,
  skipped,
  options,
  onChange,
  onStart,
  onBack,
  onRescan,
  onQuit,
}: RotateSetupScreenProps) {
  const theme = useTheme();
  const [zone, setZone] = useState<FocusZone>("maps");
  const [mapCursor, setMapCursor] = useState(0);
  const [opCursor, setOpCursor] = useState(0);
  const [optCursor, setOptCursor] = useState(0);

  const { isRawModeSupported } = useStdin();

  const setSource = (m: ScannedMap | null) =>
    onChange({ ...options, source: m });

  const toggleOp = (op: RotateOp) => {
    const has = options.ops.includes(op);
    const next = has ? options.ops.filter((o) => o !== op) : [...options.ops, op];
    onChange({ ...options, ops: next });
  };

  const compressRow = {
    label: "Compress output",
    value: options.compress ? "ON" : "OFF",
    on: options.compress,
    toggle: () => onChange({ ...options, compress: !options.compress }),
    hint: "Standard for in-game maps. Turn OFF for WorldBuilder debugging.",
  };
  const batchRow = {
    label: "Batch every map in input folder",
    value: options.batchAll ? "ON" : "OFF",
    on: options.batchAll,
    toggle: () => onChange({ ...options, batchAll: !options.batchAll }),
    hint: "When ON, applies the selected ops to every .map under input. Single-source pick is ignored.",
  };
  const optionRows = [batchRow, compressRow];

  const canStart =
    options.ops.length > 0 && (options.batchAll || options.source !== null);
  const totalOutputs =
    options.ops.length * (options.batchAll ? maps.length : options.source ? 1 : 0);

  const counts = useMemo(() => {
    let p2 = 0; let p3 = 0;
    for (const m of maps) {
      if (m.players === 3) p3++;
      else p2++;
    }
    return { p2, p3 };
  }, [maps]);

  return (
    <Box flexDirection="column">
      <Box marginTop={1}>
        <KeyValue
          items={[
            { key: "Pipeline", value: "Rotate / Flip" },
            { key: "Input", value: inputDir },
            { key: "Output", value: outputDir },
            {
              key: "Source",
              value: options.batchAll
                ? `(batch: ${maps.length} map(s))`
                : options.source
                ? options.source.name
                : "(none selected)",
            },
            {
              key: "Ops",
              value: options.ops.length === 0 ? "(none)" : options.ops.join(", "),
            },
          ]}
        />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{`Found ${maps.length}`}</Badge>
        <Badge variant="success">{`2p: ${counts.p2}`}</Badge>
        <Badge variant="success">{`3p: ${counts.p3}`}</Badge>
        {skipped.length > 0 && (
          <Badge variant="warning">{`Skipped: ${skipped.length}`}</Badge>
        )}
        <Badge variant={canStart ? "info" : "warning"}>{`Outputs: ${totalOutputs}`}</Badge>
      </Box>

      {maps.length === 0 ? (
        <Box marginTop={1}>
          <Banner variant="warning" title="No maps found">
            Place .map files (or folders containing them) in {inputDir} and press
            r to rescan.
          </Banner>
        </Box>
      ) : (
        <Box marginTop={1} flexDirection="column">
          <Box flexDirection="row" gap={1}>
            <Text bold color={zone === "maps" ? theme.colors.primary : undefined}>
              {zone === "maps" ? "▶" : " "} Pick source
            </Text>
            <Text dimColor>
              {options.batchAll ? "(batch ON — pick is ignored)" : "(space to select one map)"}
            </Text>
          </Box>
          <SourceList
            maps={maps}
            cursor={mapCursor}
            active={zone === "maps" && !options.batchAll}
            disabled={options.batchAll}
            selectedPath={options.source?.path ?? options.source?.name ?? null}
          />
        </Box>
      )}

      <Box marginTop={1} flexDirection="column">
        <Box flexDirection="row" gap={1}>
          <Text bold color={zone === "ops" ? theme.colors.primary : undefined}>
            {zone === "ops" ? "▶" : " "} Pick rotations / flips
          </Text>
          <Text dimColor>(space toggles; multi-select)</Text>
        </Box>
        <OpsList
          cursor={opCursor}
          active={zone === "ops"}
          selected={options.ops}
        />
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Box flexDirection="row" gap={1}>
          <Text bold color={zone === "options" ? theme.colors.primary : undefined}>
            {zone === "options" ? "▶" : " "} Options
          </Text>
        </Box>
        <RotateOptionList
          rows={optionRows}
          cursor={optCursor}
          active={zone === "options"}
        />
      </Box>

      <Box marginTop={1}>
        <Divider />
      </Box>

      <Footer canStart={canStart} zone={zone} />

      {!canStart && (
        <Box marginTop={1}>
          <StatusMessage variant="warning">
            {options.ops.length === 0
              ? "Pick at least one rotation / flip op (space)."
              : "Pick a source map (or enable batch mode)."}
          </StatusMessage>
        </Box>
      )}

      {isRawModeSupported && (
        <RotateKeyHandler
          maps={maps}
          zone={zone}
          mapCursor={mapCursor}
          opCursor={opCursor}
          optCursor={optCursor}
          optionRows={optionRows}
          canStart={canStart}
          batchAll={options.batchAll}
          setZone={setZone}
          setMapCursor={setMapCursor}
          setOpCursor={setOpCursor}
          setOptCursor={setOptCursor}
          setSource={setSource}
          toggleOp={toggleOp}
          onStart={onStart}
          onBack={onBack}
          onRescan={onRescan}
          onQuit={onQuit}
        />
      )}
    </Box>
  );
}

function SourceList({
  maps,
  cursor,
  active,
  disabled,
  selectedPath,
}: {
  maps: ScannedMap[];
  cursor: number;
  active: boolean;
  disabled: boolean;
  selectedPath: string | null;
}) {
  const theme = useTheme();
  const visible = useMemo(() => {
    const winSize = 8;
    if (maps.length <= winSize) return { items: maps, offset: 0 };
    const half = Math.floor(winSize / 2);
    const start = Math.max(0, Math.min(maps.length - winSize, cursor - half));
    return { items: maps.slice(start, start + winSize), offset: start };
  }, [maps, cursor]);

  return (
    <Box flexDirection="column">
      {visible.items.map((m, i) => {
        const idx = visible.offset + i;
        const isCursor = active && idx === cursor;
        const isSelected = !disabled && (m.path ?? m.name) === selectedPath;
        const icon = isSelected ? "◉" : "○";
        const playerColor = m.players === 3 ? theme.colors.warning : theme.colors.info;
        let nameColor: string | undefined;
        if (disabled) nameColor = theme.colors.mutedForeground;
        else if (isCursor) nameColor = theme.colors.primary;
        else if (!isSelected) nameColor = theme.colors.mutedForeground;
        return (
          <Box key={m.path ?? m.name} gap={1}>
            <Text color={isCursor ? theme.colors.primary : undefined}>
              {isCursor ? "›" : " "}
            </Text>
            <Text color={isSelected ? theme.colors.success : theme.colors.mutedForeground}>
              {icon}
            </Text>
            <Text color={playerColor} bold>
              {m.players}p
            </Text>
            <Text color={nameColor} bold={isCursor}>
              {m.name}
            </Text>
          </Box>
        );
      })}
      {maps.length > visible.items.length && (
        <Text dimColor>
          {"  "}… {maps.length - visible.items.length} more (j/k)
        </Text>
      )}
    </Box>
  );
}

function OpsList({
  cursor,
  active,
  selected,
}: {
  cursor: number;
  active: boolean;
  selected: RotateOp[];
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {ROTATE_OPS.map((row, idx) => {
        const isCursor = active && idx === cursor;
        const isSelected = selected.includes(row.key);
        const icon = isSelected ? "◉" : "○";
        return (
          <Box key={row.key} flexDirection="column">
            <Box gap={1}>
              <Text color={isCursor ? theme.colors.primary : undefined}>
                {isCursor ? "›" : " "}
              </Text>
              <Text color={isSelected ? theme.colors.success : theme.colors.mutedForeground}>
                {icon}
              </Text>
              <Text color={isCursor ? theme.colors.primary : undefined} bold={isCursor}>
                {row.label}
              </Text>
            </Box>
            {isCursor && (
              <Box marginLeft={4}>
                <Text dimColor>{row.hint}</Text>
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}

function RotateOptionList({
  rows,
  cursor,
  active,
}: {
  rows: { label: string; value: string; on: boolean; hint: string }[];
  cursor: number;
  active: boolean;
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {rows.map((row, idx) => {
        const isCursor = active && idx === cursor;
        const valueColor = row.on ? theme.colors.success : theme.colors.mutedForeground;
        return (
          <Box key={row.label} flexDirection="column">
            <Box gap={1}>
              <Text color={isCursor ? theme.colors.primary : undefined}>
                {isCursor ? "›" : " "}
              </Text>
              <Text color={isCursor ? theme.colors.primary : undefined} bold={isCursor}>
                {row.label.padEnd(34)}
              </Text>
              <Text color={valueColor} bold>
                {row.value}
              </Text>
            </Box>
            {isCursor && (
              <Box marginLeft={4}>
                <Text dimColor>{row.hint}</Text>
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}

function Footer({
  canStart,
  zone,
}: {
  canStart: boolean;
  zone: FocusZone;
}) {
  const theme = useTheme();
  const keys: { key: string; label: string }[] = [];
  if (zone === "maps") {
    keys.push({ key: "↑↓ / j k", label: "navigate" }, { key: "space", label: "select source" });
  } else if (zone === "ops") {
    keys.push({ key: "↑↓", label: "navigate" }, { key: "space", label: "toggle op" });
  } else {
    keys.push({ key: "↑↓", label: "navigate" }, { key: "space", label: "toggle" });
  }
  keys.push(
    { key: "tab", label: "switch panel" },
    { key: "r", label: "rescan" },
    { key: "b", label: "back" },
    { key: "enter", label: canStart ? "start" : "(incomplete)" },
    { key: "q", label: "quit" }
  );
  return (
    <Box marginTop={1} flexDirection="row" gap={2}>
      {keys.map((k) => (
        <Box key={k.key} gap={1}>
          <Text color={theme.colors.primary} bold>
            {k.key}
          </Text>
          <Text dimColor>{k.label}</Text>
        </Box>
      ))}
    </Box>
  );
}

interface RotateKeyHandlerProps {
  maps: ScannedMap[];
  zone: FocusZone;
  mapCursor: number;
  opCursor: number;
  optCursor: number;
  optionRows: { label: string; value: string; on: boolean; toggle: () => void; hint: string }[];
  canStart: boolean;
  batchAll: boolean;
  setZone: (z: FocusZone | ((p: FocusZone) => FocusZone)) => void;
  setMapCursor: (i: number | ((p: number) => number)) => void;
  setOpCursor: (i: number | ((p: number) => number)) => void;
  setOptCursor: (i: number | ((p: number) => number)) => void;
  setSource: (m: ScannedMap | null) => void;
  toggleOp: (op: RotateOp) => void;
  onStart: () => void;
  onBack: () => void;
  onRescan: () => void;
  onQuit: () => void;
}

function RotateKeyHandler({
  maps,
  zone,
  mapCursor,
  opCursor,
  optCursor,
  optionRows,
  canStart,
  batchAll,
  setZone,
  setMapCursor,
  setOpCursor,
  setOptCursor,
  setSource,
  toggleOp,
  onStart,
  onBack,
  onRescan,
  onQuit,
}: RotateKeyHandlerProps) {
  useInput((input, key) => {
    if (key.return) {
      if (!canStart) return;
      onStart();
      return;
    }
    if (input === "q" || (key.ctrl && input === "c")) {
      onQuit();
      return;
    }
    if (input === "b") { onBack(); return; }
    if (input === "r") { onRescan(); return; }
    if (key.tab) {
      setZone((z) => (z === "maps" ? "ops" : z === "ops" ? "options" : "maps"));
      return;
    }

    if (zone === "maps") {
      if (batchAll) return;
      if (key.upArrow || input === "k") setMapCursor((i) => Math.max(0, i - 1));
      else if (key.downArrow || input === "j") setMapCursor((i) => Math.min(maps.length - 1, i + 1));
      else if (input === " ") {
        const m = maps[mapCursor];
        if (m) setSource(m);
      } else if (key.pageUp || input === "g") setMapCursor(0);
      else if (key.pageDown || input === "G") setMapCursor(Math.max(0, maps.length - 1));
    } else if (zone === "ops") {
      if (key.upArrow || input === "k") setOpCursor((i) => Math.max(0, i - 1));
      else if (key.downArrow || input === "j") setOpCursor((i) => Math.min(ROTATE_OPS.length - 1, i + 1));
      else if (input === " ") {
        const op = ROTATE_OPS[opCursor];
        if (op) toggleOp(op.key);
      }
    } else {
      if (key.upArrow || input === "k") setOptCursor((i) => Math.max(0, i - 1));
      else if (key.downArrow || input === "j") setOptCursor((i) => Math.min(optionRows.length - 1, i + 1));
      else if (input === " ") optionRows[optCursor]?.toggle();
    }
  });
  return null;
}
