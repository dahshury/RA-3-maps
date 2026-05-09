import { useMemo, useState } from "react";
import { Box, Text, useStdin } from "ink";

import { useInput } from "@/hooks/use-input";

import { Heading } from "@/components/ui/heading";
import { Badge } from "@/components/ui/badge";
import { Banner } from "@/components/ui/banner";
import { Divider } from "@/components/ui/divider";
import { KeyValue } from "@/components/ui/key-value";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";
import {
  MAP_RESTRICTIONS,
  type ConversionOptions,
  type MapRestriction,
} from "@/lib/events";

export interface ScannedMap {
  name: string;
  path?: string;
  players: number;
}

export interface SetupScreenProps {
  inputDir: string;
  outputDir: string;
  maps: ScannedMap[];
  skipped: { name: string; reason: string }[];
  options: ConversionOptions;
  onChange: (opts: ConversionOptions) => void;
  onStart: () => void;
  onQuit: () => void;
  onRescan: () => void;
}

type FocusZone = "maps" | "options";

export function SetupScreen({
  inputDir,
  outputDir,
  maps,
  skipped,
  options,
  onChange,
  onStart,
  onQuit,
  onRescan,
}: SetupScreenProps) {
  const theme = useTheme();
  const [zone, setZone] = useState<FocusZone>("maps");
  const [mapCursor, setMapCursor] = useState(0);
  const [optCursor, setOptCursor] = useState(0);

  const allMapNames = useMemo(() => maps.map((m) => m.name), [maps]);
  const selected = options.selectedMaps;
  const selectedSet = useMemo(() => new Set(selected), [selected]);

  const setSelected = (next: string[]) =>
    onChange({ ...options, selectedMaps: next });

  const toggleMap = (idx: number) => {
    const m = maps[idx];
    if (!m) return;
    const next = selectedSet.has(m.name)
      ? selected.filter((n) => n !== m.name)
      : [...selected, m.name];
    setSelected(next);
  };

  const selectAll = () => setSelected(allMapNames);
  const selectNone = () => setSelected([]);
  const invertSelection = () =>
    setSelected(allMapNames.filter((n) => !selectedSet.has(n)));

  type OptionRow = {
    label: string;
    value: string;
    on: boolean;
    toggle?: () => void;
    bump?: (delta: number) => void;
    hint: string;
  };

  const archonRow: OptionRow = {
    label: "Apply Archon transform",
    value: options.applyArchon ? "ON" : "OFF",
    on: options.applyArchon,
    toggle: () =>
      onChange({ ...options, applyArchon: !options.applyArchon }),
    hint: "Pair builder with controller spawn (1-3 player sources only).",
  };

  const compressRow: OptionRow = {
    label: "Compress output",
    value: options.compress ? "ON" : "OFF",
    on: options.compress,
    toggle: () => onChange({ ...options, compress: !options.compress }),
    hint: "Standard for in-game maps. Turn OFF for WorldBuilder debugging.",
  };

  const sidecarRow: OptionRow = {
    label: "Write XML sidecars",
    value: options.writeSidecars ? "ON" : "OFF",
    on: options.writeSidecars,
    toggle: () =>
      onChange({ ...options, writeSidecars: !options.writeSidecars }),
    hint: "map.xml + overrides.xml for WorldBuilder compatibility.",
  };

  const wbRow: OptionRow = {
    label: "WB-normalize terrain",
    value: options.wbNormalizeTerrain ? "ON" : "OFF",
    on: options.wbNormalizeTerrain,
    toggle: () =>
      onChange({
        ...options,
        wbNormalizeTerrain: !options.wbNormalizeTerrain,
      }),
    hint: "Reorder texture slots WB-style. Can change blending vs. official maps. (Archon only)",
  };

  const offsetRow: OptionRow = {
    label: "Controller spawn offset",
    value: `${options.offset} units`,
    on: false,
    bump: (delta) =>
      onChange({
        ...options,
        offset: Math.max(0, Math.min(5000, options.offset + delta)),
      }),
    hint: "Distance from builder spawn to paired controller. Default 800. (Archon only)",
  };

  const restrictionSelected = new Set<MapRestriction>(options.restrictions);
  const restrictionRows: OptionRow[] = MAP_RESTRICTIONS.map((v) => {
    const enabled = restrictionSelected.has(v.key);
    return {
      label: `Restriction · ${v.label}`,
      value: enabled ? "ON" : "OFF",
      on: enabled,
      toggle: () => {
        const next = enabled
          ? options.restrictions.filter((k) => k !== v.key)
          : [...options.restrictions, v.key];
        onChange({ ...options, restrictions: next });
      },
      hint: v.hint,
    };
  });

  const optionRows: OptionRow[] = [
    archonRow,
    ...restrictionRows,
    compressRow,
    sidecarRow,
    ...(options.applyArchon ? [wbRow, offsetRow] : []),
  ];

  const { isRawModeSupported } = useStdin();

  const counts = useMemo(() => {
    let p2 = 0;
    let p3 = 0;
    for (const m of maps) {
      if (m.players === 3) p3++;
      else p2++;
    }
    return { p2, p3 };
  }, [maps]);

  const selectedCount = selected.length;
  const restrictionCount = options.restrictions.length;
  // Each source produces 1 (archon-only or single restriction) or N outputs
  // when multiple restrictions are picked. Combined: archon × restrictions.
  const runsPerSource = Math.max(1, restrictionCount);
  const canStart =
    selectedCount > 0 && (options.applyArchon || restrictionCount > 0);
  const totalOutputs = canStart ? selectedCount * runsPerSource : 0;

  const pipelineSummary = (() => {
    const parts: string[] = [];
    if (options.applyArchon) parts.push("Archon");
    if (restrictionCount > 0)
      parts.push(`${restrictionCount} restriction${restrictionCount === 1 ? "" : "s"}`);
    return parts.length ? parts.join(" + ") : "(nothing selected)";
  })();

  return (
    <Box flexDirection="column">
      <Box marginTop={1} flexDirection="column">
        <KeyValue
          items={[
            { key: "Pipeline", value: pipelineSummary },
            { key: "Input", value: inputDir },
            { key: "Output", value: outputDir },
          ]}
        />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{`Found ${maps.length}`}</Badge>
        <Badge variant="success">{`2p: ${counts.p2}`}</Badge>
        <Badge variant="success">{`3p: ${counts.p3}`}</Badge>
        {restrictionCount > 0 && (
          <Badge variant="success">{`Restrictions: ${restrictionCount}/5`}</Badge>
        )}
        {skipped.length > 0 && (
          <Badge variant="warning">{`Skipped: ${skipped.length}`}</Badge>
        )}
        <Badge
          variant={selectedCount > 0 ? "secondary" : "warning"}
        >{`Selected: ${selectedCount}`}</Badge>
        <Badge variant={totalOutputs > 0 ? "info" : "warning"}>{`Outputs: ${totalOutputs}`}</Badge>
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
              {zone === "maps" ? "▶" : " "} Maps to convert
            </Text>
            <Text dimColor>
              ({selectedCount}/{maps.length} selected)
            </Text>
          </Box>
          <MapList
            maps={maps}
            selected={selectedSet}
            cursor={mapCursor}
            active={zone === "maps"}
          />
        </Box>
      )}

      <Box marginTop={1} flexDirection="column">
        <Box flexDirection="row" gap={1}>
          <Text bold color={zone === "options" ? theme.colors.primary : undefined}>
            {zone === "options" ? "▶" : " "} Options
          </Text>
        </Box>
        <OptionList rows={optionRows} cursor={optCursor} active={zone === "options"} />
      </Box>

      {skipped.length > 0 && (
        <Box marginTop={1} flexDirection="column">
          <Text dimColor>Skipped (not convertible):</Text>
          {skipped.slice(0, 4).map((s) => (
            <Text key={s.name} dimColor>
              {"  "}• {s.name} — {s.reason}
            </Text>
          ))}
          {skipped.length > 4 && (
            <Text dimColor>
              {"  "}… and {skipped.length - 4} more
            </Text>
          )}
        </Box>
      )}

      <Box marginTop={1}>
        <Divider />
      </Box>
      <Footer
        canStart={canStart}
        zone={zone}
        applyArchon={options.applyArchon}
        restrictionCount={restrictionCount}
        selectedCount={selectedCount}
      />

      {isRawModeSupported && (
        <SetupKeyHandler
          maps={maps}
          options={options}
          zone={zone}
          mapCursor={mapCursor}
          optCursor={optCursor}
          optionRows={optionRows}
          selected={selected}
          selectedSet={selectedSet}
          allMapNames={allMapNames}
          canStart={canStart}
          setZone={setZone}
          setMapCursor={setMapCursor}
          setOptCursor={setOptCursor}
          setSelected={setSelected}
          onStart={onStart}
          onQuit={onQuit}
          onRescan={onRescan}
        />
      )}
    </Box>
  );
}

interface SetupKeyHandlerProps {
  maps: ScannedMap[];
  options: ConversionOptions;
  zone: FocusZone;
  mapCursor: number;
  optCursor: number;
  optionRows: {
    label: string;
    value: string;
    on: boolean;
    toggle?: () => void;
    bump?: (delta: number) => void;
    hint: string;
  }[];
  selected: string[];
  selectedSet: Set<string>;
  allMapNames: string[];
  canStart: boolean;
  setZone: (z: FocusZone | ((prev: FocusZone) => FocusZone)) => void;
  setMapCursor: (i: number | ((prev: number) => number)) => void;
  setOptCursor: (i: number | ((prev: number) => number)) => void;
  setSelected: (next: string[]) => void;
  onStart: () => void;
  onQuit: () => void;
  onRescan: () => void;
}

function SetupKeyHandler({
  maps,
  zone,
  mapCursor,
  optCursor,
  optionRows,
  selected,
  selectedSet,
  allMapNames,
  canStart,
  setZone,
  setMapCursor,
  setOptCursor,
  setSelected,
  onStart,
  onQuit,
  onRescan,
}: SetupKeyHandlerProps) {
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
    if (input === "r") {
      onRescan();
      return;
    }
    if (key.tab) {
      setZone((z) => (z === "maps" ? "options" : "maps"));
      return;
    }

    if (zone === "maps") {
      if (key.upArrow || input === "k") {
        setMapCursor((i) => Math.max(0, i - 1));
      } else if (key.downArrow || input === "j") {
        setMapCursor((i) => Math.min(maps.length - 1, i + 1));
      } else if (input === " ") {
        const m = maps[mapCursor];
        if (!m) return;
        const next = selectedSet.has(m.name)
          ? selected.filter((n) => n !== m.name)
          : [...selected, m.name];
        setSelected(next);
      } else if (input === "a") {
        setSelected(allMapNames);
      } else if (input === "n") {
        setSelected([]);
      } else if (input === "i") {
        setSelected(allMapNames.filter((n) => !selectedSet.has(n)));
      } else if (key.pageUp || input === "g") {
        setMapCursor(0);
      } else if (key.pageDown || input === "G") {
        setMapCursor(Math.max(0, maps.length - 1));
      }
    } else {
      if (key.upArrow || input === "k") {
        setOptCursor((i) => Math.max(0, i - 1));
      } else if (key.downArrow || input === "j") {
        setOptCursor((i) => Math.min(optionRows.length - 1, i + 1));
      } else {
        const row = optionRows[optCursor];
        if (!row) return;
        if (input === " " && row.toggle) {
          row.toggle();
        } else if (key.leftArrow && row.bump) {
          row.bump(-100);
        } else if (key.rightArrow && row.bump) {
          row.bump(100);
        } else if (input === "-" && row.bump) {
          row.bump(-50);
        } else if (input === "+" && row.bump) {
          row.bump(50);
        }
      }
    }
  });
  return null;
}

function MapList({
  maps,
  selected,
  cursor,
  active,
}: {
  maps: ScannedMap[];
  selected: Set<string>;
  cursor: number;
  active: boolean;
}) {
  const theme = useTheme();
  const visible = useMemo(() => {
    // Window of 10 around cursor.
    const winSize = 10;
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
        const isSelected = selected.has(m.name);
        const icon = isSelected ? "◉" : "○";
        const playerColor =
          m.players === 3 ? theme.colors.warning : theme.colors.info;

        let nameColor: string | undefined;
        if (isCursor) nameColor = theme.colors.primary;
        else if (!isSelected) nameColor = theme.colors.mutedForeground;

        return (
          <Box key={m.path ?? m.name} gap={1}>
            <Text color={isCursor ? theme.colors.primary : undefined}>
              {isCursor ? "›" : " "}
            </Text>
            <Text
              color={
                isSelected ? theme.colors.success : theme.colors.mutedForeground
              }
            >
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
          {"  "}… {maps.length - visible.items.length} more (use j/k or
          PgUp/PgDn)
        </Text>
      )}
    </Box>
  );
}

function OptionList({
  rows,
  cursor,
  active,
}: {
  rows: {
    label: string;
    value: string;
    on: boolean;
    hint: string;
    bump?: (delta: number) => void;
  }[];
  cursor: number;
  active: boolean;
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {rows.map((row, idx) => {
        const isCursor = active && idx === cursor;
        const isToggle = !row.bump;
        const valueColor = isToggle
          ? row.on
            ? theme.colors.success
            : theme.colors.mutedForeground
          : theme.colors.info;
        return (
          <Box key={row.label} flexDirection="column">
            <Box gap={1}>
              <Text color={isCursor ? theme.colors.primary : undefined}>
                {isCursor ? "›" : " "}
              </Text>
              <Text
                color={isCursor ? theme.colors.primary : undefined}
                bold={isCursor}
              >
                {row.label.padEnd(26)}
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
  applyArchon,
  restrictionCount,
  selectedCount,
}: {
  canStart: boolean;
  zone: FocusZone;
  applyArchon: boolean;
  restrictionCount: number;
  selectedCount: number;
}) {
  const theme = useTheme();
  const keys: { key: string; label: string }[] = [];

  if (zone === "maps") {
    keys.push(
      { key: "↑↓ / j k", label: "navigate" },
      { key: "space", label: "toggle" },
      { key: "a / n / i", label: "all / none / invert" }
    );
  } else {
    keys.push(
      { key: "↑↓", label: "navigate" },
      { key: "space", label: "toggle" },
      { key: "← →", label: "adjust value" }
    );
  }
  keys.push(
    { key: "tab", label: "switch panel" },
    { key: "r", label: "rescan" },
    { key: "enter", label: canStart ? "start" : "(setup needed)" },
    { key: "q", label: "quit" }
  );

  let warning: string | null = null;
  if (selectedCount === 0) {
    warning = "Select at least one map to enable Start (press a to select all).";
  } else if (!applyArchon && restrictionCount === 0) {
    warning =
      "Enable Apply Archon and/or one or more match restrictions in the Options panel.";
  }

  return (
    <Box marginTop={1} flexDirection="column">
      <Box flexDirection="row" gap={2}>
        {keys.map((k) => (
          <Box key={k.key} gap={1}>
            <Text color={theme.colors.primary} bold>
              {k.key}
            </Text>
            <Text dimColor>{k.label}</Text>
          </Box>
        ))}
      </Box>
      {warning && (
        <Box marginTop={1}>
          <StatusMessage variant="warning">{warning}</StatusMessage>
        </Box>
      )}
    </Box>
  );
}
