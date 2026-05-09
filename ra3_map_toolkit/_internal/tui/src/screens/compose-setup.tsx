import { useMemo, useState } from "react";
import { Box, Text, useStdin } from "ink";

import { useInput } from "@/hooks/use-input";

import { Badge } from "@/components/ui/badge";
import { Banner } from "@/components/ui/banner";
import { Divider } from "@/components/ui/divider";
import { KeyValue } from "@/components/ui/key-value";
import { StatusMessage } from "@/components/ui/status-message";
import { useTheme } from "@/components/ui/theme-provider";

import {
  ALIGN_X_LABEL,
  ALIGN_X_VALUES,
  ALIGN_Y_LABEL,
  ALIGN_Y_VALUES,
  COMPOSE_PRESETS,
  MAX_PLAYERS,
  composeFitsCap,
  composeOpLabel,
  composePresetSlotCount,
  composeProjectedPlayers,
  type AlignX,
  type AlignY,
  type ComposePreset,
} from "@/lib/events";
import type { ScannedMap } from "./setup";

export interface ComposeSetupOptions {
  preset: ComposePreset;
  slotMaps: (ScannedMap | null)[]; // length up to 3; null = unfilled
  slotAlignX: AlignX[];            // per-slot, length 3
  slotAlignY: AlignY[];            // per-slot, length 3
  nx: number;                      // duplicate preset only
  ny: number;                      // duplicate preset only
  padX: number;                    // tile-unit X gap
  padY: number;                    // tile-unit Y gap
  compress: boolean;
}

export const defaultComposeSetupOptions = (): ComposeSetupOptions => ({
  preset: "duplicate",
  slotMaps: [null, null, null],
  slotAlignX: ["center", "center", "center"],
  slotAlignY: ["center", "center", "center"],
  nx: 2,
  ny: 1,
  padX: 0,
  padY: 0,
  compress: true,
});

export interface ComposeSetupScreenProps {
  inputDir: string;
  outputDir: string;
  maps: ScannedMap[];
  skipped: { name: string; reason: string }[];
  options: ComposeSetupOptions;
  onChange: (opts: ComposeSetupOptions) => void;
  onStart: () => void;
  onBack: () => void;
  onRescan: () => void;
  onQuit: () => void;
}

type FocusZone = "preset" | "slots" | "align" | "params" | "options";

interface ParamRow {
  key: "nx" | "ny" | "padX" | "padY";
  label: string;
  hint: string;
  min: number;
  max: number;
  // only show this row when the preset has it relevant (e.g. nx/ny only for duplicate).
  appliesTo: (preset: ComposePreset) => boolean;
}

const PARAM_ROWS: ParamRow[] = [
  {
    key: "nx", label: "Nx (X-axis tiles)", min: 1, max: 6,
    hint: "Number of horizontal copies. Total players = source players × Nx × Ny (cap 6).",
    appliesTo: (p) => p === "duplicate",
  },
  {
    key: "ny", label: "Ny (Y-axis tiles)", min: 1, max: 6,
    hint: "Number of vertical copies. Total players = source players × Nx × Ny (cap 6).",
    appliesTo: (p) => p === "duplicate",
  },
  {
    key: "padX", label: "pad-x (X gap, tiles)", min: 0, max: 9999,
    hint: "Tile-unit water gap inserted between adjacent X-axis cells. 0 = none.",
    appliesTo: (p) => p !== "col",
  },
  {
    key: "padY", label: "pad-y (Y gap, tiles)", min: 0, max: 9999,
    hint: "Tile-unit water gap inserted between adjacent Y-axis cells. 0 = none.",
    appliesTo: (p) => p !== "row",
  },
];

// Slot ordering -- mirrors compose_engine._build_spec.
// Each preset's slot label tells the user where each map ends up in the output.
const SLOT_LABELS: Record<ComposePreset, string[]> = {
  duplicate: ["Source"],
  row: ["Left", "Middle", "Right"],
  col: ["Top", "Middle", "Bottom"],
  triangle_top: ["Top-Left", "Top-Right", "Bottom (spans 2 cols)"],
  triangle_bottom: ["Top (spans 2 cols)", "Bottom-Left", "Bottom-Right"],
  triangle_left: ["Top-Left", "Bottom-Left", "Right (spans 2 rows)"],
  triangle_right: ["Left (spans 2 rows)", "Top-Right", "Bottom-Right"],
};

// Big visual diagram per preset -- rendered at the top of the screen so the
// user sees what they're building.
const PRESET_DIAGRAM: Record<ComposePreset, string[]> = {
  duplicate: [
    "┌─────┬─────┐",
    "│  A  │  A  │  ← Nx × Ny grid of the same source",
    "├─────┼─────┤",
    "│  A  │  A  │",
    "└─────┴─────┘",
  ],
  row: [
    "┌─────┬─────┬─────┐",
    "│  A  │  B  │  C  │  ← left to right",
    "└─────┴─────┴─────┘",
  ],
  col: [
    "┌─────┐",
    "│  A  │",
    "├─────┤",
    "│  B  │  ← top to bottom",
    "├─────┤",
    "│  C  │",
    "└─────┘",
  ],
  triangle_top: [
    "┌─────┬─────┐",
    "│  A  │  B  │  ← 2 maps top",
    "├─────┴─────┤",
    "│     C     │  ← 1 map below, spans both cols",
    "└───────────┘",
  ],
  triangle_bottom: [
    "┌───────────┐",
    "│     A     │  ← 1 map top, spans both cols",
    "├─────┬─────┤",
    "│  B  │  C  │  ← 2 maps bottom",
    "└─────┴─────┘",
  ],
  triangle_left: [
    "┌─────┬─────┐",
    "│  A  │     │",
    "├─────┤  C  │  ← 1 map right, spans both rows",
    "│  B  │     │",
    "└─────┴─────┘",
  ],
  triangle_right: [
    "┌─────┬─────┐",
    "│     │  B  │",
    "│  A  ├─────┤  ← 1 map left, spans both rows",
    "│     │  C  │",
    "└─────┴─────┘",
  ],
};

export function ComposeSetupScreen({
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
}: ComposeSetupScreenProps) {
  const theme = useTheme();
  const [zone, setZone] = useState<FocusZone>("preset");
  const [presetCursor, setPresetCursor] = useState(
    Math.max(0, COMPOSE_PRESETS.findIndex((p) => p.key === options.preset))
  );
  const [slotCursor, setSlotCursor] = useState(0);  // which slot (0..2)
  const [mapCursor, setMapCursor] = useState(0);    // index into eligible[]
  const [alignCursor, setAlignCursor] = useState(0); // which slot's align row
  const [paramCursor, setParamCursor] = useState(0);
  const [optCursor, setOptCursor] = useState(0);

  const { isRawModeSupported } = useStdin();

  // Eligibility: every cell needs >= 2 players.
  const eligible = useMemo(
    () => maps.filter((m) => m.players >= 2 && m.players <= MAX_PLAYERS),
    [maps]
  );

  // Maps grouped by player count for the summary line.
  const counts = useMemo(() => {
    let p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, other = 0;
    for (const m of maps) {
      if (m.players === 2) p2++;
      else if (m.players === 3) p3++;
      else if (m.players === 4) p4++;
      else if (m.players === 5) p5++;
      else if (m.players === 6) p6++;
      else other++;
    }
    return { p2, p3, p4, p5, p6, other };
  }, [maps]);

  const presetCfg = COMPOSE_PRESETS[presetCursor]!;
  const slotCount = composePresetSlotCount(presetCfg.key, presetCfg.slots);
  const paramRowsActive = PARAM_ROWS.filter((r) => r.appliesTo(presetCfg.key));

  // ---- Helpers ----
  const setPreset = (preset: ComposePreset) => {
    onChange({ ...options, preset });
  };
  const setSlotMap = (slot: number, m: ScannedMap | null) => {
    const next = [...options.slotMaps];
    while (next.length <= slot) next.push(null);
    next[slot] = m;
    onChange({ ...options, slotMaps: next });
  };
  const setSlotAlignX = (slot: number, value: AlignX) => {
    const next = [...options.slotAlignX];
    while (next.length <= slot) next.push("center");
    next[slot] = value;
    onChange({ ...options, slotAlignX: next });
  };
  const setSlotAlignY = (slot: number, value: AlignY) => {
    const next = [...options.slotAlignY];
    while (next.length <= slot) next.push("center");
    next[slot] = value;
    onChange({ ...options, slotAlignY: next });
  };
  const cycleAlignX = (slot: number, dir: 1 | -1) => {
    const cur = options.slotAlignX[slot] ?? "center";
    const idx = ALIGN_X_VALUES.indexOf(cur);
    const next = ALIGN_X_VALUES[(idx + dir + ALIGN_X_VALUES.length) % ALIGN_X_VALUES.length]!;
    setSlotAlignX(slot, next);
  };
  const cycleAlignY = (slot: number, dir: 1 | -1) => {
    const cur = options.slotAlignY[slot] ?? "center";
    const idx = ALIGN_Y_VALUES.indexOf(cur);
    const next = ALIGN_Y_VALUES[(idx + dir + ALIGN_Y_VALUES.length) % ALIGN_Y_VALUES.length]!;
    setSlotAlignY(slot, next);
  };
  const setParam = (key: ParamRow["key"], v: number) => {
    onChange({ ...options, [key]: v });
  };

  const compressRow = {
    label: "Compress output",
    value: options.compress ? "ON" : "OFF",
    on: options.compress,
    toggle: () => onChange({ ...options, compress: !options.compress }),
    hint: "Standard for in-game maps. Turn OFF for WorldBuilder debugging.",
  };
  const optionRows = [compressRow];

  // Effective slot maps (truncate / extend to slot count)
  const effectiveSlots: (ScannedMap | null)[] =
    Array.from({ length: slotCount }, (_, i) => options.slotMaps[i] ?? null);

  const slotPlayers = effectiveSlots.map((m) => m?.players ?? 0);
  const projectedPlayers = composeProjectedPlayers(
    presetCfg.key, slotPlayers, options.nx, options.ny
  );
  const fitsCap = composeFitsCap(presetCfg.key, slotPlayers, options.nx, options.ny);

  // row/col accept 2 OR 3 sources; everything else needs every slot filled.
  // Triangle presets need all 3 because span layout depends on each slot.
  const minFilled =
    presetCfg.key === "row" || presetCfg.key === "col" ? 2 : slotCount;
  const filledCount = effectiveSlots.filter((m) => m !== null).length;
  const enoughFilled = filledCount >= minFilled;
  const opIsNoop = presetCfg.key === "duplicate" && options.nx === 1 && options.ny === 1;

  const canStart = !opIsNoop && enoughFilled && fitsCap && projectedPlayers >= 2;

  const opLabel = composeOpLabel({
    preset: presetCfg.key,
    maps: effectiveSlots.map((m) => m?.path ?? m?.name ?? ""),
    nx: options.nx,
    ny: options.ny,
    padX: options.padX,
    padY: options.padY,
    alignX: options.slotAlignX,
    alignY: options.slotAlignY,
  });

  // Combined output dimensions estimate (best-effort, ignoring borders/pad).
  // We don't know the exact playable areas; ScannedMap doesn't carry dims.
  // So we just preview the player-count math.

  // ---- Render ----
  return (
    <Box flexDirection="column">
      <Box marginTop={1}>
        <KeyValue
          items={[
            { key: "Pipeline", value: "Compose / Stitch" },
            { key: "Input", value: inputDir },
            { key: "Output", value: outputDir },
            { key: "Preset", value: `${presetCfg.label}  ·  ${slotCount} slot(s)` },
            {
              key: "Op",
              value: opIsNoop
                ? "(no-op: set Nx or Ny ≥ 2)"
                : `${opLabel}` +
                  (enoughFilled
                    ? `  →  ${projectedPlayers}p ${fitsCap ? "" : "[exceeds cap]"}`
                    : "  →  (slots incomplete)"),
            },
          ]}
        />
      </Box>

      <Box marginTop={1}>
        <PresetDiagram preset={presetCfg.key} slots={effectiveSlots} />
      </Box>

      <Box marginTop={1} flexDirection="row" gap={1}>
        <Badge variant="info">{`Found ${maps.length}`}</Badge>
        {counts.p2 > 0 && <Badge variant="success">{`2p: ${counts.p2}`}</Badge>}
        {counts.p3 > 0 && <Badge variant="success">{`3p: ${counts.p3}`}</Badge>}
        {counts.p4 > 0 && <Badge variant="success">{`4p: ${counts.p4}`}</Badge>}
        {counts.p5 > 0 && <Badge variant="success">{`5p: ${counts.p5}`}</Badge>}
        {counts.p6 > 0 && <Badge variant="success">{`6p: ${counts.p6}`}</Badge>}
        {counts.other > 0 && (
          <Badge variant="warning">{`>6p (skipped): ${counts.other}`}</Badge>
        )}
        {skipped.length > 0 && (
          <Badge variant="warning">{`Unparsable: ${skipped.length}`}</Badge>
        )}
        <Badge variant={canStart ? "info" : "warning"}>
          {`${projectedPlayers}p / ${MAX_PLAYERS}p`}
        </Badge>
      </Box>

      {maps.length === 0 ? (
        <Box marginTop={1}>
          <Banner variant="warning" title="No maps found">
            Place .map files (or folders containing them) in {inputDir} and press
            r to rescan.
          </Banner>
        </Box>
      ) : eligible.length === 0 ? (
        <Box marginTop={1}>
          <Banner variant="warning" title="No eligible sources">
            Composition needs sources with ≥ 2 players. None found.
          </Banner>
        </Box>
      ) : (
        <Box marginTop={1} flexDirection="column">
          {/* Preset picker */}
          <Box flexDirection="row" gap={1}>
            <Text bold color={zone === "preset" ? theme.colors.primary : undefined}>
              {zone === "preset" ? "▶" : " "} Layout preset
            </Text>
            <Text dimColor>(↑↓ to choose; preset diagram updates live)</Text>
          </Box>
          <PresetList
            cursor={presetCursor}
            active={zone === "preset"}
            selected={presetCfg.key}
          />

          {/* Slot pickers */}
          <Box marginTop={1} flexDirection="row" gap={1}>
            <Text bold color={zone === "slots" ? theme.colors.primary : undefined}>
              {zone === "slots" ? "▶" : " "} Pick maps
            </Text>
            <Text dimColor>
              {`(slot ${slotCursor + 1}/${slotCount}: ${SLOT_LABELS[presetCfg.key][slotCursor] ?? ""}; ${
                presetCfg.key === "row" || presetCfg.key === "col"
                  ? "fill 2 or 3 slots"
                  : "fill all slots"
              }; space to pick, [/] to switch)`}
            </Text>
          </Box>
          <SlotSummary
            slotLabels={SLOT_LABELS[presetCfg.key]}
            slotMaps={effectiveSlots}
            cursor={slotCursor}
            active={zone === "slots"}
          />
          <Box marginTop={0}>
            <SourceList
              maps={eligible}
              cursor={mapCursor}
              active={zone === "slots"}
              selectedSlots={effectiveSlots}
              currentSlot={slotCursor}
            />
          </Box>

          {/* Per-slot alignment */}
          <Box marginTop={1} flexDirection="column">
            <Box flexDirection="row" gap={1}>
              <Text bold color={zone === "align" ? theme.colors.primary : undefined}>
                {zone === "align" ? "▶" : " "} Alignment per slot
              </Text>
              <Text dimColor>
                (only matters when source is smaller than its allocated cell;
                ←/→ for X, ,/. for Y)
              </Text>
            </Box>
            <AlignList
              slotLabels={SLOT_LABELS[presetCfg.key]}
              slotMaps={effectiveSlots}
              slotAlignX={options.slotAlignX}
              slotAlignY={options.slotAlignY}
              cursor={alignCursor}
              active={zone === "align"}
            />
          </Box>

          {/* Param tweaks */}
          {paramRowsActive.length > 0 && (
            <Box marginTop={1} flexDirection="column">
              <Box flexDirection="row" gap={1}>
                <Text bold color={zone === "params" ? theme.colors.primary : undefined}>
                  {zone === "params" ? "▶" : " "} Parameters
                </Text>
                <Text dimColor>(←/→ to nudge; +shift for ×10)</Text>
              </Box>
              <ParamList
                rows={paramRowsActive}
                op={options}
                cursor={paramCursor}
                active={zone === "params"}
                slotPlayers={slotPlayers}
                preset={presetCfg.key}
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
        </Box>
      )}

      <Box marginTop={1}>
        <Divider />
      </Box>

      <Footer canStart={canStart} zone={zone} />

      {!canStart && (
        <Box marginTop={1}>
          <StatusMessage variant="warning">
            {opIsNoop
              ? "Set Nx or Ny ≥ 2 (1×1 produces no output)."
              : !enoughFilled
              ? `Pick at least ${minFilled} map(s) for ${presetCfg.label}. ${
                  presetCfg.key === "row" || presetCfg.key === "col"
                    ? "(2 or 3 maps work for row/col.)"
                    : ""
                }`
              : !fitsCap
              ? `${slotPlayers.filter((n) => n > 0).join("p + ")}p${presetCfg.key === "duplicate" ? ` × ${options.nx}×${options.ny}` : ""} = ${projectedPlayers} > ${MAX_PLAYERS} (cap). Reduce slots / Nx / Ny.`
              : "Incomplete."}
          </StatusMessage>
        </Box>
      )}

      {isRawModeSupported && (
        <ComposeKeyHandler
          eligible={eligible}
          slotCount={slotCount}
          paramRows={paramRowsActive}
          zone={zone}
          presetCursor={presetCursor}
          slotCursor={slotCursor}
          mapCursor={mapCursor}
          alignCursor={alignCursor}
          paramCursor={paramCursor}
          optCursor={optCursor}
          optionRows={optionRows}
          canStart={canStart}
          options={options}
          setZone={setZone}
          setPresetCursor={setPresetCursor}
          setSlotCursor={setSlotCursor}
          setMapCursor={setMapCursor}
          setAlignCursor={setAlignCursor}
          setParamCursor={setParamCursor}
          setOptCursor={setOptCursor}
          setPreset={setPreset}
          setSlotMap={setSlotMap}
          setSlotAlignX={setSlotAlignX}
          setSlotAlignY={setSlotAlignY}
          cycleAlignX={cycleAlignX}
          cycleAlignY={cycleAlignY}
          setParam={setParam}
          onStart={onStart}
          onBack={onBack}
          onRescan={onRescan}
          onQuit={onQuit}
        />
      )}
    </Box>
  );
}

function PresetDiagram({
  preset,
  slots,
}: {
  preset: ComposePreset;
  slots: (ScannedMap | null)[];
}) {
  const theme = useTheme();
  const lines = PRESET_DIAGRAM[preset];

  // Build a label key A/B/C → slot's name (truncated) so the diagram shows
  // who each cell holds when filled.
  const labels = ["A", "B", "C"]
    .map((k, i) => ({
      key: k,
      name: slots[i]?.name ?? "(empty)",
      players: slots[i]?.players ?? 0,
    }));

  return (
    <Box flexDirection="column">
      <Divider label={`Layout · ${preset}`} />
      <Box marginTop={1} flexDirection="row" gap={3}>
        <Box flexDirection="column">
          {lines.map((line, idx) => (
            <Text key={idx} color={theme.colors.accent}>{line}</Text>
          ))}
        </Box>
        <Box flexDirection="column">
          {labels.slice(0, slots.length).map((l, idx) => (
            <Box key={l.key} gap={1}>
              <Text color={theme.colors.primary} bold>{l.key}</Text>
              <Text dimColor>=</Text>
              <Text color={l.name === "(empty)" ? theme.colors.mutedForeground : theme.colors.success}>
                {l.name}
              </Text>
              {l.players > 0 && (
                <Text color={theme.colors.info}>({l.players}p)</Text>
              )}
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}

function PresetList({
  cursor,
  active,
  selected,
}: {
  cursor: number;
  active: boolean;
  selected: ComposePreset;
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {COMPOSE_PRESETS.map((p, idx) => {
        const isCursor = active && idx === cursor;
        const isSelected = p.key === selected;
        const icon = isSelected ? "◉" : "○";
        return (
          <Box key={p.key} flexDirection="column">
            <Box gap={1}>
              <Text color={isCursor ? theme.colors.primary : undefined}>
                {isCursor ? "›" : " "}
              </Text>
              <Text color={isSelected ? theme.colors.success : theme.colors.mutedForeground}>
                {icon}
              </Text>
              <Text bold={isCursor || isSelected}
                    color={isCursor ? theme.colors.primary : isSelected ? theme.colors.success : undefined}>
                {p.label.padEnd(32)}
              </Text>
              <Text dimColor>{p.ascii.join("  ")}</Text>
            </Box>
            {isCursor && (
              <Box marginLeft={4}>
                <Text dimColor>{p.hint}</Text>
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}

function SlotSummary({
  slotLabels,
  slotMaps,
  cursor,
  active,
}: {
  slotLabels: string[];
  slotMaps: (ScannedMap | null)[];
  cursor: number;
  active: boolean;
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {slotMaps.map((m, idx) => {
        const isCursor = active && idx === cursor;
        const label = slotLabels[idx] ?? `Slot ${idx + 1}`;
        const status = m
          ? `${m.name}  (${m.players}p)`
          : "(empty)";
        const statusColor = m
          ? theme.colors.success
          : isCursor
          ? theme.colors.warning
          : theme.colors.mutedForeground;
        return (
          <Box key={idx} gap={1}>
            <Text color={isCursor ? theme.colors.primary : undefined}>
              {isCursor ? "›" : " "}
            </Text>
            <Text color={theme.colors.primary} bold>
              {(["A", "B", "C"][idx] ?? "?")}
            </Text>
            <Text dimColor>·</Text>
            <Text color={isCursor ? theme.colors.primary : undefined}>
              {label.padEnd(24)}
            </Text>
            <Text color={statusColor} bold={isCursor}>
              {status}
            </Text>
          </Box>
        );
      })}
    </Box>
  );
}

function SourceList({
  maps,
  cursor,
  active,
  selectedSlots,
  currentSlot,
}: {
  maps: ScannedMap[];
  cursor: number;
  active: boolean;
  selectedSlots: (ScannedMap | null)[];
  currentSlot: number;
}) {
  const theme = useTheme();
  const visible = useMemo(() => {
    const winSize = 8;
    if (maps.length <= winSize) return { items: maps, offset: 0 };
    const half = Math.floor(winSize / 2);
    const start = Math.max(0, Math.min(maps.length - winSize, cursor - half));
    return { items: maps.slice(start, start + winSize), offset: start };
  }, [maps, cursor]);

  const slotIdx = (m: ScannedMap): number => {
    const id = m.path ?? m.name;
    return selectedSlots.findIndex((sm) => sm && (sm.path ?? sm.name) === id);
  };

  return (
    <Box flexDirection="column">
      {visible.items.map((m, i) => {
        const idx = visible.offset + i;
        const isCursor = active && idx === cursor;
        const usedInSlot = slotIdx(m);
        const isCurrentSlotPick = usedInSlot === currentSlot;
        const playerColor =
          m.players >= 5
            ? theme.colors.error
            : m.players >= 4
            ? theme.colors.warning
            : theme.colors.info;
        const slotChip = usedInSlot >= 0 ? `[${(["A", "B", "C"][usedInSlot] ?? "?")}]` : "   ";
        const slotChipColor = isCurrentSlotPick
          ? theme.colors.success
          : usedInSlot >= 0
          ? theme.colors.warning
          : theme.colors.mutedForeground;
        return (
          <Box key={m.path ?? m.name} gap={1}>
            <Text color={isCursor ? theme.colors.primary : undefined}>
              {isCursor ? "›" : " "}
            </Text>
            <Text color={slotChipColor} bold>
              {slotChip}
            </Text>
            <Text color={playerColor} bold>
              {m.players}p
            </Text>
            <Text
              color={isCursor ? theme.colors.primary : usedInSlot >= 0 ? undefined : theme.colors.mutedForeground}
              bold={isCursor || usedInSlot >= 0}
            >
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

function AlignList({
  slotLabels,
  slotMaps,
  slotAlignX,
  slotAlignY,
  cursor,
  active,
}: {
  slotLabels: string[];
  slotMaps: (ScannedMap | null)[];
  slotAlignX: AlignX[];
  slotAlignY: AlignY[];
  cursor: number;
  active: boolean;
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {slotMaps.map((m, idx) => {
        const isCursor = active && idx === cursor;
        const label = slotLabels[idx] ?? `Slot ${idx + 1}`;
        const ax = slotAlignX[idx] ?? "center";
        const ay = slotAlignY[idx] ?? "center";
        const labelColor = m
          ? isCursor ? theme.colors.primary : undefined
          : theme.colors.mutedForeground;
        return (
          <Box key={idx} gap={1}>
            <Text color={isCursor ? theme.colors.primary : undefined}>
              {isCursor ? "›" : " "}
            </Text>
            <Text color={theme.colors.primary} bold>
              {(["A", "B", "C"][idx] ?? "?")}
            </Text>
            <Text dimColor>·</Text>
            <Text color={labelColor}>{label.padEnd(24)}</Text>
            <Text dimColor>x:</Text>
            <Text color={ax === "center" ? theme.colors.mutedForeground : theme.colors.success} bold>
              {ALIGN_X_LABEL[ax].padEnd(7)}
            </Text>
            <Text dimColor>y:</Text>
            <Text color={ay === "center" ? theme.colors.mutedForeground : theme.colors.success} bold>
              {ALIGN_Y_LABEL[ay]}
            </Text>
            {!m && <Text color={theme.colors.warning}>(slot empty)</Text>}
          </Box>
        );
      })}
    </Box>
  );
}

function ParamList({
  rows,
  op,
  cursor,
  active,
  slotPlayers,
  preset,
}: {
  rows: ParamRow[];
  op: ComposeSetupOptions;
  cursor: number;
  active: boolean;
  slotPlayers: number[];
  preset: ComposePreset;
}) {
  const theme = useTheme();
  return (
    <Box flexDirection="column">
      {rows.map((row, idx) => {
        const isCursor = active && idx === cursor;
        const value = op[row.key];
        let valueColor: string | undefined = theme.colors.success;
        if (preset === "duplicate" && (row.key === "nx" || row.key === "ny")) {
          const projected = composeProjectedPlayers(preset, slotPlayers, op.nx, op.ny);
          if (projected > MAX_PLAYERS) valueColor = theme.colors.error;
        }
        return (
          <Box key={row.key} flexDirection="column">
            <Box gap={1}>
              <Text color={isCursor ? theme.colors.primary : undefined}>
                {isCursor ? "›" : " "}
              </Text>
              <Text color={isCursor ? theme.colors.primary : undefined} bold={isCursor}>
                {row.label.padEnd(28)}
              </Text>
              <Text color={valueColor} bold>
                {value}
              </Text>
              <Text dimColor>{`(${row.min}–${row.max})`}</Text>
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

function OptionList({
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

function Footer({ canStart, zone }: { canStart: boolean; zone: FocusZone }) {
  const theme = useTheme();
  const keys: { key: string; label: string }[] = [];
  if (zone === "preset") {
    keys.push({ key: "↑↓ / j k", label: "preset" });
  } else if (zone === "slots") {
    keys.push(
      { key: "↑↓ / j k", label: "scroll maps" },
      { key: "space", label: "assign to slot" },
      { key: "[ ]", label: "switch slot" },
      { key: "x", label: "clear slot" }
    );
  } else if (zone === "align") {
    keys.push(
      { key: "↑↓ / j k", label: "navigate slot" },
      { key: "← →", label: "cycle align_x" },
      { key: ", .", label: "cycle align_y" },
      { key: "0", label: "reset to center" }
    );
  } else if (zone === "params") {
    keys.push(
      { key: "↑↓", label: "navigate" },
      { key: "←→ / h l", label: "± value" },
      { key: "shift+←→", label: "±10" }
    );
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

interface ComposeKeyHandlerProps {
  eligible: ScannedMap[];
  slotCount: number;
  paramRows: ParamRow[];
  zone: FocusZone;
  presetCursor: number;
  slotCursor: number;
  mapCursor: number;
  alignCursor: number;
  paramCursor: number;
  optCursor: number;
  optionRows: { toggle: () => void }[];
  canStart: boolean;
  options: ComposeSetupOptions;
  setZone: (z: FocusZone | ((p: FocusZone) => FocusZone)) => void;
  setPresetCursor: (i: number | ((p: number) => number)) => void;
  setSlotCursor: (i: number | ((p: number) => number)) => void;
  setMapCursor: (i: number | ((p: number) => number)) => void;
  setAlignCursor: (i: number | ((p: number) => number)) => void;
  setParamCursor: (i: number | ((p: number) => number)) => void;
  setOptCursor: (i: number | ((p: number) => number)) => void;
  setPreset: (preset: ComposePreset) => void;
  setSlotMap: (slot: number, m: ScannedMap | null) => void;
  setSlotAlignX: (slot: number, value: AlignX) => void;
  setSlotAlignY: (slot: number, value: AlignY) => void;
  cycleAlignX: (slot: number, dir: 1 | -1) => void;
  cycleAlignY: (slot: number, dir: 1 | -1) => void;
  setParam: (key: ParamRow["key"], v: number) => void;
  onStart: () => void;
  onBack: () => void;
  onRescan: () => void;
  onQuit: () => void;
}

function ComposeKeyHandler({
  eligible,
  slotCount,
  paramRows,
  zone,
  presetCursor,
  slotCursor,
  mapCursor,
  alignCursor,
  paramCursor,
  optCursor,
  optionRows,
  canStart,
  options,
  setZone,
  setPresetCursor,
  setSlotCursor,
  setMapCursor,
  setAlignCursor,
  setParamCursor,
  setOptCursor,
  setPreset,
  setSlotMap,
  setSlotAlignX,
  setSlotAlignY,
  cycleAlignX,
  cycleAlignY,
  setParam,
  onStart,
  onBack,
  onRescan,
  onQuit,
}: ComposeKeyHandlerProps) {
  useInput((input, key) => {
    if (key.return) {
      if (!canStart) return;
      onStart();
      return;
    }
    if (input === "q" || (key.ctrl && input === "c")) { onQuit(); return; }
    if (input === "b") { onBack(); return; }
    if (input === "r") { onRescan(); return; }
    if (key.tab) {
      setZone((z) =>
        z === "preset" ? "slots" :
        z === "slots" ? "align" :
        z === "align" ? "params" :
        z === "params" ? "options" : "preset"
      );
      return;
    }

    if (zone === "preset") {
      if (key.upArrow || input === "k") {
        const next = Math.max(0, presetCursor - 1);
        setPresetCursor(next);
        setPreset(COMPOSE_PRESETS[next]!.key);
      } else if (key.downArrow || input === "j") {
        const next = Math.min(COMPOSE_PRESETS.length - 1, presetCursor + 1);
        setPresetCursor(next);
        setPreset(COMPOSE_PRESETS[next]!.key);
      }
    } else if (zone === "slots") {
      if (key.upArrow || input === "k") {
        setMapCursor((i) => Math.max(0, i - 1));
      } else if (key.downArrow || input === "j") {
        setMapCursor((i) => Math.min(eligible.length - 1, i + 1));
      } else if (input === "[") {
        setSlotCursor((i) => Math.max(0, i - 1));
      } else if (input === "]") {
        setSlotCursor((i) => Math.min(slotCount - 1, i + 1));
      } else if (input === " ") {
        const m = eligible[mapCursor];
        if (m) {
          setSlotMap(slotCursor, m);
          // Auto-advance to next empty slot if there is one.
          for (let i = (slotCursor + 1) % slotCount; i !== slotCursor; i = (i + 1) % slotCount) {
            if (!options.slotMaps[i]) { setSlotCursor(i); break; }
          }
        }
      } else if (input === "x") {
        setSlotMap(slotCursor, null);
      } else if (key.pageUp || input === "g") {
        setMapCursor(0);
      } else if (key.pageDown || input === "G") {
        setMapCursor(Math.max(0, eligible.length - 1));
      }
    } else if (zone === "align") {
      if (key.upArrow || input === "k") {
        setAlignCursor((i) => Math.max(0, i - 1));
      } else if (key.downArrow || input === "j") {
        setAlignCursor((i) => Math.min(slotCount - 1, i + 1));
      } else if (key.leftArrow || input === "h") {
        cycleAlignX(alignCursor, -1);
      } else if (key.rightArrow || input === "l") {
        cycleAlignX(alignCursor, +1);
      } else if (input === ",") {
        cycleAlignY(alignCursor, -1);
      } else if (input === ".") {
        cycleAlignY(alignCursor, +1);
      } else if (input === "0") {
        setSlotAlignX(alignCursor, "center");
        setSlotAlignY(alignCursor, "center");
      }
    } else if (zone === "params") {
      if (paramRows.length === 0) return;
      if (key.upArrow || input === "k") {
        setParamCursor((i) => Math.max(0, i - 1));
      } else if (key.downArrow || input === "j") {
        setParamCursor((i) => Math.min(paramRows.length - 1, i + 1));
      } else {
        const row = paramRows[paramCursor];
        if (!row) return;
        const step = key.shift ? 10 : 1;
        const cur = options[row.key];
        if (key.leftArrow || input === "h") {
          setParam(row.key, Math.max(row.min, cur - step));
        } else if (key.rightArrow || input === "l") {
          setParam(row.key, Math.min(row.max, cur + step));
        }
      }
    } else {
      if (key.upArrow || input === "k") setOptCursor((i) => Math.max(0, i - 1));
      else if (key.downArrow || input === "j") setOptCursor((i) => Math.min(optionRows.length - 1, i + 1));
      else if (input === " ") optionRows[optCursor]?.toggle();
    }
  });
  return null;
}
