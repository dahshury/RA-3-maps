export type ConvertStep =
  | "parse_source"
  | "parse_archon_template"
  | "parse_restriction_donor"
  | "archon_transform"
  | "restriction_transform"
  | "save"
  | "minimap"
  | "sidecars"
  | "copy_template";

export const STEP_LABEL: Record<ConvertStep, string> = {
  parse_source: "Parsing source map",
  parse_archon_template: "Parsing archon template",
  parse_restriction_donor: "Parsing restriction donor",
  archon_transform: "Applying archon transform",
  restriction_transform: "Applying match restriction",
  copy_template: "Copying template",
  save: "Saving map",
  minimap: "Generating minimap",
  sidecars: "Writing metadata",
};

export const STEP_ORDER: ConvertStep[] = [
  "parse_source",
  "parse_archon_template",
  "parse_restriction_donor",
  "archon_transform",
  "restriction_transform",
  "copy_template",
  "save",
  "minimap",
  "sidecars",
];

export type MapRestriction =
  | "nosw"
  | "noair"
  | "nosw_noair"
  | "inf_only"
  | "tanks_only";

export const MAP_RESTRICTIONS: { key: MapRestriction; label: string; hint: string }[] = [
  { key: "nosw", label: "No Superweapons", hint: "Disables all faction superweapons." },
  { key: "noair", label: "No Air Units", hint: "Disables aircraft, AA structures and AA units." },
  {
    key: "nosw_noair",
    label: "No Superweapons + No Air",
    hint: "No Superweapons + No Air + No Upgrades.",
  },
  { key: "inf_only", label: "Infantry Only", hint: "Infantry-only; no war factories or airfields." },
  {
    key: "tanks_only",
    label: "Tanks Only",
    hint: "Vehicles only; no infantry, navy, or aircraft.",
  },
];

export type ProgressEvent =
  | {
      event: "start";
      input: string;
      output: string;
      scan_only?: boolean;
      apply_archon?: boolean;
      variations?: MapRestriction[];
    }
  | { event: "scan_start" }
  | {
      event: "scan_progress";
      name: string;
      path?: string;
      players: number | null;
      reason: string;
    }
  | {
      event: "scan_complete";
      total: number;
      by_2p: number;
      by_3p: number;
      skipped: { name: string; reason: string }[];
    }
  | {
      event: "convert_start";
      name: string;
      archon_name: string;
      index: number;
      total: number;
      player_count: number;
      apply_archon?: boolean;
      variation?: MapRestriction | "";
    }
  | {
      event: "convert_step";
      name: string;
      step: ConvertStep;
      detail?: string;
    }
  | {
      event: "convert_complete";
      name: string;
      archon_name: string;
      success: boolean;
      output?: string;
      error?: string;
      variation?: MapRestriction | "";
    }
  | {
      event: "done";
      success: number;
      fail: number;
      skipped: number;
      output?: string;
      scan_only?: boolean;
    }
  | { event: "fatal"; error: string };

// ---------------------------------------------------------------------------
// Skin / decompose mode events. Emitted by skin_map.py (via batch_convert.py
// --mode skin). Kept as a separate event union so the convert-mode reducer
// stays narrow.
// ---------------------------------------------------------------------------

export type SkinVariantName =
  | "iso1_no_blends"
  | "iso2_no_textures"
  | "iso3_no_objects"
  | "iso4_flat"
  | "cum1_blends_off"
  | "cum2_blends_textures_off"
  | "cum3_blends_textures_objects_off"
  | "cum4_skeleton";

export const SKIN_VARIANT_LABELS: Record<SkinVariantName, string> = {
  iso1_no_blends: "Isolated · Strip blends",
  iso2_no_textures: "Isolated · Strip textures",
  iso3_no_objects: "Isolated · Strip objects",
  iso4_flat: "Isolated · Flatten elevations",
  cum1_blends_off: "Cumulative · Blends off",
  cum2_blends_textures_off: "Cumulative · Blends + textures off",
  cum3_blends_textures_objects_off: "Cumulative · Blends + textures + objects off",
  cum4_skeleton: "Cumulative · Skeleton (everything off)",
};

export const SKIN_VARIANT_ORDER: SkinVariantName[] = [
  "iso1_no_blends",
  "iso2_no_textures",
  "iso3_no_objects",
  "iso4_flat",
  "cum1_blends_off",
  "cum2_blends_textures_off",
  "cum3_blends_textures_objects_off",
  "cum4_skeleton",
];

export type SkinEvent =
  | {
      event: "skin_start";
      source: string;
      output: string;
      total: number;
    }
  | {
      event: "skin_variant_start";
      index: number;
      total: number;
      name: SkinVariantName;
      description: string;
    }
  | {
      event: "skin_step";
      name: SkinVariantName;
      step: string;
    }
  | {
      event: "skin_variant_complete";
      index: number;
      total: number;
      name: SkinVariantName;
      success: boolean;
      output?: string;
      error?: string;
    }
  | {
      event: "skin_done";
      success: number;
      fail: number;
      total: number;
      output: string;
    }
  | { event: "fatal"; error: string };

export function parseSkinEvent(line: string): SkinEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    const obj = JSON.parse(trimmed);
    if (typeof obj === "object" && obj !== null && typeof obj.event === "string") {
      return obj as SkinEvent;
    }
  } catch {
    // not JSON
  }
  return null;
}

// ---------------------------------------------------------------------------
// Rotate / flip mode events. Emitted by rotate_engine.py (via batch_convert.py
// --mode rotate).
// ---------------------------------------------------------------------------

export type RotateOp = "rot90cw" | "rot90ccw" | "rot180" | "flipx" | "flipy";

export const ROTATE_OPS: { key: RotateOp; label: string; hint: string }[] = [
  { key: "rot90cw",  label: "Rotate 90° right (CW)", hint: "Quarter turn clockwise." },
  { key: "rot90ccw", label: "Rotate 90° left (CCW)", hint: "Quarter turn counter-clockwise." },
  { key: "rot180",   label: "Rotate 180°",           hint: "Half turn — equivalent to flipx + flipy." },
  { key: "flipx",    label: "Flip across X axis",    hint: "Top↔bottom mirror. Asymmetric meshes get a +180° fixup." },
  { key: "flipy",    label: "Flip across Y axis",    hint: "Left↔right mirror. Asymmetric meshes get a +180° fixup." },
];

export type RotateStep = "parse" | "rotate" | "save" | "tga";

export const ROTATE_STEP_LABEL: Record<RotateStep, string> = {
  parse: "Parsing source map",
  rotate: "Applying transform",
  save: "Saving rotated map",
  tga: "Copying preview TGA",
};

export type RotateEvent =
  | {
      event: "rotate_start";
      source: string;
      output: string;
      total_ops: number;
    }
  | {
      event: "rotate_op_start";
      index: number;
      total: number;
      op: RotateOp;
      source: string;
    }
  | {
      event: "rotate_step";
      op: RotateOp;
      step: RotateStep;
    }
  | {
      event: "rotate_op_complete";
      index: number;
      total: number;
      op: RotateOp;
      success: boolean;
      output?: string;
      error?: string;
    }
  | {
      event: "rotate_done";
      success: number;
      fail: number;
      total: number;
      output: string;
    }
  | { event: "fatal"; error: string };

export function parseRotateEvent(line: string): RotateEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    const obj = JSON.parse(trimmed);
    if (typeof obj === "object" && obj !== null && typeof obj.event === "string") {
      return obj as RotateEvent;
    }
  } catch {
    // not JSON
  }
  return null;
}

// ---------------------------------------------------------------------------
// Compose / stitch mode. Spawns the engine with --mode compose. Composition
// is a strict superset of the old "duplicate" mode -- duplication is just
// the `duplicate` preset with a single source tiled Nx*Ny times.
// ---------------------------------------------------------------------------

export type ComposePreset =
  | "duplicate"
  | "row"
  | "col"
  | "triangle_top"
  | "triangle_bottom"
  | "triangle_left"
  | "triangle_right";

export const COMPOSE_PRESETS: { key: ComposePreset; label: string; slots: number; hint: string; ascii: string[] }[] = [
  {
    key: "duplicate",
    label: "Duplicate / Tile",
    slots: 1,
    hint: "Tile a single source Nx×Ny times.",
    ascii: ["[A][A]", "[A][A]"],
  },
  {
    key: "row",
    label: "Row (X axis)",
    slots: 3,
    hint: "Two or three maps stacked horizontally, left to right.",
    ascii: ["[A][B][C]"],
  },
  {
    key: "col",
    label: "Column (Y axis)",
    slots: 3,
    hint: "Two or three maps stacked vertically, top to bottom.",
    ascii: ["[A]", "[B]", "[C]"],
  },
  {
    key: "triangle_top",
    label: "Triangle (2 top, 1 bottom)",
    slots: 3,
    hint: "Two maps top-row, third map centered under both.",
    ascii: ["[A][B]", "[ C  ]"],
  },
  {
    key: "triangle_bottom",
    label: "Triangle (1 top, 2 bottom)",
    slots: 3,
    hint: "First map top spanning, two maps bottom row.",
    ascii: ["[ A  ]", "[B][C]"],
  },
  {
    key: "triangle_left",
    label: "Triangle (2 left, 1 right)",
    slots: 3,
    hint: "Two maps stacked on the left, third map centered on the right.",
    ascii: ["[A][C]", "[B][C]"],
  },
  {
    key: "triangle_right",
    label: "Triangle (1 left, 2 right)",
    slots: 3,
    hint: "First map spanning the left column, two maps stacked on the right.",
    ascii: ["[A][B]", "[A][C]"],
  },
];

// For triangle_left/right, the maps[1] (B) is actually the second source --
// the rendered ASCII above re-uses A/C for the spanned cells but slot order
// for --maps is always {A, B, C} with the spanning map at position 0 or 2 as
// declared in compose_engine._build_spec.

export type AlignX = "left" | "center" | "right";
export type AlignY = "top" | "center" | "bottom";

export const ALIGN_X_VALUES: AlignX[] = ["left", "center", "right"];
export const ALIGN_Y_VALUES: AlignY[] = ["top", "center", "bottom"];

export const ALIGN_X_LABEL: Record<AlignX, string> = {
  left: "left",
  center: "center",
  right: "right",
};
export const ALIGN_Y_LABEL: Record<AlignY, string> = {
  top: "top",
  center: "center",
  bottom: "bottom",
};

export interface ComposeOp {
  preset: ComposePreset;
  maps: string[];   // absolute paths in slot order; length == slot count
  nx: number;       // duplicate preset only
  ny: number;       // duplicate preset only
  padX: number;     // tile-unit gap between adjacent X cells
  padY: number;     // tile-unit gap between adjacent Y cells
  // Per-slot alignment for sources smaller than their allocated cell. Length
  // matches `maps`. Defaults to ["center"...] / ["center"...] if absent.
  alignX: AlignX[];
  alignY: AlignY[];
}

export const MAX_PLAYERS = 6;

/** Number of map "slots" the preset uses (1 for duplicate, 2 for row/col with 2 sources, 3 otherwise). */
export function composePresetSlotCount(preset: ComposePreset, fillCount?: number): number {
  if (preset === "duplicate") return 1;
  // row / col accept 2 or 3 maps; default to 3 in the picker but accept 2.
  if (preset === "row" || preset === "col") {
    return Math.max(2, Math.min(3, fillCount ?? 3));
  }
  return 3;
}

export function composeOpLabel(op: ComposeOp): string {
  if (op.preset === "duplicate") {
    const parts: string[] = [];
    if (op.nx > 1) parts.push(`x${op.nx}`);
    if (op.ny > 1) parts.push(`y${op.ny}`);
    return parts.join("_") || "noop";
  }
  const short: Record<ComposePreset, string> = {
    duplicate: "dup",
    row: "row",
    col: "col",
    triangle_top: "triTop",
    triangle_bottom: "triBot",
    triangle_left: "triLeft",
    triangle_right: "triRight",
  };
  return short[op.preset];
}

/** Total players if `op` were composed from these per-slot player counts. */
export function composeProjectedPlayers(
  preset: ComposePreset,
  slotPlayers: number[],
  nx: number,
  ny: number
): number {
  if (preset === "duplicate") {
    return (slotPlayers[0] ?? 0) * Math.max(1, nx) * Math.max(1, ny);
  }
  return slotPlayers.reduce((s, n) => s + (n ?? 0), 0);
}

export function composeFitsCap(
  preset: ComposePreset,
  slotPlayers: number[],
  nx: number,
  ny: number
): boolean {
  return composeProjectedPlayers(preset, slotPlayers, nx, ny) <= MAX_PLAYERS;
}

export type ComposeStep = "parse" | "layout" | "compose" | "save" | "tga";

export const COMPOSE_STEP_LABEL: Record<ComposeStep, string> = {
  parse: "Parsing source maps",
  layout: "Solving layout",
  compose: "Stitching playable areas",
  save: "Saving composed map",
  tga: "Copying preview TGA",
};

export type ComposeEvent =
  | {
      event: "compose_start";
      preset: ComposePreset;
      maps: string[];
      output: string;
      total_steps: number;
    }
  | {
      event: "compose_step";
      step: ComposeStep;
      detail?: string;
    }
  | {
      event: "compose_complete";
      preset: ComposePreset;
      success: boolean;
      output?: string;
      error?: string;
    }
  | {
      event: "compose_done";
      success: number;
      fail: number;
      total: number;
      output: string;
    }
  | { event: "fatal"; error: string };

export function parseComposeEvent(line: string): ComposeEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    const obj = JSON.parse(trimmed);
    if (typeof obj === "object" && obj !== null && typeof obj.event === "string") {
      return obj as ComposeEvent;
    }
  } catch {
    // not JSON
  }
  return null;
}

export interface RotateOptions {
  source: { name: string; path: string } | null;  // null = whole input dir
  ops: RotateOp[];
  compress: boolean;
}

export const defaultRotateOptions = (): RotateOptions => ({
  source: null,
  ops: ["rot90cw"],
  compress: true,
});

export interface ConversionOptions {
  selectedMaps: string[]; // basenames; empty = none selected
  // Independent toggles. Both can be on for combined runs.
  applyArchon: boolean;
  restrictions: MapRestriction[];
  // Common knobs.
  compress: boolean;
  writeSidecars: boolean;
  // Archon-only knobs (ignored unless applyArchon).
  wbNormalizeTerrain: boolean;
  offset: number;
}

export const defaultOptions = (): ConversionOptions => ({
  selectedMaps: [],
  applyArchon: true,
  restrictions: [],
  compress: true,
  writeSidecars: true,
  wbNormalizeTerrain: false,
  offset: 800,
});

export function parseEvent(line: string): ProgressEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    const obj = JSON.parse(trimmed);
    if (typeof obj === "object" && obj !== null && typeof obj.event === "string") {
      return obj as ProgressEvent;
    }
  } catch {
    // not JSON — silently drop
  }
  return null;
}
