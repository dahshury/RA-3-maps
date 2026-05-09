import { execSync, spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import {
  parseComposeEvent,
  parseEvent,
  parseRotateEvent,
  parseSkinEvent,
  type ComposeEvent,
  type ComposeOp,
  type ConversionOptions,
  type ProgressEvent,
  type RotateEvent,
  type RotateOp,
  type SkinEvent,
} from "./events";

function whichUv(): string | null {
  try {
    const out = execSync("where uv", { encoding: "utf8" });
    const first = out.split(/\r?\n/).find((l) => l.trim().endsWith(".exe"));
    return first ? first.trim() : null;
  } catch {
    return null;
  }
}

function pickCommand(internalDir: string): { cmd: string; args: string[] } {
  // Prefer the bundled ra3_engine.exe in production. Fall back to
  // `uv run python batch_convert.py` for dev when no exe is present.
  const exe = path.join(internalDir, "ra3_engine.exe");
  if (existsSync(exe)) {
    return { cmd: exe, args: [] };
  }
  const uvPath = whichUv();
  if (!uvPath) {
    throw new Error(
      "ra3_engine.exe not found and 'uv' is not on PATH. Build the exe (build.bat) or install uv for dev mode."
    );
  }
  return {
    cmd: uvPath,
    args: ["run", "python", "batch_convert.py"],
  };
}

export interface SpawnOptions {
  internalDir: string;
  inputDir?: string;
  outputDir?: string;
  jsonProgress?: boolean;
  scanOnly?: boolean;
  conversion?: ConversionOptions;
  onEvent: (ev: ProgressEvent) => void;
  onStderr?: (chunk: string) => void;
  onExit: (code: number | null, signal: NodeJS.Signals | null) => void;
}

function buildArgs(opts: SpawnOptions): string[] {
  const positional: string[] = [];
  if (opts.inputDir) positional.push(opts.inputDir);
  if (opts.outputDir) positional.push(opts.outputDir);

  const flags: string[] = [];
  if (opts.jsonProgress ?? true) flags.push("--json-progress");
  if (opts.scanOnly) flags.push("--scan-only");

  const c = opts.conversion;
  if (c) {
    if (c.applyArchon) flags.push("--apply-archon");
    if (c.restrictions.length > 0) {
      flags.push("--restrictions", c.restrictions.join(","));
    }
    if (!c.compress) flags.push("--no-compress");
    if (!c.writeSidecars) flags.push("--no-sidecars");
    if (c.applyArchon) {
      if (c.wbNormalizeTerrain) flags.push("--wb-normalize-terrain");
      if (c.offset !== 800) flags.push("--offset", String(c.offset));
    }
  }

  // --maps takes nargs=*, must come last so trailing tokens don't slurp
  // unrelated args.
  const trailing: string[] = [];
  if (c && c.selectedMaps.length > 0) {
    trailing.push("--maps", ...c.selectedMaps);
  }

  return [...positional, ...flags, ...trailing];
}

export function runConverter(opts: SpawnOptions): ChildProcessWithoutNullStreams {
  const { cmd, args: baseArgs } = pickCommand(opts.internalDir);
  const args = [...baseArgs, ...buildArgs(opts)];
  const child = spawn(cmd, args, {
    cwd: opts.internalDir,
    windowsHide: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  }) as ChildProcessWithoutNullStreams;

  let buf = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      const ev = parseEvent(line);
      if (ev) opts.onEvent(ev);
    }
  });

  child.stdout.on("end", () => {
    if (buf.trim()) {
      const ev = parseEvent(buf);
      if (ev) opts.onEvent(ev);
    }
  });

  if (opts.onStderr) {
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", opts.onStderr);
  }

  child.on("close", (code, signal) => opts.onExit(code, signal));

  return child;
}

// ---------------------------------------------------------------------------
// Skin / decompose mode. Spawns the same engine binary with --mode skin and
// streams the SkinEvent JSON union.
// ---------------------------------------------------------------------------

export interface SkinSpawnOptions {
  internalDir: string;
  inputDir?: string;
  outputDir?: string;
  source: string; // absolute path to the .map file to skin
  noRender?: boolean;
  noCompress?: boolean;
  onEvent: (ev: SkinEvent) => void;
  onStderr?: (chunk: string) => void;
  onExit: (code: number | null, signal: NodeJS.Signals | null) => void;
}

function buildSkinArgs(opts: SkinSpawnOptions): string[] {
  const positional: string[] = [];
  if (opts.inputDir) positional.push(opts.inputDir);
  if (opts.outputDir) positional.push(opts.outputDir);

  const flags: string[] = ["--json-progress", "--mode", "skin",
                           "--skin-source", opts.source];
  if (opts.noRender) flags.push("--no-render");
  if (opts.noCompress) flags.push("--no-compress");

  return [...positional, ...flags];
}

export function runSkin(opts: SkinSpawnOptions): ChildProcessWithoutNullStreams {
  const { cmd, args: baseArgs } = pickCommand(opts.internalDir);
  const args = [...baseArgs, ...buildSkinArgs(opts)];
  const child = spawn(cmd, args, {
    cwd: opts.internalDir,
    windowsHide: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  }) as ChildProcessWithoutNullStreams;

  let buf = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      const ev = parseSkinEvent(line);
      if (ev) opts.onEvent(ev);
    }
  });

  child.stdout.on("end", () => {
    if (buf.trim()) {
      const ev = parseSkinEvent(buf);
      if (ev) opts.onEvent(ev);
    }
  });

  if (opts.onStderr) {
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", opts.onStderr);
  }

  child.on("close", (code, signal) => opts.onExit(code, signal));

  return child;
}

// ---------------------------------------------------------------------------
// Rotate / flip mode. Spawns the engine binary with --mode rotate.
// ---------------------------------------------------------------------------

export interface RotateSpawnOptions {
  internalDir: string;
  inputDir?: string;
  outputDir?: string;
  source: string | null;  // absolute path to single .map; null = batch the input folder
  ops: RotateOp[];
  noCompress?: boolean;
  onEvent: (ev: RotateEvent) => void;
  onStderr?: (chunk: string) => void;
  onExit: (code: number | null, signal: NodeJS.Signals | null) => void;
}

function buildRotateArgs(opts: RotateSpawnOptions): string[] {
  const positional: string[] = [];
  if (opts.inputDir) positional.push(opts.inputDir);
  if (opts.outputDir) positional.push(opts.outputDir);

  const flags: string[] = [
    "--json-progress",
    "--mode", "rotate",
    "--rotate-ops", opts.ops.join(","),
  ];
  if (opts.source) {
    flags.push("--rotate-source", opts.source);
  }
  if (opts.noCompress) flags.push("--no-compress");

  return [...positional, ...flags];
}

export function runRotate(opts: RotateSpawnOptions): ChildProcessWithoutNullStreams {
  const { cmd, args: baseArgs } = pickCommand(opts.internalDir);
  const args = [...baseArgs, ...buildRotateArgs(opts)];
  const child = spawn(cmd, args, {
    cwd: opts.internalDir,
    windowsHide: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  }) as ChildProcessWithoutNullStreams;

  let buf = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      const ev = parseRotateEvent(line);
      if (ev) opts.onEvent(ev);
    }
  });

  child.stdout.on("end", () => {
    if (buf.trim()) {
      const ev = parseRotateEvent(buf);
      if (ev) opts.onEvent(ev);
    }
  });

  if (opts.onStderr) {
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", opts.onStderr);
  }

  child.on("close", (code, signal) => opts.onExit(code, signal));

  return child;
}

// ---------------------------------------------------------------------------
// Compose / stitch mode. Spawns the engine binary with --mode compose.
// Composition is a strict superset of the old duplicate mode; the
// `duplicate` preset re-implements the original tile-Nx*Ny behaviour.
// ---------------------------------------------------------------------------

export interface ComposeSpawnOptions {
  internalDir: string;
  inputDir?: string;
  outputDir?: string;
  op: ComposeOp;
  noCompress?: boolean;
  onEvent: (ev: ComposeEvent) => void;
  onStderr?: (chunk: string) => void;
  onExit: (code: number | null, signal: NodeJS.Signals | null) => void;
}

function buildComposeArgs(opts: ComposeSpawnOptions): string[] {
  const positional: string[] = [];
  if (opts.inputDir) positional.push(opts.inputDir);
  if (opts.outputDir) positional.push(opts.outputDir);

  const flags: string[] = [
    "--json-progress",
    "--mode", "compose",
    "--compose-preset", opts.op.preset,
  ];
  if (opts.op.preset === "duplicate") {
    flags.push("--compose-nx", String(opts.op.nx));
    flags.push("--compose-ny", String(opts.op.ny));
  }
  if (opts.op.padX > 0) flags.push("--compose-pad-x", String(opts.op.padX));
  if (opts.op.padY > 0) flags.push("--compose-pad-y", String(opts.op.padY));
  // Only forward alignment when at least one slot is non-default; saves
  // typing in the ps audit log and keeps the engine fast-path clean.
  const ax = opts.op.alignX.slice(0, opts.op.maps.length);
  const ay = opts.op.alignY.slice(0, opts.op.maps.length);
  if (ax.some((a) => a !== "center")) {
    flags.push("--compose-align-x", ax.join(","));
  }
  if (ay.some((a) => a !== "center")) {
    flags.push("--compose-align-y", ay.join(","));
  }
  if (opts.noCompress) flags.push("--no-compress");

  // --compose-maps takes nargs=+, must come last so trailing tokens don't
  // get pulled into other flags.
  const trailing: string[] = ["--compose-maps", ...opts.op.maps];

  return [...positional, ...flags, ...trailing];
}

export function runCompose(opts: ComposeSpawnOptions): ChildProcessWithoutNullStreams {
  const { cmd, args: baseArgs } = pickCommand(opts.internalDir);
  const args = [...baseArgs, ...buildComposeArgs(opts)];
  const child = spawn(cmd, args, {
    cwd: opts.internalDir,
    windowsHide: true,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  }) as ChildProcessWithoutNullStreams;

  let buf = "";
  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    buf += chunk;
    let nl: number;
    while ((nl = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      const ev = parseComposeEvent(line);
      if (ev) opts.onEvent(ev);
    }
  });

  child.stdout.on("end", () => {
    if (buf.trim()) {
      const ev = parseComposeEvent(buf);
      if (ev) opts.onEvent(ev);
    }
  });

  if (opts.onStderr) {
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", opts.onStderr);
  }

  child.on("close", (code, signal) => opts.onExit(code, signal));

  return child;
}
