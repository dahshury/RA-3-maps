import { useCallback, useEffect, useReducer, useRef } from "react";
import { Box, Text, useApp } from "ink";

import { Heading } from "@/components/ui/heading";
import { Banner } from "@/components/ui/banner";
import { useTheme } from "@/components/ui/theme-provider";
import { useInput } from "@/hooks/use-input";
import {
  runCompose,
  runConverter,
  runRotate,
  runSkin,
  type SpawnOptions,
} from "@/lib/runner";
import {
  defaultOptions,
  type ComposeEvent,
  type ConversionOptions,
  type ProgressEvent,
  type RotateEvent,
  type SkinEvent,
  type SkinVariantName,
} from "@/lib/events";

import { ScanningScreen } from "@/screens/scanning";
import { SetupScreen, type ScannedMap } from "@/screens/setup";
import { ConvertingScreen, type ConvertingState, type CompletedItem } from "@/screens/converting";
import { DoneScreen } from "@/screens/done";
import { ModeSelectScreen, type ToolkitMode } from "@/screens/mode-select";
import {
  SkinSetupScreen,
  defaultSkinOptions,
  type SkinOptions,
} from "@/screens/skin-setup";
import {
  SkinConvertingScreen,
  type SkinConvertingState,
  type SkinCompletedItem,
} from "@/screens/skin-converting";
import {
  RotateSetupScreen,
  defaultRotateSetupOptions,
  type RotateSetupOptions,
} from "@/screens/rotate-setup";
import {
  RotateConvertingScreen,
  type RotateConvertingState,
  type RotateCompletedItem,
} from "@/screens/rotate-converting";
import {
  ComposeSetupScreen,
  defaultComposeSetupOptions,
  type ComposeSetupOptions,
} from "@/screens/compose-setup";
import {
  ComposeConvertingScreen,
  type ComposeConvertingState,
  type ComposeCompletedItem,
} from "@/screens/compose-converting";

type Phase =
  | "mode_select"
  | "scanning"
  | "setup"
  | "converting"
  | "skin_setup"
  | "skin_converting"
  | "rotate_setup"
  | "rotate_converting"
  | "compose_setup"
  | "compose_converting"
  | "done"
  | "skin_done"
  | "rotate_done"
  | "compose_done"
  | "fatal";

interface State {
  mode: ToolkitMode;
  phase: Phase;
  inputDir: string;
  outputDir: string;
  scannedMaps: ScannedMap[];
  skipped: { name: string; reason: string }[];
  options: ConversionOptions;
  current: ConvertingState | null;
  completed: CompletedItem[];
  successCount: number;
  failCount: number;
  // Skin state
  skinOptions: SkinOptions;
  skinState: SkinConvertingState | null;
  skinRecent: SkinCompletedItem[];
  skinSummary: { success: number; fail: number; output: string } | null;
  // Rotate state
  rotateOptions: RotateSetupOptions;
  rotateState: RotateConvertingState | null;
  rotateRecent: RotateCompletedItem[];
  rotateSummary: { success: number; fail: number; output: string } | null;
  // Compose state
  composeOptions: ComposeSetupOptions;
  composeState: ComposeConvertingState | null;
  composeRecent: ComposeCompletedItem[];
  composeSummary: { success: number; fail: number; output: string } | null;
  fatal: string | null;
  exitCode: number | null;
}

const initialState: State = {
  mode: "convert",
  phase: "mode_select",
  inputDir: "",
  outputDir: "",
  scannedMaps: [],
  skipped: [],
  options: defaultOptions(),
  current: null,
  completed: [],
  successCount: 0,
  failCount: 0,
  skinOptions: defaultSkinOptions(),
  skinState: null,
  skinRecent: [],
  skinSummary: null,
  rotateOptions: defaultRotateSetupOptions(),
  rotateState: null,
  rotateRecent: [],
  rotateSummary: null,
  composeOptions: defaultComposeSetupOptions(),
  composeState: null,
  composeRecent: [],
  composeSummary: null,
  fatal: null,
  exitCode: null,
};

type Action =
  | { type: "event"; ev: ProgressEvent }
  | { type: "skin_event"; ev: SkinEvent }
  | { type: "rotate_event"; ev: RotateEvent }
  | { type: "compose_event"; ev: ComposeEvent }
  | { type: "exit"; code: number | null }
  | { type: "stderr"; chunk: string }
  | { type: "options"; opts: ConversionOptions }
  | { type: "skin_options"; opts: SkinOptions }
  | { type: "rotate_options"; opts: RotateSetupOptions }
  | { type: "compose_options"; opts: ComposeSetupOptions }
  | { type: "pick_mode"; mode: ToolkitMode }
  | { type: "back_to_mode" }
  | { type: "start_convert" }
  | { type: "start_skin" }
  | { type: "start_rotate" }
  | { type: "start_compose" }
  | { type: "rescan" }
  | { type: "rescan_skin" }
  | { type: "rescan_rotate" }
  | { type: "rescan_compose" }
  | { type: "convert_finished" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "stderr":
      return state;

    case "options":
      return { ...state, options: action.opts };

    case "skin_options":
      return { ...state, skinOptions: action.opts };

    case "rotate_options":
      return { ...state, rotateOptions: action.opts };

    case "compose_options":
      return { ...state, composeOptions: action.opts };

    case "pick_mode":
      return {
        ...state,
        mode: action.mode,
        phase: "scanning",
        scannedMaps: [],
        skipped: [],
        current: null,
        completed: [],
        successCount: 0,
        failCount: 0,
        skinState: null,
        skinRecent: [],
        skinSummary: null,
        rotateState: null,
        rotateRecent: [],
        rotateSummary: null,
      };

    case "back_to_mode":
      return {
        ...initialState,
        // Preserve user's last options across mode swaps.
        options: state.options,
        skinOptions: { ...state.skinOptions, source: null },
        rotateOptions: { ...state.rotateOptions, source: null },
        composeOptions: { ...state.composeOptions, slotMaps: [null, null, null] },
      };

    case "rescan":
      return {
        ...state,
        phase: "scanning",
        scannedMaps: [],
        skipped: [],
        current: null,
        completed: [],
        successCount: 0,
        failCount: 0,
      };

    case "rescan_skin":
      return {
        ...state,
        phase: "scanning",
        scannedMaps: [],
        skipped: [],
        skinOptions: { ...state.skinOptions, source: null },
        skinState: null,
        skinRecent: [],
        skinSummary: null,
      };

    case "rescan_rotate":
      return {
        ...state,
        phase: "scanning",
        scannedMaps: [],
        skipped: [],
        rotateOptions: { ...state.rotateOptions, source: null },
        rotateState: null,
        rotateRecent: [],
        rotateSummary: null,
      };

    case "rescan_compose":
      return {
        ...state,
        phase: "scanning",
        scannedMaps: [],
        skipped: [],
        composeOptions: { ...state.composeOptions, slotMaps: [null, null, null] },
        composeState: null,
        composeRecent: [],
        composeSummary: null,
      };

    case "start_convert":
      return {
        ...state,
        phase: "converting",
        current: null,
        completed: [],
        successCount: 0,
        failCount: 0,
      };

    case "start_skin":
      return {
        ...state,
        phase: "skin_converting",
        skinState: null,
        skinRecent: [],
        skinSummary: null,
      };

    case "start_rotate":
      return {
        ...state,
        phase: "rotate_converting",
        rotateState: null,
        rotateRecent: [],
        rotateSummary: null,
      };

    case "start_compose":
      return {
        ...state,
        phase: "compose_converting",
        composeState: null,
        composeRecent: [],
        composeSummary: null,
      };

    case "convert_finished":
      return { ...state, phase: "done" };

    case "exit":
      if (
        state.phase === "fatal" ||
        state.phase === "done" ||
        state.phase === "skin_done" ||
        state.phase === "rotate_done" ||
        state.phase === "compose_done"
      ) {
        return { ...state, exitCode: action.code };
      }
      return state;

    case "skin_event": {
      const { ev } = action;
      switch (ev.event) {
        case "skin_start": {
          const total = ev.total;
          return {
            ...state,
            skinState: {
              sourceName: state.skinOptions.source?.name ?? "",
              outputDir: ev.output,
              current: null,
              step: null,
              index: 0,
              total,
              completed: new Set(),
              failed: new Set(),
              successCount: 0,
              failCount: 0,
            },
            skinRecent: [],
          };
        }
        case "skin_variant_start": {
          if (!state.skinState) return state;
          return {
            ...state,
            skinState: {
              ...state.skinState,
              current: ev.name,
              step: null,
              index: ev.index,
              total: ev.total,
            },
          };
        }
        case "skin_step": {
          if (!state.skinState) return state;
          return {
            ...state,
            skinState: { ...state.skinState, step: ev.step },
          };
        }
        case "skin_variant_complete": {
          if (!state.skinState) return state;
          const completed = new Set(state.skinState.completed);
          const failed = new Set(state.skinState.failed);
          if (ev.success) completed.add(ev.name as SkinVariantName);
          else failed.add(ev.name as SkinVariantName);
          const recent = [
            ...state.skinRecent,
            { name: ev.name as SkinVariantName, success: ev.success, error: ev.error },
          ].slice(-3);
          return {
            ...state,
            skinState: {
              ...state.skinState,
              completed,
              failed,
              successCount: state.skinState.successCount + (ev.success ? 1 : 0),
              failCount: state.skinState.failCount + (ev.success ? 0 : 1),
            },
            skinRecent: recent,
          };
        }
        case "skin_done":
          return {
            ...state,
            phase: "skin_done",
            skinSummary: {
              success: ev.success,
              fail: ev.fail,
              output: ev.output,
            },
          };
        case "fatal":
          return { ...state, phase: "fatal", fatal: ev.error };
        default:
          return state;
      }
    }

    case "compose_event": {
      const { ev } = action;
      switch (ev.event) {
        case "compose_start": {
          const basenames = ev.maps.map((p) => {
            const norm = p.replace(/\\/g, "/");
            return norm.slice(norm.lastIndexOf("/") + 1);
          });
          return {
            ...state,
            composeState: {
              preset: ev.preset,
              maps: basenames,
              outputDir: ev.output,
              step: null,
              detail: null,
              done: false,
              success: null,
            },
            composeRecent: [],
          };
        }
        case "compose_step": {
          if (!state.composeState) return state;
          return {
            ...state,
            composeState: {
              ...state.composeState,
              step: ev.step,
              detail: ev.detail ?? null,
            },
          };
        }
        case "compose_complete": {
          if (!state.composeState) return state;
          const item: ComposeCompletedItem = {
            preset: ev.preset,
            maps: state.composeState.maps,
            success: ev.success,
            error: ev.error,
          };
          const recent = [...state.composeRecent, item].slice(-3);
          return {
            ...state,
            composeState: {
              ...state.composeState,
              done: true,
              success: ev.success,
              error: ev.error,
            },
            composeRecent: recent,
          };
        }
        case "compose_done":
          return {
            ...state,
            phase: "compose_done",
            composeSummary: {
              success: ev.success,
              fail: ev.fail,
              output: ev.output,
            },
          };
        case "fatal":
          return { ...state, phase: "fatal", fatal: ev.error };
        default:
          return state;
      }
    }

    case "rotate_event": {
      const { ev } = action;
      switch (ev.event) {
        case "rotate_start": {
          return {
            ...state,
            rotateState: {
              sourceName: state.rotateOptions.source?.name ?? "",
              outputDir: ev.output,
              currentOp: null,
              currentSource: "",
              step: null,
              index: 0,
              total: ev.total_ops,
              successCount: 0,
              failCount: 0,
            },
            rotateRecent: [],
          };
        }
        case "rotate_op_start": {
          if (!state.rotateState) return state;
          return {
            ...state,
            rotateState: {
              ...state.rotateState,
              currentOp: ev.op,
              currentSource: ev.source,
              step: null,
              index: ev.index,
              total: ev.total,
            },
          };
        }
        case "rotate_step": {
          if (!state.rotateState) return state;
          return {
            ...state,
            rotateState: { ...state.rotateState, step: ev.step },
          };
        }
        case "rotate_op_complete": {
          if (!state.rotateState) return state;
          const item: RotateCompletedItem = {
            source: state.rotateState.currentSource,
            op: ev.op,
            success: ev.success,
            error: ev.error,
          };
          const recent = [...state.rotateRecent, item].slice(-3);
          return {
            ...state,
            rotateState: {
              ...state.rotateState,
              successCount: state.rotateState.successCount + (ev.success ? 1 : 0),
              failCount: state.rotateState.failCount + (ev.success ? 0 : 1),
            },
            rotateRecent: recent,
          };
        }
        case "rotate_done":
          return {
            ...state,
            phase: "rotate_done",
            rotateSummary: {
              success: ev.success,
              fail: ev.fail,
              output: ev.output,
            },
          };
        case "fatal":
          return { ...state, phase: "fatal", fatal: ev.error };
        default:
          return state;
      }
    }

    case "event": {
      const { ev } = action;
      switch (ev.event) {
        case "start":
          return {
            ...state,
            inputDir: ev.input,
            outputDir: ev.output,
          };

        case "scan_start":
          return { ...state, phase: "scanning", scannedMaps: [], skipped: [] };

        case "scan_progress":
          if (ev.players === 1 || ev.players === 2 || ev.players === 3) {
            return {
              ...state,
              scannedMaps: [
                ...state.scannedMaps,
                {
                  name: ev.name,
                  path: ev.path,
                  players: ev.players === 1 ? 2 : ev.players,
                },
              ],
            };
          }
          // Skin / rotate / compose modes accept >3p sources too — record
          // them with the real count so the user can still pick them
          // (compose mode validates the per-slot 2p minimum + 6p total cap
          // at start time).
          if (
            (state.mode === "skin" ||
              state.mode === "rotate" ||
              state.mode === "compose") &&
            typeof ev.players === "number" &&
            ev.players > 0
          ) {
            return {
              ...state,
              scannedMaps: [
                ...state.scannedMaps,
                {
                  name: ev.name,
                  path: ev.path,
                  players: ev.players,
                },
              ],
            };
          }
          return {
            ...state,
            skipped: [...state.skipped, { name: ev.name, reason: ev.reason }],
          };

        case "scan_complete":
          return {
            ...state,
            phase:
              state.mode === "skin"
                ? "skin_setup"
                : state.mode === "rotate"
                ? "rotate_setup"
                : state.mode === "compose"
                ? "compose_setup"
                : "setup",
            options:
              state.mode === "convert"
                ? {
                    ...state.options,
                    selectedMaps: state.scannedMaps.map((m) => m.name),
                  }
                : state.options,
          };

        case "convert_start":
          return {
            ...state,
            phase: "converting",
            current: {
              name: ev.name,
              archonName: ev.archon_name,
              index: ev.index,
              total: ev.total,
              playerCount: ev.player_count,
              applyArchon: !!ev.apply_archon,
              variation: ev.variation || "",
              step: null,
              completedSteps: new Set(),
            },
          };

        case "convert_step": {
          if (!state.current) return state;
          const prev = state.current;
          const completedSteps = new Set(prev.completedSteps);
          if (prev.step && prev.step !== ev.step) {
            completedSteps.add(prev.step);
          }
          return {
            ...state,
            current: { ...prev, step: ev.step, completedSteps },
          };
        }

        case "convert_complete": {
          const item: CompletedItem = {
            name: ev.name,
            archonName: ev.archon_name,
            success: ev.success,
            error: ev.error,
          };
          return {
            ...state,
            current: null,
            completed: [...state.completed, item],
            successCount: state.successCount + (ev.success ? 1 : 0),
            failCount: state.failCount + (ev.success ? 0 : 1),
          };
        }

        case "done":
          if (ev.scan_only) return state;
          return {
            ...state,
            phase: "done",
            successCount: ev.success,
            failCount: ev.fail,
          };

        case "fatal":
          return { ...state, phase: "fatal", fatal: ev.error };

        default:
          return state;
      }
    }
  }
}

export interface AppProps {
  internalDir: string;
  inputDir?: string;
  outputDir?: string;
}

export function App({ internalDir, inputDir, outputDir }: AppProps) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const { exit } = useApp();
  const childRef = useRef<{ kill: () => boolean } | null>(null);

  const spawn = useCallback(
    (extra: Partial<SpawnOptions>) => {
      childRef.current?.kill();
      const child = runConverter({
        internalDir,
        inputDir,
        outputDir,
        jsonProgress: true,
        onEvent: (ev) => dispatch({ type: "event", ev }),
        onStderr: (chunk) => dispatch({ type: "stderr", chunk }),
        onExit: (code) => dispatch({ type: "exit", code }),
        ...extra,
      });
      childRef.current = child;
      return child;
    },
    [internalDir, inputDir, outputDir]
  );

  const spawnSkin = useCallback(
    (sourcePath: string, opts: SkinOptions) => {
      childRef.current?.kill();
      const child = runSkin({
        internalDir,
        inputDir,
        outputDir,
        source: sourcePath,
        noRender: !opts.render,
        noCompress: !opts.compress,
        onEvent: (ev) => dispatch({ type: "skin_event", ev }),
        onStderr: (chunk) => dispatch({ type: "stderr", chunk }),
        onExit: (code) => dispatch({ type: "exit", code }),
      });
      childRef.current = child;
      return child;
    },
    [internalDir, inputDir, outputDir]
  );

  const spawnRotate = useCallback(
    (opts: RotateSetupOptions) => {
      childRef.current?.kill();
      const sourcePath = opts.batchAll || !opts.source
        ? null
        : (opts.source.path ?? opts.source.name);
      const child = runRotate({
        internalDir,
        inputDir,
        outputDir,
        source: sourcePath,
        ops: opts.ops,
        noCompress: !opts.compress,
        onEvent: (ev) => dispatch({ type: "rotate_event", ev }),
        onStderr: (chunk) => dispatch({ type: "stderr", chunk }),
        onExit: (code) => dispatch({ type: "exit", code }),
      });
      childRef.current = child;
      return child;
    },
    [internalDir, inputDir, outputDir]
  );

  const spawnCompose = useCallback(
    (opts: ComposeSetupOptions) => {
      childRef.current?.kill();
      // Translate ComposeSetupOptions -> ComposeOp expected by runCompose.
      // Collect non-null slots in order so a gap (e.g. slots A and C with
      // empty B) compacts to two consecutive cells -- only meaningful for
      // row/col where order is the only thing that matters anyway.
      const filledIndices: number[] = [];
      opts.slotMaps.forEach((m, i) => { if (m) filledIndices.push(i); });
      const slotPaths = filledIndices.map((i) => {
        const m = opts.slotMaps[i]!;
        return m.path ?? m.name;
      });
      const slotAlignX = filledIndices.map((i) => opts.slotAlignX[i] ?? "center");
      const slotAlignY = filledIndices.map((i) => opts.slotAlignY[i] ?? "center");
      const child = runCompose({
        internalDir,
        inputDir,
        outputDir,
        op: {
          preset: opts.preset,
          maps: slotPaths,
          nx: opts.nx,
          ny: opts.ny,
          padX: opts.padX,
          padY: opts.padY,
          alignX: slotAlignX,
          alignY: slotAlignY,
        },
        noCompress: !opts.compress,
        onEvent: (ev) => dispatch({ type: "compose_event", ev }),
        onStderr: (chunk) => dispatch({ type: "stderr", chunk }),
        onExit: (code) => dispatch({ type: "exit", code }),
      });
      childRef.current = child;
      return child;
    },
    [internalDir, inputDir, outputDir]
  );

  // Mode pick triggers the initial scan. For skin/rotate/compose modes,
  // scan with apply-archon off so >3p maps aren't filtered out (compose
  // mode validates the per-slot cap at start time).
  const handlePickMode = useCallback(
    (mode: ToolkitMode) => {
      dispatch({ type: "pick_mode", mode });
      if (mode === "skin" || mode === "rotate" || mode === "compose") {
        spawn({
          scanOnly: true,
          conversion: { ...state.options, applyArchon: false },
        });
      } else {
        spawn({ scanOnly: true, conversion: state.options });
      }
    },
    [spawn, state.options]
  );

  const handleStart = useCallback(() => {
    dispatch({ type: "start_convert" });
    spawn({ scanOnly: false, conversion: state.options });
  }, [spawn, state.options]);

  const handleSkinStart = useCallback(() => {
    const src = state.skinOptions.source;
    if (!src) return;
    const sourcePath = src.path ?? src.name;
    dispatch({ type: "start_skin" });
    spawnSkin(sourcePath, state.skinOptions);
  }, [spawnSkin, state.skinOptions]);

  const handleRotateStart = useCallback(() => {
    if (state.rotateOptions.ops.length === 0) return;
    if (!state.rotateOptions.batchAll && !state.rotateOptions.source) return;
    dispatch({ type: "start_rotate" });
    spawnRotate(state.rotateOptions);
  }, [spawnRotate, state.rotateOptions]);

  const handleComposeStart = useCallback(() => {
    const o = state.composeOptions;
    // Guard: enforce minimum slot fill before spawning the engine.
    const filled = o.slotMaps.filter((m): m is NonNullable<typeof m> => m !== null).length;
    const minRequired =
      o.preset === "duplicate" ? 1
      : o.preset === "row" || o.preset === "col" ? 2
      : 3;
    if (filled < minRequired) return;
    if (o.preset === "duplicate" && o.nx === 1 && o.ny === 1) return;
    dispatch({ type: "start_compose" });
    spawnCompose(o);
  }, [spawnCompose, state.composeOptions]);

  // Auto-start hook for non-TTY smoke tests.
  useEffect(() => {
    if (
      (process.env.RA3_TOOLKIT_AUTO_START === "1" ||
        process.env.ARCHON_AUTO_START === "1") &&
      state.phase === "setup" &&
      state.scannedMaps.length > 0
    ) {
      handleStart();
    }
  }, [state.phase, state.scannedMaps.length, handleStart]);

  const handleRescan = useCallback(() => {
    dispatch({ type: "rescan" });
    spawn({ scanOnly: true, conversion: state.options });
  }, [spawn, state.options]);

  const handleSkinRescan = useCallback(() => {
    dispatch({ type: "rescan_skin" });
    spawn({
      scanOnly: true,
      conversion: { ...state.options, applyArchon: false },
    });
  }, [spawn, state.options]);

  const handleRotateRescan = useCallback(() => {
    dispatch({ type: "rescan_rotate" });
    spawn({
      scanOnly: true,
      conversion: { ...state.options, applyArchon: false },
    });
  }, [spawn, state.options]);

  const handleComposeRescan = useCallback(() => {
    dispatch({ type: "rescan_compose" });
    spawn({
      scanOnly: true,
      conversion: { ...state.options, applyArchon: false },
    });
  }, [spawn, state.options]);

  // Re-scan when archon toggle changes (convert mode only).
  const lastApplyArchon = useRef(state.options.applyArchon);
  useEffect(() => {
    if (
      state.mode === "convert" &&
      lastApplyArchon.current !== state.options.applyArchon &&
      state.phase === "setup"
    ) {
      lastApplyArchon.current = state.options.applyArchon;
      dispatch({ type: "rescan" });
      spawn({ scanOnly: true, conversion: state.options });
    } else {
      lastApplyArchon.current = state.options.applyArchon;
    }
  }, [state.options.applyArchon, state.phase, state.mode, spawn, state.options]);

  const handleQuit = useCallback(() => {
    childRef.current?.kill();
    exit();
  }, [exit]);

  const handleBackToMode = useCallback(() => {
    childRef.current?.kill();
    dispatch({ type: "back_to_mode" });
  }, []);

  return (
    <Box flexDirection="column" paddingX={1} paddingY={0}>
      <Header phase={state.phase} mode={state.mode} options={state.options} />

      {state.phase === "mode_select" && (
        <ModeSelectScreen onPick={handlePickMode} onQuit={handleQuit} />
      )}

      {state.phase === "scanning" && (
        <ScanningScreen
          scannedCount={state.scannedMaps.length + state.skipped.length}
          lastFile={
            state.scannedMaps[state.scannedMaps.length - 1]?.name ??
            state.skipped[state.skipped.length - 1]?.name
          }
        />
      )}

      {state.phase === "setup" && (
        <SetupScreen
          inputDir={state.inputDir}
          outputDir={state.outputDir}
          maps={state.scannedMaps}
          skipped={state.skipped}
          options={state.options}
          onChange={(opts) => dispatch({ type: "options", opts })}
          onStart={handleStart}
          onRescan={handleRescan}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "skin_setup" && (
        <SkinSetupScreen
          inputDir={state.inputDir}
          outputDir={state.outputDir}
          maps={state.scannedMaps}
          skipped={state.skipped}
          options={state.skinOptions}
          onChange={(opts) => dispatch({ type: "skin_options", opts })}
          onStart={handleSkinStart}
          onBack={handleBackToMode}
          onRescan={handleSkinRescan}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "rotate_setup" && (
        <RotateSetupScreen
          inputDir={state.inputDir}
          outputDir={state.outputDir}
          maps={state.scannedMaps}
          skipped={state.skipped}
          options={state.rotateOptions}
          onChange={(opts) => dispatch({ type: "rotate_options", opts })}
          onStart={handleRotateStart}
          onBack={handleBackToMode}
          onRescan={handleRotateRescan}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "compose_setup" && (
        <ComposeSetupScreen
          inputDir={state.inputDir}
          outputDir={state.outputDir}
          maps={state.scannedMaps}
          skipped={state.skipped}
          options={state.composeOptions}
          onChange={(opts) => dispatch({ type: "compose_options", opts })}
          onStart={handleComposeStart}
          onBack={handleBackToMode}
          onRescan={handleComposeRescan}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "converting" && (
        <ConvertingScreen
          current={state.current}
          successCount={state.successCount}
          failCount={state.failCount}
          recent={state.completed.slice(-3)}
        />
      )}

      {state.phase === "skin_converting" && (
        <SkinConvertingScreen state={state.skinState} recent={state.skinRecent} />
      )}

      {state.phase === "rotate_converting" && (
        <RotateConvertingScreen state={state.rotateState} recent={state.rotateRecent} />
      )}

      {state.phase === "compose_converting" && (
        <ComposeConvertingScreen state={state.composeState} recent={state.composeRecent} />
      )}

      {state.phase === "done" && (
        <DoneScreen
          success={state.successCount}
          fail={state.failCount}
          skipped={state.skipped.length}
          completed={state.completed}
          inputDir={state.inputDir}
          outputDir={state.outputDir}
          onAgain={handleRescan}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "skin_done" && state.skinSummary && (
        <SkinDoneScreen
          summary={state.skinSummary}
          sourceName={state.skinOptions.source?.name ?? ""}
          onAgain={handleBackToMode}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "rotate_done" && state.rotateSummary && (
        <RotateDoneScreen
          summary={state.rotateSummary}
          sourceName={
            state.rotateOptions.batchAll
              ? `${state.scannedMaps.length} map(s)`
              : state.rotateOptions.source?.name ?? ""
          }
          onAgain={handleBackToMode}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "compose_done" && state.composeSummary && (
        <ComposeDoneScreen
          summary={state.composeSummary}
          sourceLabel={(() => {
            const o = state.composeOptions;
            const names = o.slotMaps
              .filter((m): m is NonNullable<typeof m> => m !== null)
              .map((m) => m.name);
            return names.length > 0
              ? `${o.preset} · ${names.join(" + ")}`
              : o.preset;
          })()}
          onAgain={handleBackToMode}
          onQuit={handleQuit}
        />
      )}

      {state.phase === "fatal" && state.fatal && (
        <Box marginTop={1}>
          <Banner variant="error" title="Fatal error">
            {state.fatal}
          </Banner>
        </Box>
      )}
    </Box>
  );
}

function SkinDoneScreen({
  summary,
  sourceName,
  onAgain,
  onQuit,
}: {
  summary: { success: number; fail: number; output: string };
  sourceName: string;
  onAgain: () => void;
  onQuit: () => void;
}) {
  const theme = useTheme();
  const variant = summary.fail === 0 ? "success" : summary.success === 0 ? "error" : "warning";
  return (
    <Box marginTop={1} flexDirection="column">
      <Banner variant={variant} title="Skin complete">
        {summary.success}/{summary.success + summary.fail} variants from {sourceName} ready in {summary.output}.
      </Banner>
      <Box marginTop={1} flexDirection="row" gap={2}>
        <Box gap={1}>
          <Text bold color={theme.colors.primary}>a</Text>
          <Text dimColor>back to menu</Text>
        </Box>
        <Box gap={1}>
          <Text bold color={theme.colors.primary}>q / enter</Text>
          <Text dimColor>quit</Text>
        </Box>
      </Box>
      <SkinDoneKeys onAgain={onAgain} onQuit={onQuit} />
    </Box>
  );
}

function SkinDoneKeys({
  onAgain,
  onQuit,
}: {
  onAgain: () => void;
  onQuit: () => void;
}) {
  useInput((input, key) => {
    if (input === "a") onAgain();
    if (input === "q" || key.return) onQuit();
  });
  return null;
}

function RotateDoneScreen({
  summary,
  sourceName,
  onAgain,
  onQuit,
}: {
  summary: { success: number; fail: number; output: string };
  sourceName: string;
  onAgain: () => void;
  onQuit: () => void;
}) {
  const theme = useTheme();
  const variant =
    summary.fail === 0 ? "success" : summary.success === 0 ? "error" : "warning";
  return (
    <Box marginTop={1} flexDirection="column">
      <Banner variant={variant} title="Rotation complete">
        {summary.success}/{summary.success + summary.fail} outputs from {sourceName} ready in {summary.output}.
      </Banner>
      <Box marginTop={1} flexDirection="row" gap={2}>
        <Box gap={1}>
          <Text bold color={theme.colors.primary}>a</Text>
          <Text dimColor>back to menu</Text>
        </Box>
        <Box gap={1}>
          <Text bold color={theme.colors.primary}>q / enter</Text>
          <Text dimColor>quit</Text>
        </Box>
      </Box>
      <SkinDoneKeys onAgain={onAgain} onQuit={onQuit} />
    </Box>
  );
}

function ComposeDoneScreen({
  summary,
  sourceLabel,
  onAgain,
  onQuit,
}: {
  summary: { success: number; fail: number; output: string };
  sourceLabel: string;
  onAgain: () => void;
  onQuit: () => void;
}) {
  const theme = useTheme();
  const variant =
    summary.fail === 0 ? "success" : summary.success === 0 ? "error" : "warning";
  return (
    <Box marginTop={1} flexDirection="column">
      <Banner variant={variant} title="Composition complete">
        {summary.success}/{summary.success + summary.fail} output(s) for {sourceLabel} ready in {summary.output}.
      </Banner>
      <Box marginTop={1} flexDirection="row" gap={2}>
        <Box gap={1}>
          <Text bold color={theme.colors.primary}>a</Text>
          <Text dimColor>back to menu</Text>
        </Box>
        <Box gap={1}>
          <Text bold color={theme.colors.primary}>q / enter</Text>
          <Text dimColor>quit</Text>
        </Box>
      </Box>
      <SkinDoneKeys onAgain={onAgain} onQuit={onQuit} />
    </Box>
  );
}

function Header({ phase, mode }: { phase: Phase; mode: ToolkitMode; options: ConversionOptions }) {
  const theme = useTheme();
  const phaseLabel: Record<Phase, string> = {
    mode_select: "Pick mode",
    scanning: "Scanning",
    setup: "Setup",
    converting: "Converting",
    skin_setup: "Skin setup",
    skin_converting: "Skinning",
    rotate_setup: "Rotate setup",
    rotate_converting: "Rotating",
    compose_setup: "Compose setup",
    compose_converting: "Composing",
    done: "Done",
    skin_done: "Done",
    rotate_done: "Done",
    compose_done: "Done",
    fatal: "Error",
  };
  const tagline =
    mode === "skin"
      ? "Decompose one map into 8 layered strip variants for layer analysis."
      : mode === "rotate"
      ? "Rotate / flip RA3 maps. Pick one or more ops per source; outputs land in converted_maps."
      : mode === "compose"
      ? "Stitch 1–3 maps into one output via row · col · triangle · duplicate presets (≤6 players total)."
      : "Batch-transform RA3 maps. Pick one or more transforms; combine them as you like.";
  return (
    <Box flexDirection="column">
      <Box flexDirection="row" gap={2} alignItems="center">
        <Heading level={1} prefix1="▰▰▰ ">
          RA3 Map Toolkit
        </Heading>
        <Box>
          <Text color={theme.colors.mutedForeground}>·</Text>
          <Text color={theme.colors.primary} bold>
            {" "}
            {phaseLabel[phase]}
          </Text>
        </Box>
      </Box>
      <Box>
        <Text dimColor>{tagline}</Text>
      </Box>
    </Box>
  );
}

export default App;
