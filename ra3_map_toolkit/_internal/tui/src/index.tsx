import { render } from "ink";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ThemeProvider } from "@/components/ui/theme-provider";
import App from "@/app";

function resolveInternalDir(): string {
  // When packaged via `bun build --compile`, process.execPath is the final
  // exe (in _internal/). In dev (`bun src/index.tsx`), execPath is bun.exe
  // and we use the source file location instead.
  const exeName = path.basename(process.execPath).toLowerCase();
  if (
    exeName === "ra3_map_toolkit.exe" ||
    exeName === "ra3_engine.exe" ||
    // Back-compat: still accept old names.
    exeName === "archon_transformer.exe" ||
    exeName === "archon_converter.exe"
  ) {
    return path.dirname(process.execPath);
  }

  let here: string;
  try {
    here = path.dirname(fileURLToPath(import.meta.url));
  } catch {
    here = process.cwd();
  }
  return path.resolve(here, "..", "..");
}

const internalDir = resolveInternalDir();
const argv = process.argv.slice(2);
const inputDir = argv[0];
const outputDir = argv[1];

render(
  <ThemeProvider>
    <App internalDir={internalDir} inputDir={inputDir} outputDir={outputDir} />
  </ThemeProvider>
);
