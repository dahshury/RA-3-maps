// LOCAL PATCH: registry version called Ink's useInput unconditionally, which
// throws in non-TTY contexts (raw mode unsupported). Wrap so it becomes a
// no-op when the terminal can't enter raw mode. Re-apply if you re-run
// `shadcn add @termcn/use-input --overwrite`.
import { useInput as inkUseInput, useStdin } from "ink";

export interface Key {
  upArrow: boolean;
  downArrow: boolean;
  leftArrow: boolean;
  rightArrow: boolean;
  pageDown: boolean;
  pageUp: boolean;
  return: boolean;
  escape: boolean;
  ctrl: boolean;
  shift: boolean;
  tab: boolean;
  backspace: boolean;
  delete: boolean;
  meta: boolean;
  eventType?: "press" | "repeat" | "release";
  home?: boolean;
  end?: boolean;
  fn?: boolean;
}

export type InputHandler = (input: string, key: Key) => void;

export const useInput = (
  handler: InputHandler,
  options?: { isActive?: boolean }
): void => {
  const { isRawModeSupported } = useStdin();
  const active = (options?.isActive ?? true) && isRawModeSupported;
  inkUseInput(handler, { isActive: active });
};
