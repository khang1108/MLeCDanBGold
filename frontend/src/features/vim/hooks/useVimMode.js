import { useCallback, useEffect, useState } from "react";

export const useVimMode = ({
  onCloseAllModals,
  queryInputRef,
  enableTopK = true,
}) => {
  const [mode, setMode] = useState("NORMAL"); // 'NORMAL' | 'INSERT'
  const [isTopKOpen, setIsTopKOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const enterInsertMode = useCallback(() => {
    setMode("INSERT");
    setTimeout(() => {
      queryInputRef?.current?.focus();
    }, 10);
  }, [queryInputRef]);

  const enterNormalMode = useCallback(() => {
    setMode("NORMAL");
    queryInputRef?.current?.blur();
  }, [queryInputRef]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      const activeElement = document.activeElement;
      const isInputFocused =
        activeElement &&
        ["INPUT", "TEXTAREA", "SELECT"].includes(activeElement.tagName);

      // Handle Escape in any mode
      if (event.key === "Escape") {
        if (isTopKOpen) {
          setIsTopKOpen(false);
          return;
        }
        if (isHelpOpen) {
          setIsHelpOpen(false);
          return;
        }
        onCloseAllModals?.();
        enterNormalMode();
        return;
      }

      // If typing in input or in INSERT mode, allow normal typing; Esc exits INSERT mode
      if (isInputFocused || mode === "INSERT") {
        return;
      }

      // --- NORMAL MODE KEYBINDINGS ---

      // / -> Focus search query input and enter INSERT mode
      if (event.key === "/") {
        event.preventDefault();
        enterInsertMode();
        return;
      }

      // 4. t -> Open Top-K quick edit dialog
      if (enableTopK && event.key.toLowerCase() === "t") {
        event.preventDefault();
        setIsTopKOpen(true);
        return;
      }

      // ? -> Open Vim Shortcuts Help Cheat Sheet
      if (event.key === "?") {
        event.preventDefault();
        setIsHelpOpen((prev) => !prev);
        return;
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [
    enterInsertMode,
    enterNormalMode,
    isHelpOpen,
    isTopKOpen,
    mode,
    enableTopK,
    onCloseAllModals,
  ]);

  return {
    mode,
    setMode,
    isTopKOpen,
    setIsTopKOpen,
    isHelpOpen,
    setIsHelpOpen,
    enterInsertMode,
    enterNormalMode,
  };
};
