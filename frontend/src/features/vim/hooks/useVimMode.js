import { useCallback, useEffect, useState } from "react";

export const useVimMode = ({
  activeTab,
  setActiveTab,
  topK,
  setTopK,
  queryType,
  setQueryType,
  onNewSession,
  onToggleHistory,
  onToggleOptions,
  onCloseAllModals,
  queryInputRef,
  adhocQueryInputRef,
}) => {
  const [mode, setMode] = useState("NORMAL"); // 'NORMAL' | 'INSERT'
  const [isTopKOpen, setIsTopKOpen] = useState(false);
  const [isHelpOpen, setIsHelpOpen] = useState(false);

  const enterInsertMode = useCallback(() => {
    setMode("INSERT");
    setTimeout(() => {
      const targetRef =
        activeTab === "ad_hoc" ? adhocQueryInputRef : queryInputRef;
      targetRef?.current?.focus();
    }, 10);
  }, [activeTab, adhocQueryInputRef, queryInputRef]);

  const enterNormalMode = useCallback(() => {
    setMode("NORMAL");
    queryInputRef?.current?.blur();
    adhocQueryInputRef?.current?.blur();
  }, [adhocQueryInputRef, queryInputRef]);

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

      // 1. Keys 1 -> 5: Select Query Type ('kis', 'kisc', 'vkis', 'vqa', 'trake')
      if (["1", "2", "3", "4", "5"].includes(event.key)) {
        event.preventDefault();
        const typeMap = {
          1: "kis",
          2: "kisc",
          3: "vkis",
          4: "vqa",
          5: "trake",
        };
        if (typeMap[event.key]) {
          setQueryType?.(typeMap[event.key]);
        }
        return;
      }

      // 2. Tab -> Switch between Conversation & AdHoc tabs
      if (event.key === "Tab") {
        event.preventDefault();
        setActiveTab((prev) =>
          prev === "conversation" ? "ad_hoc" : "conversation",
        );
        return;
      }

      // 3. / -> Focus search query input and enter INSERT mode
      if (event.key === "/") {
        event.preventDefault();
        enterInsertMode();
        return;
      }

      // 4. t -> Open Top-K quick edit dialog
      if (event.key.toLowerCase() === "t") {
        event.preventDefault();
        setIsTopKOpen(true);
        return;
      }

      // 5. n -> Create new conversation session
      if (event.key.toLowerCase() === "n") {
        event.preventDefault();
        onNewSession?.();
        return;
      }

      // 6. h -> Toggle History popover
      if (event.key.toLowerCase() === "h") {
        event.preventDefault();
        onToggleHistory?.();
        return;
      }

      // 7. o -> Toggle Options drawer
      if (event.key.toLowerCase() === "o") {
        event.preventDefault();
        onToggleOptions?.();
        return;
      }

      // 8. ? -> Open Vim Shortcuts Help Cheat Sheet
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
    onCloseAllModals,
    onNewSession,
    onToggleHistory,
    onToggleOptions,
    setActiveTab,
    setQueryType,
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
