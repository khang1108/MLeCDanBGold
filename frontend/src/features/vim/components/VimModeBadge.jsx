import React from "react";

const VimModeBadge = ({ mode, onToggleMode }) => {
  const isNormal = mode === "NORMAL";

  return (
    <button
      type="button"
      className={`vim-badge ${isNormal ? "mode-normal" : "mode-insert"}`}
      onClick={onToggleMode}
      title={
        isNormal
          ? "NORMAL mode active. Press '/' to insert, 't' for Top-K, '?' for help."
          : "INSERT mode active. Press 'Esc' to exit to NORMAL mode."
      }
    >
      <span className="vim-dot" />
      <span className="vim-mode-name">{mode}</span>
    </button>
  );
};

export default VimModeBadge;
