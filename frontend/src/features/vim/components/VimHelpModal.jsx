import React from "react";

const VimHelpModal = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const shortcuts = [
    { key: "Tab", desc: "Switch active tab (Conversation ↔ Ad-Hoc Search)" },
    { key: "/", desc: "Focus search query input & enter INSERT mode" },
    {
      key: "Esc",
      desc: "Exit INSERT mode back to NORMAL mode (or close popups)",
    },
    { key: "t", desc: "Set Top-K number (Quick prompt)" },
    { key: "a", desc: "Toggle Search Mode (Accurate ↔ Fast)" },
    { key: "n", desc: "Create new Conversation session" },
    { key: "h", desc: "Toggle Conversation History list" },
    { key: "o", desc: "Toggle Options Drawer" },
    { key: "?", desc: "Toggle this Vim Keyboard Shortcuts help cheat sheet" },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="vim-help-card"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="vim-help-header">
          <div>
            <span className="vim-prompt-tag">KEYBOARD NAVIGATION</span>
            <h3 className="vim-help-title">Vim Keybindings Cheat Sheet</h3>
          </div>
          <button
            type="button"
            className="modal-close-btn"
            onClick={onClose}
            title="Close [Esc]"
          >
            ✕
          </button>
        </div>

        <div className="vim-help-body">
          <table className="vim-shortcuts-table">
            <thead>
              <tr>
                <th>Key</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {shortcuts.map((item) => (
                <tr key={item.key}>
                  <td>
                    <kbd className="vim-kbd">{item.key}</kbd>
                  </td>
                  <td>{item.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default VimHelpModal;
