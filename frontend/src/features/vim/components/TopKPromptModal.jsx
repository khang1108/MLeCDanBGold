import React, { useEffect, useRef, useState } from "react";

const TopKPromptModal = ({ isOpen, currentTopK, onSave, onClose }) => {
  const [val, setVal] = useState(String(currentTopK));
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setVal(String(currentTopK));
      setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 30);
    }
  }, [isOpen, currentTopK]);

  if (!isOpen) return null;

  const handleSubmit = (event) => {
    event.preventDefault();
    const num = parseInt(val, 10);
    if (!isNaN(num) && num > 0 && num <= 100) {
      onSave(num);
    }
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="vim-prompt-card"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="vim-prompt-header">
          <span className="vim-prompt-tag">VIM SHORTCUT</span>
          <h3 className="vim-prompt-title">Set Top-K Results</h3>
        </div>
        <form onSubmit={handleSubmit} className="vim-prompt-form">
          <input
            ref={inputRef}
            type="number"
            min="1"
            max="100"
            value={val}
            onChange={(e) => setVal(e.target.value)}
            className="input-text vim-prompt-input"
            placeholder="e.g. 15"
          />
          <div className="vim-prompt-actions">
            <button type="button" className="btn-utility" onClick={onClose}>
              Cancel [Esc]
            </button>
            <button type="submit" className="btn-primary">
              Set Top-K [Enter]
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default TopKPromptModal;
