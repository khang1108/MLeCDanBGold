import React, { useEffect, useId, useState } from "react";
import SubmissionWorktree from "../../../features/submission/components/SubmissionWorktree";

const TOP_K_MIN = 1;
const TOP_K_MAX = 100;
const TOP_K_PRESETS = [10, 20, 50, 100];

/**
 * User-tunable search controls with direct numeric input and quick presets.
 */
const ToolBox = ({ topK, setTopK, onReset }) => {
  const topKInputId = useId();
  const [topKText, setTopKText] = useState(String(topK));

  useEffect(() => {
    setTopKText(String(topK));
  }, [topK]);

  const commitTopKValue = (val) => {
    let num = parseInt(val, 10);
    if (Number.isNaN(num)) {
      num = topK || 20;
    }
    const clamped = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, num));
    setTopK(clamped);
    setTopKText(String(clamped));
  };

  const handleTopKTextChange = (e) => {
    const val = e.target.value;
    setTopKText(val);
    const num = parseInt(val, 10);
    if (!Number.isNaN(num) && num >= TOP_K_MIN && num <= TOP_K_MAX) {
      setTopK(num);
    }
  };

  const handleTopKKeyDown = (e) => {
    if (e.key === "Enter") {
      commitTopKValue(topKText);
      e.target.blur();
    }
  };

  const handleTopKBlur = () => {
    commitTopKValue(topKText);
  };

  const stepTopK = (delta) => {
    const current = parseInt(topKText, 10) || topK || 20;
    const nextVal = Math.max(TOP_K_MIN, Math.min(TOP_K_MAX, current + delta));
    setTopK(nextVal);
    setTopKText(String(nextVal));
  };

  return (
    <aside className="toolbox-sidebar">
      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor={topKInputId} className="toolbox-label">
            Top-K results
          </label>
        </div>

        <div className="toolbox-direct-input-container">
          <div className="toolbox-stepper-row">
            <button
              type="button"
              className="toolbox-stepper-btn"
              onClick={() => stepTopK(-1)}
              disabled={topK <= TOP_K_MIN}
              aria-label="Decrease Top-K"
            >
              −
            </button>
            <input
              id={topKInputId}
              type="number"
              min={TOP_K_MIN}
              max={TOP_K_MAX}
              value={topKText}
              onChange={handleTopKTextChange}
              onKeyDown={handleTopKKeyDown}
              onBlur={handleTopKBlur}
              className="toolbox-number-input"
              placeholder="20"
              aria-label="Top-K value"
            />
            <button
              type="button"
              className="toolbox-stepper-btn"
              onClick={() => stepTopK(1)}
              disabled={topK >= TOP_K_MAX}
              aria-label="Increase Top-K"
            >
              +
            </button>
          </div>
          <div className="toolbox-presets-row">
            {TOP_K_PRESETS.map((preset) => (
              <button
                key={preset}
                type="button"
                className={`toolbox-preset-chip ${topK === preset ? "active" : ""}`}
                onClick={() => {
                  setTopK(preset);
                  setTopKText(String(preset));
                }}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>
      </div>

      <button type="button" className="btn-utility toolbox-reset-btn" onClick={onReset}>
        Reset Parameters
      </button>

      <SubmissionWorktree />
    </aside>
  );
};

export default ToolBox;
