import React, { useId } from "react";
import ManualSubmissionBox from "./ManualSubmissionBox";

const TOP_K_MIN = 1;
const TOP_K_MAX = 100;
const SUGGEST_COUNT_MIN = 1;
const SUGGEST_COUNT_MAX = 10;

// Contains the user-tunable search controls.
const ToolBox = ({
  topK,
  setTopK,
  suggestCount,
  setSuggestCount,
  onReset,
}) => {
  const topKSliderId = useId();
  const suggestCountSliderId = useId();
  
  const topKProgress =
    ((topK - TOP_K_MIN) / (TOP_K_MAX - TOP_K_MIN)) * 100;
  
  const suggestProgress = suggestCount !== undefined
    ? ((suggestCount - SUGGEST_COUNT_MIN) / (SUGGEST_COUNT_MAX - SUGGEST_COUNT_MIN)) * 100
    : 0;

  return (
    <aside className="toolbox-sidebar">
      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor={topKSliderId} className="toolbox-label">
            Top-K results
          </label>
          <output className="toolbox-value" htmlFor={topKSliderId}>
            {topK}
          </output>
        </div>
        <input
          id={topKSliderId}
          type="range"
          min={TOP_K_MIN}
          max={TOP_K_MAX}
          step="1"
          value={topK}
          onChange={(event) => setTopK(Number(event.target.value))}
          className="toolbox-slider"
          style={{
            "--slider-progress": `${Math.min(100, Math.max(0, topKProgress))}%`,
          }}
        />
      </div>

      {setSuggestCount && (
        <div className="toolbox-section">
          <div className="toolbox-label-row">
            <label htmlFor={suggestCountSliderId} className="toolbox-label">
              Suggestion Count
            </label>
            <output className="toolbox-value" htmlFor={suggestCountSliderId}>
              {suggestCount}
            </output>
          </div>
          <input
            id={suggestCountSliderId}
            type="range"
            min={SUGGEST_COUNT_MIN}
            max={SUGGEST_COUNT_MAX}
            step="1"
            value={suggestCount}
            onChange={(event) => setSuggestCount(Number(event.target.value))}
            className="toolbox-slider"
            style={{
              "--slider-progress": `${Math.min(100, Math.max(0, suggestProgress))}%`,
            }}
          />
        </div>
      )}

      <button className="btn-utility toolbox-reset-btn" onClick={onReset} style={{ marginBottom: '12px' }}>
        Reset Parameters
      </button>

      <ManualSubmissionBox />
    </aside>
  );
};

export default ToolBox;
