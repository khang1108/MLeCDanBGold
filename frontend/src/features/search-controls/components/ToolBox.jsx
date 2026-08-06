import React, { useId } from "react";

const TOP_K_MIN = 1;
const TOP_K_MAX = 100;
const SUGGESTION_COUNT_MIN = 5;
const SUGGESTION_COUNT_MAX = 10;

// Contains the user-tunable search and suggestion controls.
const ToolBox = ({
  topK,
  setTopK,
  suggestionCount,
  setSuggestionCount,
  onReset,
}) => {
  const topKSliderId = useId();
  const suggestionSliderId = useId();
  const progress =
    ((topK - TOP_K_MIN) / (TOP_K_MAX - TOP_K_MIN)) * 100;
  const suggestionProgress =
    ((suggestionCount - SUGGESTION_COUNT_MIN) /
      (SUGGESTION_COUNT_MAX - SUGGESTION_COUNT_MIN)) *
    100;

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
            "--slider-progress": `${Math.min(100, Math.max(0, progress))}%`,
          }}
        />
      </div>

      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor={suggestionSliderId} className="toolbox-label">
            Query suggestions
          </label>
          <output className="toolbox-value" htmlFor={suggestionSliderId}>
            {suggestionCount}
          </output>
        </div>
        <input
          id={suggestionSliderId}
          type="range"
          min={SUGGESTION_COUNT_MIN}
          max={SUGGESTION_COUNT_MAX}
          step="1"
          value={suggestionCount}
          onChange={(event) => setSuggestionCount(Number(event.target.value))}
          className="toolbox-slider"
          style={{
            "--slider-progress": `${Math.min(
              100,
              Math.max(0, suggestionProgress),
            )}%`,
          }}
        />
      </div>

      <button className="btn-utility toolbox-reset-btn" onClick={onReset}>
        Reset Parameters
      </button>
    </aside>
  );
};

export default ToolBox;