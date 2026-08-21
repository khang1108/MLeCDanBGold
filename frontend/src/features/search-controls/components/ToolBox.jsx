import React, { useState, useEffect, useId } from "react";
import SubmissionWorktree from "../../../features/submission/components/SubmissionWorktree";

const TOP_K_MIN = 1;
const TOP_K_MAX = 100;
const TOP_K_PRESETS = [10, 20, 50, 100];
const SUGGEST_COUNT_MIN = 1;
const SUGGEST_COUNT_MAX = 10;

/**
 * User-tunable search controls with direct numeric input and quick presets.
 */
const ToolBox = ({
  topK,
  setTopK,
  suggestCount,
  setSuggestCount,
  onReset,
}) => {
  const topKInputId = useId();
  const suggestCountInputId = useId();

  const [topKText, setTopKText] = useState(String(topK));
  const [suggestText, setSuggestText] = useState(
    suggestCount !== undefined ? String(suggestCount) : "3"
  );

  useEffect(() => {
    setTopKText(String(topK));
  }, [topK]);

  useEffect(() => {
    if (suggestCount !== undefined) {
      setSuggestText(String(suggestCount));
    }
  }, [suggestCount]);

  const commitTopKValue = (val) => {
    let num = parseInt(val, 10);
    if (isNaN(num)) {
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
    if (!isNaN(num) && num >= TOP_K_MIN && num <= TOP_K_MAX) {
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

  const commitSuggestValue = (val) => {
    if (!setSuggestCount) return;
    let num = parseInt(val, 10);
    if (isNaN(num)) {
      num = suggestCount || 3;
    }
    const clamped = Math.max(SUGGEST_COUNT_MIN, Math.min(SUGGEST_COUNT_MAX, num));
    setSuggestCount(clamped);
    setSuggestText(String(clamped));
  };

  const handleSuggestTextChange = (e) => {
    const val = e.target.value;
    setSuggestText(val);
    if (!setSuggestCount) return;
    const num = parseInt(val, 10);
    if (!isNaN(num) && num >= SUGGEST_COUNT_MIN && num <= SUGGEST_COUNT_MAX) {
      setSuggestCount(num);
    }
  };

  const handleSuggestKeyDown = (e) => {
    if (e.key === "Enter") {
      commitSuggestValue(suggestText);
      e.target.blur();
    }
  };

  const handleSuggestBlur = () => {
    commitSuggestValue(suggestText);
  };

  const stepSuggest = (delta) => {
    if (!setSuggestCount) return;
    const current = parseInt(suggestText, 10) || suggestCount || 3;
    const nextVal = Math.max(SUGGEST_COUNT_MIN, Math.min(SUGGEST_COUNT_MAX, current + delta));
    setSuggestCount(nextVal);
    setSuggestText(String(nextVal));
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

      {setSuggestCount && (
        <div className="toolbox-section">
          <div className="toolbox-label-row">
            <label htmlFor={suggestCountInputId} className="toolbox-label">
              Suggestion Count
            </label>
          </div>
          <div className="toolbox-direct-input-container">
            <div className="toolbox-stepper-row">
              <button
                type="button"
                className="toolbox-stepper-btn"
                onClick={() => stepSuggest(-1)}
                disabled={suggestCount <= SUGGEST_COUNT_MIN}
                aria-label="Decrease Suggestion Count"
              >
                −
              </button>
              <input
                id={suggestCountInputId}
                type="number"
                min={SUGGEST_COUNT_MIN}
                max={SUGGEST_COUNT_MAX}
                value={suggestText}
                onChange={handleSuggestTextChange}
                onKeyDown={handleSuggestKeyDown}
                onBlur={handleSuggestBlur}
                className="toolbox-number-input"
                placeholder="3"
                aria-label="Suggestion Count value"
              />
              <button
                type="button"
                className="toolbox-stepper-btn"
                onClick={() => stepSuggest(1)}
                disabled={suggestCount >= SUGGEST_COUNT_MAX}
                aria-label="Increase Suggestion Count"
              >
                +
              </button>
            </div>
          </div>
        </div>
      )}

      <button
        type="button"
        className="btn-utility toolbox-reset-btn"
        onClick={onReset}
        style={{ marginBottom: '12px' }}
      >
        Reset Parameters
      </button>

      <SubmissionWorktree />
    </aside>
  );
};

export default ToolBox;
