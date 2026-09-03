import React, { useEffect, useId, useState } from "react";
import SubmissionWorktree from "../../../features/submission/components/SubmissionWorktree";

const TOP_K_MIN = 1;
const NOOP = () => {};

/**
 * User-tunable search controls with direct numeric input and quick presets.
 */
const ToolBox = ({
  topK,
  setTopK,
  useDense = true,
  setUseDense = NOOP,
  useBm25 = true,
  setUseBm25 = NOOP,
  includeSubmissionWorktree = true,
}) => {
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
    const normalized = Math.max(TOP_K_MIN, num);
    setTopK(normalized);
    setTopKText(String(normalized));
  };

  const handleTopKTextChange = (e) => {
    const val = e.target.value;
    setTopKText(val);
    const num = parseInt(val, 10);
    if (!Number.isNaN(num) && num >= TOP_K_MIN) {
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

  return (
    <aside className="toolbox-sidebar">
      <div className="toolbox-section">
        <div className="toolbox-label-row toolbox-top-k-row">
          <label htmlFor={topKInputId} className="toolbox-label">
            Top-K results
          </label>
          <input
            id={topKInputId}
            type="number"
            min={TOP_K_MIN}
            value={topKText}
            onChange={handleTopKTextChange}
            onKeyDown={handleTopKKeyDown}
            onBlur={handleTopKBlur}
            className="toolbox-number-input toolbox-top-k-input"
            placeholder="20"
            aria-label="Top-K value"
          />
        </div>
      </div>

      <fieldset className="toolbox-section toolbox-retrieval-section">
        <legend className="toolbox-label">Retrieval sources</legend>
        <div className="toolbox-toggle-list">
          <label className="toolbox-toggle-row">
            <span className="toolbox-toggle-copy">
              <span className="toolbox-toggle-name">Dense</span>
              <span className="toolbox-toggle-description">Semantic matching</span>
            </span>
            <input
              type="checkbox"
              role="switch"
              className="toolbox-switch"
              checked={useDense}
              onChange={(event) => setUseDense(event.target.checked)}
              disabled={useDense && !useBm25}
              aria-label="Use Dense retrieval"
            />
          </label>

          <label className="toolbox-toggle-row">
            <span className="toolbox-toggle-copy">
              <span className="toolbox-toggle-name">BM25</span>
              <span className="toolbox-toggle-description">Lexical matching</span>
            </span>
            <input
              type="checkbox"
              role="switch"
              className="toolbox-switch"
              checked={useBm25}
              onChange={(event) => setUseBm25(event.target.checked)}
              disabled={useBm25 && !useDense}
              aria-label="Use BM25 retrieval"
            />
          </label>
        </div>
      </fieldset>

      {includeSubmissionWorktree && <SubmissionWorktree />}
    </aside>
  );
};

export default ToolBox;
