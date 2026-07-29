import React from "react";

// Contains the only user-tunable search control.
const ToolBox = ({ topK, setTopK, onReset }) => (
  <aside className="toolbox-sidebar">
    <div className="toolbox-section">
      <div className="toolbox-label-row">
        <label htmlFor="top-k-slider" className="toolbox-label">
          Top_K
        </label>
        <span className="toolbox-value">{topK}</span>
      </div>
      <input
        id="top-k-slider"
        type="range"
        min="5"
        max="50"
        step="5"
        value={topK}
        onChange={(event) => setTopK(Number(event.target.value))}
        className="toolbox-slider"
      />
    </div>
    <button className="btn-utility toolbox-reset-btn" onClick={onReset}>
      Reset Parameters
    </button>
  </aside>
);

export default ToolBox;
