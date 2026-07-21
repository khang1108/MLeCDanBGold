import React from 'react';

const ToolBox = ({
  topK,
  setTopK,
  temperature,
  setTemperature,
  filter,
  setFilter,
  searchMode,
  setSearchMode,
  onReset
}) => {
  return (
    <aside className="toolbox-sidebar">
      {/* Top_K Slider Section */}
      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor="top-k-slider" className="toolbox-label">Top_K</label>
          <span className="toolbox-value">{topK}</span>
        </div>
        <input
          id="top-k-slider"
          type="range"
          min="5"
          max="50"
          step="5"
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="toolbox-slider"
        />
      </div>

      {/* Temperature Slider Section */}
      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor="temp-slider" className="toolbox-label">Temperature</label>
          <span className="toolbox-value">{temperature.toFixed(1)}</span>
        </div>
        <input
          id="temp-slider"
          type="range"
          min="0.0"
          max="1.0"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
          className="toolbox-slider"
        />
      </div>

      {/* Filter Text Input Section */}
      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor="filter-input" className="toolbox-label">Filter</label>
        </div>
        <input
          id="filter-input"
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="e.g. keyframe_type:wide"
          className="toolbox-input"
        />
      </div>

      {/* Search Mode Select Section */}
      <div className="toolbox-section">
        <div className="toolbox-label-row">
          <label htmlFor="search-mode-select" className="toolbox-label">Search Mode</label>
        </div>
        <select
          id="search-mode-select"
          value={searchMode}
          onChange={(e) => setSearchMode(e.target.value)}
          className="toolbox-input"
        >
          <option value="accurate">Accurate (Default)</option>
          <option value="fast">Fast (Finalist)</option>
        </select>
      </div>

      {/* Reset Parameters Button */}
      <button className="btn-utility toolbox-reset-btn" onClick={onReset}>
        Reset Parameters
      </button>
    </aside>
  );
};

export default ToolBox;
