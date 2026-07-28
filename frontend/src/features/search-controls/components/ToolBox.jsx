import React from 'react';

// Contains only published search controls: Top K and profile mode.
const ToolBox = ({ topK, setTopK, searchMode, setSearchMode, onReset }) => <aside className="toolbox-sidebar">
  <div className="toolbox-section"><div className="toolbox-label-row"><label htmlFor="top-k-slider" className="toolbox-label">Top_K</label><span className="toolbox-value">{topK}</span></div><input id="top-k-slider" type="range" min="5" max="50" step="5" value={topK} onChange={(event) => setTopK(Number(event.target.value))} className="toolbox-slider" /></div>
  <div className="toolbox-section"><div className="toolbox-label-row"><label htmlFor="search-mode-select" className="toolbox-label">Search Mode</label></div><select id="search-mode-select" value={searchMode} onChange={(event) => setSearchMode(event.target.value)} className="toolbox-input"><option value="accurate">Accurate (Default)</option><option value="fast">Fast (Finalist)</option></select></div>
  <button className="btn-utility toolbox-reset-btn" onClick={onReset}>Reset Parameters</button>
</aside>;

export default ToolBox;