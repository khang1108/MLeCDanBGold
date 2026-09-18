import React, { useEffect, useId, useState } from "react";
import { useVbsSession } from "../../vbs/contexts/VbsSessionContext";
import VbsTaskSelector from "../../vbs/components/VbsTaskSelector";
import ManualVideoOpener from "./ManualVideoOpener";
const TOP_K_MIN = 1;
const NOOP = () => {};

/**
 * Available competition datasets for VBS 2027.
 * `enabled` controls whether the option is selectable in the current build.
 * Disabled entries are shown with a "Coming soon" hint.
 */
const DATASETS = [
  { id: 'aic', label: 'AIC', description: 'AI Challenge – current dataset', enabled: true },
  { id: 'v3c', label: 'V3C', description: 'Vimeo Creative Commons Collection', enabled: false },
  { id: 'trecvid', label: 'TRECVID', description: 'NIST Ad-Hoc Video Search', enabled: false },
  { id: 'marine', label: 'MARINE', description: 'Marine / underwater video', enabled: false },
  { id: 'lapgynlhe', label: 'LapGynLHE', description: 'Laparoscopic gynecology', enabled: false },
];

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
  showRetrievalSources = true,
  evaluations: propEvaluations,
  selectedTask: propSelectedTask,
  setSelectedTask: propSetSelectedTask,
  connectedUserId: propConnectedUserId,
  gridSize = 'normal',
  setGridSize = NOOP,
  onOpenFrame,
}) => {
  const vbsSession = useVbsSession();
  const connectedUserId = propConnectedUserId ?? vbsSession.connectedUserId ?? '';
  const selectedTask = propSelectedTask ?? vbsSession.selectedTask ?? null;
  const setSelectedTask = propSetSelectedTask ?? vbsSession.setSelectedTask ?? NOOP;

  const topKInputId = useId();
  const datasetSelectId = useId();
  const [topKText, setTopKText] = useState(String(topK));
  const [activeDataset, setActiveDataset] = useState('aic');

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
    <div className="toolbox-stack">
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

      <div className="toolbox-section toolbox-grid-size-section">
        <div className="toolbox-label-row toolbox-grid-size-row">
          <span className="toolbox-label">Frame Size</span>
        </div>
        <div className="toolbox-segmented-group" role="group" aria-label="Frame size selection">
          <button
            type="button"
            className={`toolbox-segment-btn ${gridSize === 'compact' ? 'active' : ''}`}
            onClick={() => setGridSize?.('compact')}
            aria-pressed={gridSize === 'compact'}
            title="Compact frame cards (180px)"
          >
            Compact
          </button>
          <button
            type="button"
            className={`toolbox-segment-btn ${gridSize === 'normal' ? 'active' : ''}`}
            onClick={() => setGridSize?.('normal')}
            aria-pressed={gridSize === 'normal'}
            title="Normal frame cards (270px)"
          >
            Normal
          </button>
          <button
            type="button"
            className={`toolbox-segment-btn ${gridSize === 'large' ? 'active' : ''}`}
            onClick={() => setGridSize?.('large')}
            aria-pressed={gridSize === 'large'}
            title="Large frame cards (360px)"
          >
            Large
          </button>
        </div>
      </div>

      <div className="toolbox-section toolbox-dataset-section">
        <div className="toolbox-label-row toolbox-dataset-row">
          <label htmlFor={datasetSelectId} className="toolbox-label">
            Dataset
          </label>
          <select
            id={datasetSelectId}
            className="toolbox-dataset-select"
            value={activeDataset}
            onChange={(e) => setActiveDataset(e.target.value)}
            aria-label="Select dataset"
          >
            {DATASETS.map((ds) => (
              <option key={ds.id} value={ds.id} disabled={!ds.enabled}>
                {ds.label}{!ds.enabled ? ' — Coming soon' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {showRetrievalSources && (
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
      )}

      <div className="toolbox-section toolbox-task-section">
        <VbsTaskSelector
          connectedUserId={connectedUserId}
          evaluations={propEvaluations ?? vbsSession.evaluations}
          selectedTask={selectedTask}
          onRequestChange={setSelectedTask}
        />
      </div>

      <ManualVideoOpener onOpenFrame={onOpenFrame} />
      </aside>
    </div>
  );
};

export default ToolBox;
