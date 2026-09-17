import React, { useEffect, useId, useMemo, useState } from "react";
import { useVbsSession } from "../../vbs/contexts/VbsSessionContext";
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
}) => {
  const vbsSession = useVbsSession();
  const connectedUserId = propConnectedUserId ?? vbsSession.connectedUserId ?? '';
  const selectedTask = propSelectedTask ?? vbsSession.selectedTask ?? null;
  const setSelectedTask = propSetSelectedTask ?? vbsSession.setSelectedTask ?? NOOP;

  const topKInputId = useId();
  const datasetSelectId = useId();
  const taskSelectId = useId();
  const [topKText, setTopKText] = useState(String(topK));
  const [activeDataset, setActiveDataset] = useState('aic');

  const availableTasks = useMemo(() => {
    const list = [];
    const evaluations = propEvaluations ?? vbsSession.evaluations;
    if (Array.isArray(evaluations)) {
      evaluations.forEach((ev) => {
        if (Array.isArray(ev.taskTemplates)) {
          ev.taskTemplates.forEach((t) => {
            list.push({
              evaluationId: ev.id,
              evaluationName: ev.name,
              taskName: t.name,
              taskGroup: t.taskGroup,
              taskType: t.taskType,
              duration: t.duration,
            });
          });
        }
      });
    }
    return list;
  }, [propEvaluations, vbsSession.evaluations]);

  const selectedKey = selectedTask
    ? `${selectedTask.evaluationId}:${selectedTask.taskName}`
    : '';

  const handleTaskChange = (e) => {
    const val = e.target.value;
    const match = availableTasks.find((t) => `${t.evaluationId}:${t.taskName}` === val);
    if (match) {
      setSelectedTask(match);
    }
  };

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
        <div className="toolbox-label-row toolbox-task-row">
          <label htmlFor={taskSelectId} className="toolbox-label">
            Task
          </label>
          <select
            id={taskSelectId}
            className="toolbox-task-select"
            value={selectedKey}
            onChange={handleTaskChange}
            disabled={!connectedUserId || availableTasks.length === 0}
            aria-label="Select evaluation task"
          >
            {!connectedUserId ? (
              <option value="">Connect VBS to load tasks</option>
            ) : availableTasks.length === 0 ? (
              <option value="">No tasks available</option>
            ) : (
              availableTasks.map((t) => (
                <option
                  key={`${t.evaluationId}:${t.taskName}`}
                  value={`${t.evaluationId}:${t.taskName}`}
                >
                  {t.taskName}
                </option>
              ))
            )}
          </select>
        </div>
      </div>
      </aside>
    </div>
  );
};

export default ToolBox;
