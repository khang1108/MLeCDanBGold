import React, { useId, useMemo } from 'react';

/**
 * Reusable DRES evaluation task selector dropdown.
 * Flattens evaluation templates and calls onRequestChange with the selected task object.
 */
const VbsTaskSelector = ({
  connectedUserId = '',
  evaluations = [],
  selectedTask = null,
  onRequestChange,
  disabled = false,
  id: propId,
  label = 'Task',
  showLabel = true,
  className = 'toolbox-label-row toolbox-task-row',
}) => {
  const generatedId = useId();
  const selectId = propId || generatedId;

  const availableTasks = useMemo(() => {
    const list = [];
    if (Array.isArray(evaluations)) {
      evaluations.forEach((ev) => {
        if (Array.isArray(ev?.taskTemplates)) {
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
  }, [evaluations]);

  const selectedKey = selectedTask
    ? `${selectedTask.evaluationId}:${selectedTask.taskName}`
    : '';

  const handleTaskChange = (e) => {
    const val = e.target.value;
    if (!val) {
      onRequestChange?.(null);
      return;
    }
    const match = availableTasks.find((t) => `${t.evaluationId}:${t.taskName}` === val);
    if (match) {
      onRequestChange?.(match);
    }
  };

  const isDisabled = disabled || !connectedUserId || availableTasks.length === 0;

  return (
    <div className={`vbs-task-selector ${className}`.trim()}>
      {showLabel && (
        <label htmlFor={selectId} className="toolbox-label">
          {label}
        </label>
      )}
      <select
        id={selectId}
        className="toolbox-task-select"
        value={selectedKey}
        onChange={handleTaskChange}
        disabled={isDisabled}
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
  );
};

export default VbsTaskSelector;
