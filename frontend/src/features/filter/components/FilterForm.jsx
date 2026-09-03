import React, { useId, useLayoutEffect, useRef } from 'react';

const TEXT_FIELDS = [
  { key: 'title', label: 'Title', icon: '📝', placeholder: 'Video title keywords…' },
  { key: 'caption', label: 'Caption', icon: '💬', placeholder: 'Visual caption keywords…' },
  { key: 'ocr', label: 'OCR', icon: '🔤', placeholder: 'On-screen text keywords…' },
  { key: 'asr', label: 'ASR / Transcript', icon: '🎙️', placeholder: 'Spoken transcript keywords…' },
];

const resizeTextArea = (textArea) => {
  textArea.style.height = '0px';
  textArea.style.height = `${Math.max(textArea.scrollHeight, 36)}px`;
};

/** Compact metadata-only filter form. Scope controls live beside the result view. */
const FilterForm = ({ values, onChange, onSubmit, onReset, isLoading }) => {
  const formId = useId();
  const textAreaRefs = useRef({});

  useLayoutEffect(() => {
    Object.values(textAreaRefs.current).forEach(resizeTextArea);
  }, [values]);

  const updateText = (field, event) => {
    onChange({ ...values, [field]: event.target.value });
  };

  const updateObject = (rowId, event) => {
    onChange({
      ...values,
      objects: values.objects.map((row) => (
        row.id === rowId ? { ...row, value: event.target.value } : row
      )),
    });
  };

  const addObject = () => {
    const usedIds = new Set(values.objects.map((row) => row.id));
    let nextIndex = values.objects.length + 1;
    while (usedIds.has(`${formId}-object-${nextIndex}`)) {
      nextIndex += 1;
    }
    onChange({
      ...values,
      objects: [...values.objects, { id: `${formId}-object-${nextIndex}`, value: '' }],
    });
  };

  const removeObject = (rowId) => {
    onChange({
      ...values,
      objects: values.objects.filter((row) => row.id !== rowId),
    });
  };

  return (
    <form className="filter-form" onSubmit={onSubmit}>
      <div className="filter-text-grid">
        {TEXT_FIELDS.map(({ key, label, icon, placeholder }) => {
          const inputId = `${formId}-${key}`;
          return (
            <div className="filter-field-col" key={key}>
              <label className="filter-field-label" htmlFor={inputId}>
                <span className="filter-field-icon" aria-hidden="true">{icon}</span>
                {label}
              </label>
              <textarea
                id={inputId}
                aria-label={label}
                className="input-text filter-input"
                ref={(node) => {
                  if (node) textAreaRefs.current[key] = node;
                  else delete textAreaRefs.current[key];
                }}
                rows="1"
                value={values[key]}
                onChange={(event) => updateText(key, event)}
                onInput={(event) => resizeTextArea(event.currentTarget)}
                placeholder={placeholder}
              />
            </div>
          );
        })}
      </div>

      <section className="filter-objects-section" aria-label="Object filters">
        <div className="filter-objects-group">
          <span className="filter-objects-title">
            <span className="filter-field-icon" aria-hidden="true">📦</span>
            Detected Objects
          </span>
          <div className="filter-object-list">
            {values.objects.map((row, index) => {
              const value = row.value ?? '';
              return (
                <div className="filter-object-row" key={row.id}>
                  <label className="filter-object-name-field">
                    <span className="sr-only">Object {index + 1} name</span>
                    <input
                      className="filter-object-input"
                      type="text"
                      value={value}
                      onChange={(event) => updateObject(row.id, event)}
                      placeholder="name: count"
                      aria-label={`Object ${index + 1}, format name colon count`}
                      style={{ width: `${Math.max(9, value.length + 2)}ch` }}
                    />
                  </label>
                  <button
                    type="button"
                    className="filter-remove-object-btn"
                    onClick={() => removeObject(row.id)}
                    aria-label={`Remove object ${index + 1}`}
                    title="Remove object"
                  >
                    ×
                  </button>
                </div>
              );
            })}
            <button
              type="button"
              className="filter-add-object-btn"
              onClick={addObject}
              aria-label="Add object filter"
              title="Add object filter"
            >
              + Add object
            </button>
          </div>
        </div>

        <div className="filter-form-actions">
          <button type="button" className="btn-secondary filter-reset-btn" onClick={onReset}>
            Clear
          </button>
          <button type="submit" className="btn-primary filter-submit-btn" disabled={isLoading}>
            {isLoading ? 'Filtering…' : 'Filter'}
          </button>
        </div>
      </section>
    </form>
  );
};

export default FilterForm;
