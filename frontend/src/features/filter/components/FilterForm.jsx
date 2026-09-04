import React, { useId, useLayoutEffect, useRef } from 'react';


const TEXT_FIELDS = [
  { key: 'title', label: 'Title', placeholder: 'Video title keywords…' },
  { key: 'asr', label: 'ASR / Transcript', placeholder: 'Spoken transcript keywords…' },
  { key: 'caption', label: 'Caption', placeholder: 'Visual caption keywords…' },
  { key: 'ocr', label: 'OCR', placeholder: 'On-screen text keywords…' },
];


const resizeTextArea = (textArea) => {
  textArea.style.height = '0px';
  textArea.style.height = `${Math.max(textArea.scrollHeight, 36)}px`;
};


/** Render source-specific literal inputs and repeatable object thresholds. */
const FilterForm = ({ values, onChange, onSubmit, onReset, isLoading }) => {
  const formId = useId();
  const textAreaRefs = useRef({});

  useLayoutEffect(() => {
    Object.values(textAreaRefs.current).forEach(resizeTextArea);
  }, [values]);

  const updateText = (field, value) => {
    onChange({ ...values, [field]: value });
  };

  const updateObject = (rowId, value) => {
    onChange({
      ...values,
      objects: values.objects.map((row) => (
        row.id === rowId ? { ...row, value } : row
      )),
    });
  };

  const addObject = () => {
    const nextNumber = values.objects.length + 1;
    onChange({
      ...values,
      objects: [...values.objects, { id: `${formId}-object-${nextNumber}`, value: '' }],
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
        {TEXT_FIELDS.map(({ key, label, placeholder }) => {
          const inputId = `${formId}-${key}`;
          return (
            <div className="filter-field-col" key={key}>
              <label className="filter-field-label" htmlFor={inputId}>{label}</label>
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
                onChange={(event) => updateText(key, event.target.value)}
                onInput={(event) => resizeTextArea(event.currentTarget)}
                placeholder={placeholder}
              />
            </div>
          );
        })}
      </div>

      <section className="filter-objects-section" aria-label="Object filters">
        <div className="filter-objects-group">
          <span className="filter-objects-title">Detected Objects</span>
          <div className="filter-object-list">
            {values.objects.map((row, index) => {
              const value = row.value || '';
              return (
                <div className="filter-object-row" key={row.id}>
                  <label className="filter-object-name-field">
                    <span className="sr-only">Object {index + 1} minimum count</span>
                    <input
                      className="filter-object-input"
                      type="text"
                      value={value}
                      onChange={(event) => updateObject(row.id, event.target.value)}
                      placeholder="name: count"
                      aria-label={`Object ${index + 1}, format name colon count`}
                      style={{ width: `${Math.max(10, value.length + 2)}ch` }}
                    />
                  </label>
                  <button
                    type="button"
                    className="filter-remove-object-btn"
                    onClick={() => removeObject(row.id)}
                    aria-label={`Remove object ${index + 1}`}
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
