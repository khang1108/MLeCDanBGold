import React, { useId, useLayoutEffect, useRef } from 'react';

const TEXT_FIELDS = [
  { key: 'title', label: 'Title', placeholder: 'title' },
  { key: 'asr', label: 'ASR / Transcript', placeholder: 'asr' },
  { key: 'caption', label: 'Caption', placeholder: 'caption' },
  { key: 'ocr', label: 'OCR', placeholder: 'ocr' },
];

const resizeTextArea = (textArea) => {
  textArea.style.height = '0px';
  textArea.style.height = `${Math.max(textArea.scrollHeight, 31)}px`;
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
        {TEXT_FIELDS.map(({ key, label, placeholder }) => {
          const inputId = `${formId}-${key}`;
          return (
            <label className="filter-field" htmlFor={inputId} key={key}>
              <span className="sr-only">{label}</span>
              <textarea
                id={inputId}
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
            </label>
          );
        })}
      </div>

      <section className="filter-objects-section" aria-label="Object filters">
        <div className="filter-form-actions">
          <button type="button" className="btn-secondary filter-reset-btn" onClick={onReset}>
            Clear
          </button>
          <button type="submit" className="btn-secondary filter-submit-btn" disabled={isLoading}>
            {isLoading ? 'Filtering…' : 'Filter'}
          </button>
        </div>
        <div className="filter-object-list">
          {values.objects.map((row, index) => {
            const value = row.value ?? '';
            return (
              <div className="filter-object-row" key={row.id}>
                <label className="filter-object-name-field">
                  <span className="sr-only">Object {index + 1} name</span>
                  <input
                    className="input-text filter-object-input"
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
            +
          </button>
        </div>
      </section>
    </form>
  );
};

export default FilterForm;
