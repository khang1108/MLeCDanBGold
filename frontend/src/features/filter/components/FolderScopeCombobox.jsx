import React, { useId, useMemo, useState } from 'react';

/** Compact scope picker with type-to-filter options. */
const FolderScopeCombobox = ({
  value,
  options,
  onChange,
  inputId = 'filter-folder-id',
  scopeLabel = 'folder',
  placeholder = 'folder_id',
}) => {
  const rawId = useId();
  const listboxId = `${rawId}-folder-options`.replace(/:/g, '');
  const [isOpen, setIsOpen] = useState(false);
  const [activeOptionIndex, setActiveOptionIndex] = useState(0);

  const filteredOptions = useMemo(() => {
    const query = value.trim().toLowerCase();
    return options.filter((option) => option.toLowerCase().includes(query));
  }, [options, value]);

  const selectOption = (option) => {
    onChange(option);
    setIsOpen(false);
    setActiveOptionIndex(0);
  };

  const handleInputChange = (event) => {
    onChange(event.target.value);
    setIsOpen(true);
    setActiveOptionIndex(0);
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Escape') {
      setIsOpen(false);
      return;
    }

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setIsOpen(true);
      setActiveOptionIndex((index) => Math.min(index + 1, Math.max(filteredOptions.length - 1, 0)));
      return;
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault();
      setIsOpen(true);
      setActiveOptionIndex((index) => Math.max(index - 1, 0));
      return;
    }

    if (event.key === 'Enter' && isOpen && filteredOptions[activeOptionIndex]) {
      event.preventDefault();
      selectOption(filteredOptions[activeOptionIndex]);
    }
  };

  return (
    <div
      className="filter-scope-combobox"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setIsOpen(false);
      }}
    >
      <div className="filter-scope-input-wrap">
        <input
          id={inputId}
          className="input-text filter-scope-input"
          type="text"
          role="combobox"
          value={value}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          autoComplete="off"
          aria-autocomplete="list"
          aria-expanded={isOpen}
          aria-controls={listboxId}
          aria-activedescendant={isOpen && filteredOptions[activeOptionIndex]
            ? `${listboxId}-${activeOptionIndex}`
            : undefined}
        />
        <button
          type="button"
          className={`filter-scope-toggle${isOpen ? ' open' : ''}`}
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => setIsOpen((open) => !open)}
          tabIndex={-1}
          aria-label={`Toggle ${scopeLabel} options`}
          aria-expanded={isOpen}
        >
          ▾
        </button>
      </div>

      {isOpen && (
        <div className="filter-scope-options" id={listboxId} role="listbox">
          {filteredOptions.length > 0 ? filteredOptions.map((option, index) => (
            <button
              type="button"
              id={`${listboxId}-${index}`}
              className={`filter-scope-option${index === activeOptionIndex ? ' active' : ''}`}
              key={option}
              role="option"
              aria-selected={option === value}
              onMouseDown={(event) => event.preventDefault()}
              onMouseEnter={() => setActiveOptionIndex(index)}
              onClick={() => selectOption(option)}
            >
              {option}
            </button>
          )) : (
            <div className="filter-scope-empty" role="status">
              No {scopeLabel} options found
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FolderScopeCombobox;
