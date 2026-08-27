import React, { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Search existing submission files or edit one file's CSV content.
 *
 * Picker Enter chooses the highlighted file. Editor Enter saves and closes;
 * Shift+Enter remains available for inserting a newline.
 */
const SubmissionFileModal = ({
  mode,
  files,
  editorFile,
  pendingLine,
  onSelectFile,
  onSave,
  onClose,
}) => {
  const searchRef = useRef(null);
  const editorRef = useRef(null);
  const fileOptionRefs = useRef(new Map());
  const [searchText, setSearchText] = useState('');
  const [highlightedFileId, setHighlightedFileId] = useState(null);
  const [draft, setDraft] = useState('');

  const filteredFiles = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    if (!query) return files;
    return files.filter((file) => file.name.toLowerCase().includes(query));
  }, [files, searchText]);

  useEffect(() => {
    if (mode === 'picker') {
      setSearchText('');
      setHighlightedFileId(files[0]?.id || null);
      window.setTimeout(() => searchRef.current?.focus(), 0);
    }
  }, [mode, files]);

  useEffect(() => {
    if (mode === 'editor') {
      setDraft(editorFile?.content || '');
      window.setTimeout(() => editorRef.current?.focus(), 0);
    }
  }, [mode, editorFile]);

  useEffect(() => {
    if (filteredFiles.length === 0) {
      setHighlightedFileId(null);
    } else if (!filteredFiles.some((file) => file.id === highlightedFileId)) {
      setHighlightedFileId(filteredFiles[0].id);
    }
  }, [filteredFiles, highlightedFileId]);

  useEffect(() => {
    if (!mode) return undefined;

    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        // This modal can be opened above the frame inspector. Capture Escape
        // before the inspector's window listener so only this layer closes.
        event.stopPropagation();
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape, true);
    return () => window.removeEventListener('keydown', handleEscape, true);
  }, [mode, onClose]);

  useEffect(() => {
    if (!highlightedFileId) return;
    fileOptionRefs.current.get(highlightedFileId)?.scrollIntoView?.({
      block: 'nearest',
    });
  }, [highlightedFileId]);

  if (!mode) return null;

  const chooseHighlightedFile = () => {
    const file = filteredFiles.find((item) => item.id === highlightedFileId)
      || filteredFiles[0];
    if (file) onSelectFile(file.id);
  };

  const handlePickerKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      chooseHighlightedFile();
      return;
    }
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      if (filteredFiles.length === 0) return;
      const currentIndex = Math.max(
        filteredFiles.findIndex((file) => file.id === highlightedFileId),
        0,
      );
      const offset = event.key === 'ArrowDown' ? 1 : -1;
      const nextIndex = (currentIndex + offset + filteredFiles.length) % filteredFiles.length;
      setHighlightedFileId(filteredFiles[nextIndex].id);
    }
  };

  const handleEditorKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSave(draft);
    }
  };

  return (
    <div className="submission-modal-overlay" role="presentation" onClick={onClose}>
      <section
        className="submission-file-modal"
        role="dialog"
        aria-modal="true"
        aria-label={mode === 'picker'
          ? 'Choose a CSV file'
          : 'Edit ' + (editorFile?.name || 'CSV file')}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="submission-modal-header">
          <div>
            <p className="submission-modal-kicker">Submission workspace</p>
            <h2>{mode === 'picker' ? 'Choose a CSV file' : editorFile?.name}</h2>
          </div>
          <button type="button" className="submission-modal-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>

        {mode === 'picker' ? (
          <>
            <div className="submission-modal-body">
              {pendingLine && (
                <div className="submission-pending-row">
                  <span>Row to add</span>
                  <code>{pendingLine}</code>
                </div>
              )}
              <input
                ref={searchRef}
                className="submission-file-search"
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
                onKeyDown={handlePickerKeyDown}
                placeholder="Search CSV file..."
                aria-label="Search submission files"
              />
              <ul className="submission-file-picker-list">
                {filteredFiles.length > 0 ? filteredFiles.map((file) => (
                  <li key={file.id}>
                    <button
                      type="button"
                      className={'submission-file-option ' + (file.id === highlightedFileId ? 'highlighted' : '')}
                      ref={(node) => {
                        if (node) fileOptionRefs.current.set(file.id, node);
                        else fileOptionRefs.current.delete(file.id);
                      }}
                      onClick={() => onSelectFile(file.id)}
                      onMouseEnter={() => setHighlightedFileId(file.id)}
                    >
                      <span className="submission-file-option-name">{file.name}</span>
                      <span className="submission-file-option-meta">
                        {file.content?.trim() ? 'has entries' : 'empty'}
                      </span>
                    </button>
                  </li>
                )) : (
                  <li className="submission-file-picker-empty">No matching CSV files.</li>
                )}
              </ul>
            </div>
            <footer className="submission-modal-footer">
              <span>Use ↑/↓ to highlight, Enter to choose</span>
              <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
            </footer>
          </>
        ) : (
          <>
            <div className="submission-modal-body submission-editor-body">
              <textarea
                ref={editorRef}
                className="submission-file-editor"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleEditorKeyDown}
                spellCheck="false"
                aria-label={'Edit ' + (editorFile?.name || 'CSV file') + ' content'}
              />
            </div>
            <footer className="submission-modal-footer">
              <span>Enter to save · Shift+Enter for a new line</span>
              <button type="button" className="btn-secondary" onClick={onClose}>Cancel</button>
            </footer>
          </>
        )}
      </section>
    </div>
  );
};

export default SubmissionFileModal;
