/** Presentational picker/editor for the global shared submission dialog. */
import React, { useEffect, useMemo, useRef, useState } from 'react';

const SubmissionFileModal = ({
  mode,
  files = [],
  editorFile,
  pendingLine,
  draft: controlledDraft,
  onDraftChange,
  onSelectFile,
  onSave,
  onValidate,
  onDelete,
  onClose,
  isMutating = false,
  remoteConflict,
  onLoadConflict,
  onRebaseConflict,
  error,
  historyPatchError,
  onRetryHistoryPatch,
}) => {
  const searchRef = useRef(null);
  const editorRef = useRef(null);
  const fileOptionRefs = useRef(new Map());
  const [searchText, setSearchText] = useState('');
  const [highlightedFileName, setHighlightedFileName] = useState(null);
  const [localDraft, setLocalDraft] = useState('');
  const draft = controlledDraft === undefined ? localDraft : controlledDraft;

  const filteredFiles = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    return query ? files.filter((file) => file.name.toLowerCase().includes(query)) : files;
  }, [files, searchText]);

  useEffect(() => {
    if (mode !== 'picker') return;
    setSearchText('');
    setHighlightedFileName(files[0]?.name || null);
    window.setTimeout(() => searchRef.current?.focus(), 0);
  }, [files, mode]);

  useEffect(() => {
    if (mode !== 'editor') return;
    setLocalDraft(editorFile?.content || '');
    window.setTimeout(() => editorRef.current?.focus(), 0);
  }, [editorFile, mode]);

  useEffect(() => {
    if (filteredFiles.length === 0) setHighlightedFileName(null);
    else if (!filteredFiles.some((file) => file.name === highlightedFileName)) {
      setHighlightedFileName(filteredFiles[0].name);
    }
  }, [filteredFiles, highlightedFileName]);

  useEffect(() => {
    if (!mode) return undefined;
    const handleEscape = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      if (!isMutating) onClose?.();
    };
    window.addEventListener('keydown', handleEscape, true);
    return () => window.removeEventListener('keydown', handleEscape, true);
  }, [isMutating, mode, onClose]);

  useEffect(() => {
    if (!highlightedFileName) return;
    fileOptionRefs.current.get(highlightedFileName)?.scrollIntoView?.({ block: 'nearest' });
  }, [highlightedFileName]);

  if (!mode) return null;

  const chooseHighlightedFile = () => {
    const file = filteredFiles.find((item) => item.name === highlightedFileName) || filteredFiles[0];
    if (file) onSelectFile?.(file.name);
  };

  const handlePickerKeyDown = (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      chooseHighlightedFile();
      return;
    }
    if (!['ArrowDown', 'ArrowUp'].includes(event.key) || filteredFiles.length === 0) return;
    event.preventDefault();
    const currentIndex = Math.max(
      filteredFiles.findIndex((file) => file.name === highlightedFileName),
      0,
    );
    const offset = event.key === 'ArrowDown' ? 1 : -1;
    setHighlightedFileName(filteredFiles[
      (currentIndex + offset + filteredFiles.length) % filteredFiles.length
    ].name);
  };

  const handleEditorKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!isMutating) onSave?.();
    }
  };

  const setDraft = (value) => {
    if (controlledDraft === undefined) setLocalDraft(value);
    onDraftChange?.(value);
  };
  const close = () => {
    if (!isMutating) onClose?.();
  };

  return (
    <div className="submission-modal-overlay" role="presentation" onClick={close}>
      <section
        className="submission-file-modal"
        role="dialog"
        aria-modal="true"
        aria-label={mode === 'picker' ? 'Choose a CSV file' : `Edit ${editorFile?.name || 'CSV file'}`}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="submission-modal-header">
          <div>
            <p className="submission-modal-kicker">Submission workspace</p>
            <h2>{mode === 'picker' ? 'Choose a CSV file' : editorFile?.name}</h2>
          </div>
          <button type="button" className="submission-modal-close" onClick={close} disabled={isMutating} aria-label="Close">
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
                  <li key={file.name}>
                    <button
                      type="button"
                      className={`submission-file-option ${file.name === highlightedFileName ? 'highlighted' : ''}`}
                      ref={(node) => {
                        if (node) fileOptionRefs.current.set(file.name, node);
                        else fileOptionRefs.current.delete(file.name);
                      }}
                      onClick={() => onSelectFile?.(file.name)}
                      onMouseEnter={() => setHighlightedFileName(file.name)}
                    >
                      <span className="submission-file-option-name">{file.name}</span>
                    </button>
                  </li>
                )) : (
                  <li className="submission-file-picker-empty">No matching CSV files.</li>
                )}
              </ul>
            </div>
            <footer className="submission-modal-footer">
              <span>Use ↑/↓ to highlight, Enter to choose</span>
              <button type="button" className="btn-secondary" onClick={close}>Cancel</button>
            </footer>
          </>
        ) : (
          <>
            <div className="submission-modal-body submission-editor-body">
              {error && <div className="submission-status-banner error" role="alert">{error}</div>}
              {historyPatchError && (
                <div className="submission-status-banner error" role="alert">
                  History state was not recorded.{' '}
                  <button type="button" className="btn-link" onClick={onRetryHistoryPatch} disabled={isMutating}>
                    Retry history update
                  </button>
                </div>
              )}
              {remoteConflict && (
                <div className="submission-conflict" role="alert">
                  <strong>This file changed on the server.</strong>
                  <pre>{remoteConflict.content}</pre>
                  <div className="submission-conflict-actions">
                    <button type="button" className="btn-secondary" onClick={onLoadConflict} disabled={isMutating}>
                      Load server copy
                    </button>
                    <button type="button" className="btn-primary" onClick={onRebaseConflict} disabled={isMutating}>
                      Keep draft and rebase
                    </button>
                  </div>
                </div>
              )}
              <textarea
                ref={editorRef}
                className="submission-file-editor"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={handleEditorKeyDown}
                spellCheck="false"
                aria-label={`Edit ${editorFile?.name || 'CSV file'} content`}
                disabled={isMutating}
              />
            </div>
            <footer className="submission-modal-footer submission-editor-footer">
              <span>Enter to save · Shift+Enter for a new line</span>
              <div className="submission-editor-actions">
                <button type="button" className="btn-secondary" onClick={onDelete} disabled={isMutating}>
                  Delete
                </button>
                <button type="button" className="btn-secondary" onClick={onValidate} disabled={isMutating || !draft.trim() || editorFile?.is_validated}>
                  Validate
                </button>
                <button type="button" className="btn-primary" onClick={onSave} disabled={isMutating}>
                  Lưu
                </button>
                <button type="button" className="btn-secondary" onClick={close} disabled={isMutating}>Cancel</button>
              </div>
            </footer>
          </>
        )}
      </section>
    </div>
  );
};

export default SubmissionFileModal;
