import React, { useEffect, useState } from 'react';
import { useSubmission } from '../contexts/SubmissionContext';
import SubmissionFileModal from './SubmissionFileModal';
import { downloadCsvArchive, getNonEmptyCsvFiles } from '../submissionArchive';

/**
 * Manage locally created submission CSV files without making the focused tree
 * item a hidden submission destination.
 */
const SubmissionWorktree = ({
  submissionRequest: externalSubmissionRequest,
  onSubmissionRequestHandled,
}) => {
  const {
    files,
    setFiles,
    removeFile,
    clearAllFiles,
    updateFileContent,
    submissionRequest,
    clearSubmissionRequest,
  } = useSubmission();
  const activeSubmissionRequest = externalSubmissionRequest || submissionRequest;

  const [status, setStatus] = useState(null);
  const [modalMode, setModalMode] = useState(null);
  const [pendingSubmission, setPendingSubmission] = useState(null);
  const [editorFileId, setEditorFileId] = useState(null);

  useEffect(() => {
    if (!activeSubmissionRequest) return;
    setPendingSubmission(activeSubmissionRequest);
    setModalMode('picker');
    setStatus(null);
    if (externalSubmissionRequest) {
      onSubmissionRequestHandled?.();
    } else {
      clearSubmissionRequest();
    }
  }, [
    activeSubmissionRequest,
    clearSubmissionRequest,
    externalSubmissionRequest,
    onSubmissionRequestHandled,
  ]);

  const handleNewCsv = () => {
    const existingNames = new Set(files.map((file) => file.name));
    let suffix = 1;
    let name = 'submission.csv';

    while (existingNames.has(name)) {
      suffix += 1;
      name = `submission-${suffix}.csv`;
    }

    const newFile = { id: name, name, content: '' };
    setFiles([...files, newFile]);
  };

  const handleSelectFile = (fileId) => {
    const file = files.find((item) => item.id === fileId);
    if (!file) return;

    if (pendingSubmission?.line) {
      const nextContent = file.content?.trim()
        ? file.content.trimEnd() + '\n' + pendingSubmission.line
        : pendingSubmission.line;
      updateFileContent(fileId, nextContent);
    }

    setPendingSubmission(null);
    setEditorFileId(fileId);
    setModalMode('editor');
  };

  const handleOpenEditor = (fileId) => {
    if (files.some((file) => file.id === fileId)) {
      setPendingSubmission(null);
      setEditorFileId(fileId);
      setModalMode('editor');
    }
  };

  const handleSaveEditor = (content) => {
    if (editorFileId) updateFileContent(editorFileId, content);
    setEditorFileId(null);
    setModalMode(null);
  };

  const handleCloseModal = () => {
    setPendingSubmission(null);
    setEditorFileId(null);
    setModalMode(null);
  };

  const handleDownloadAll = () => {
    const nonEmptyFiles = getNonEmptyCsvFiles(files);
    if (nonEmptyFiles.length === 0) {
      setStatus({ type: 'error', message: 'No CSV file contains submission rows yet.' });
      return;
    }

    downloadCsvArchive(nonEmptyFiles);
    setStatus({
      type: 'success',
      message: 'Downloaded ' + nonEmptyFiles.length
        + ' non-empty CSV file' + (nonEmptyFiles.length === 1 ? '' : 's')
        + ' as submissions.zip.',
    });
  };

  const countLines = (content) => {
    if (!content?.trim()) return 0;
    return content.trim().split(/\r?\n/).filter(Boolean).length;
  };

  const editorFile = files.find((file) => file.id === editorFileId) || null;
  const nonEmptyCount = getNonEmptyCsvFiles(files).length;

  return (
    <div className="toolbox-section submission-worktree">
      <div className="toolbox-label-row submission-header-row">
        <label className="toolbox-label">Submission Files</label>
        <div className="submission-actions-group">
          <button
            type="button"
            className="btn-utility"
            onClick={handleNewCsv}
          >
            New CSV
          </button>
          {files.length > 0 && (
            <>
            <span className="submission-count-badge">{files.length}</span>
            <button
              type="button"
              className="submission-icon-btn"
              onClick={clearAllFiles}
              title="Clear all files"
              aria-label="Clear all files"
            >
              ✕
            </button>
            </>
          )}
        </div>
      </div>

      {files.length === 0 ? (
        <div className="submission-upload-card">
          <div className="submission-upload-icon">📄</div>
          <p className="submission-upload-title">No Submission Files</p>
          <p className="submission-upload-desc">
            Create a CSV file to start collecting submission rows.
          </p>
        </div>
      ) : (
        <div className="submission-tree-container">
          <div className="submission-tree-root">
            <span className="tree-icon">📂</span>
            <span className="tree-root-label">submissions/</span>
          </div>
          <ul className="submission-tree-list">
            {files.map((file) => {
              const lines = countLines(file.content);
              return (
                <li
                  key={file.id}
                  className="submission-tree-item"
                  onDoubleClick={() => handleOpenEditor(file.id)}
                  title={'Double-click to edit ' + file.name + ' (' + lines + ' entries)'}
                >
                  <div className="tree-item-main">
                    <span className="tree-file-icon">📄</span>
                    <span className="tree-file-name">{file.name}</span>
                  </div>
                  <div className="tree-item-meta">
                    <span className={'tree-line-badge ' + (lines > 0 ? 'has-lines' : '')}>
                      {lines > 0 ? lines + ' lines' : 'empty'}
                    </span>
                    <button
                      type="button"
                      className="tree-remove-btn"
                      onClick={(event) => {
                        event.stopPropagation();
                        removeFile(file.id);
                      }}
                      title="Remove file"
                      aria-label={'Remove ' + file.name}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>

          {status && (
            <div className={'submission-status-banner ' + status.type} role="status">
              {status.message}
            </div>
          )}

          <button
            type="button"
            className="btn-primary submission-submit-btn"
            onClick={handleDownloadAll}
            disabled={nonEmptyCount === 0}
          >
            Download CSV ZIP ({nonEmptyCount})
          </button>
        </div>
      )}

      <SubmissionFileModal
        mode={modalMode}
        files={files}
        editorFile={editorFile}
        pendingLine={pendingSubmission?.line}
        onSelectFile={handleSelectFile}
        onSave={handleSaveEditor}
        onClose={handleCloseModal}
      />
    </div>
  );
};

export default SubmissionWorktree;
