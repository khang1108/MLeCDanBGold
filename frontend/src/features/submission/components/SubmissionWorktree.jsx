import React, { useEffect, useRef, useState } from 'react';
import { uploadQueryFiles } from '../../../api/search';
import { useSubmission } from '../contexts/SubmissionContext';
import SubmissionFileModal from './SubmissionFileModal';
import { downloadCsvArchive, getNonEmptyCsvFiles } from '../submissionArchive';

const collectDirectoryFiles = async (directoryHandle) => {
  const files = [];
  for await (const entry of directoryHandle.values()) {
    if (entry.kind === 'file') {
      files.push(await entry.getFile());
    } else if (entry.kind === 'directory') {
      files.push(...await collectDirectoryFiles(entry));
    }
  }
  return files;
};

/**
 * Manage uploaded BTC CSV targets without making the focused tree item a
 * hidden submission destination.
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

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
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

  const processSelectedFiles = async (selectedFiles) => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    setIsUploading(true);
    setStatus(null);

    try {
      const parsedFiles = await uploadQueryFiles(selectedFiles);
      if (parsedFiles && parsedFiles.length > 0) {
        setFiles(parsedFiles);
      }
    } catch (error) {
      setStatus({ type: 'error', message: error.message || 'Failed to load query files.' });
    } finally {
      setIsUploading(false);
    }
  };

  const handleFilesSelected = async (event) => {
    await processSelectedFiles(Array.from(event.target.files || []));
    event.target.value = '';
  };

  const handleChooseFiles = async () => {
    if (typeof window.showOpenFilePicker !== 'function') {
      fileInputRef.current?.click();
      return;
    }

    try {
      const handles = await window.showOpenFilePicker({
        multiple: true,
        types: [{
          description: 'Query and CSV files',
          accept: {
            'text/plain': ['.txt'],
            'text/csv': ['.csv'],
            'application/json': ['.json'],
          },
        }],
      });
      const selectedFiles = await Promise.all(handles.map((handle) => handle.getFile()));
      await processSelectedFiles(selectedFiles);
    } catch (error) {
      if (error.name !== 'AbortError') {
        setStatus({ type: 'error', message: error.message || 'Failed to choose files.' });
      }
    }
  };

  const handleChooseFolder = async () => {
    if (typeof window.showDirectoryPicker !== 'function') {
      folderInputRef.current?.click();
      return;
    }

    try {
      const directoryHandle = await window.showDirectoryPicker({ mode: 'read' });
      const selectedFiles = await collectDirectoryFiles(directoryHandle);
      await processSelectedFiles(selectedFiles);
    } catch (error) {
      if (error.name !== 'AbortError') {
        setStatus({ type: 'error', message: error.message || 'Failed to choose folder.' });
      }
    }
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
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".txt,.csv,.json,*"
        onChange={handleFilesSelected}
        style={{ display: 'none' }}
        data-testid="query-file-input"
      />
      <input
        ref={folderInputRef}
        type="file"
        webkitdirectory=""
        directory=""
        multiple
        onChange={handleFilesSelected}
        style={{ display: 'none' }}
        data-testid="query-folder-input"
      />

      <div className="toolbox-label-row submission-header-row">
        <label className="toolbox-label">Submission Files</label>
        {files.length > 0 && (
          <div className="submission-actions-group">
            <span className="submission-count-badge">{files.length}</span>
            <button
              type="button"
              className="submission-icon-btn"
              onClick={handleChooseFiles}
              title="Add more query files"
              aria-label="Add files"
            >
              +
            </button>
            <button
              type="button"
              className="submission-icon-btn"
              onClick={clearAllFiles}
              title="Clear all files"
              aria-label="Clear all files"
            >
              ✕
            </button>
          </div>
        )}
      </div>

      {files.length === 0 ? (
        <div className="submission-upload-card">
          <div className="submission-upload-icon">📁</div>
          <p className="submission-upload-title">No Query Files</p>
          <p className="submission-upload-desc">
            Upload query files (.txt) or folder to generate .csv targets.
          </p>
          <div className="submission-upload-btn-group">
            <button
              type="button"
              className="btn-primary submission-upload-btn"
              onClick={handleChooseFiles}
              disabled={isUploading}
            >
              {isUploading ? 'Parsing Files...' : 'Upload Query Files'}
            </button>
            <button
              type="button"
              className="btn-utility submission-folder-btn"
              onClick={handleChooseFolder}
              disabled={isUploading}
            >
              Select Folder
            </button>
          </div>
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
