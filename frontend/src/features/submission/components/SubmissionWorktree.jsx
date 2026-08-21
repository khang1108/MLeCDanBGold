import React, { useRef, useState } from 'react';
import { useSubmission } from '../contexts/SubmissionContext';
import { submitCsvFiles, uploadQueryFiles } from '../../../api/search';

/**
 * IDE-style Source Tree for Submission Files.
 * Allows uploading query files/folders (.txt) and manages corresponding .csv submission targets.
 */
const SubmissionWorktree = () => {
  const {
    files,
    selectedFileId,
    setSelectedFileId,
    setFiles,
    removeFile,
    clearAllFiles,
  } = useSubmission();

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);

  const handleFilesSelected = async (e) => {
    const selectedFiles = e.target.files;
    if (!selectedFiles || selectedFiles.length === 0) return;

    setIsUploading(true);
    setSubmitStatus(null);

    try {
      // Send files to backend endpoint /api/v1/parse-query-files or fallback to client parsing
      const parsedFiles = await uploadQueryFiles(selectedFiles);
      if (parsedFiles && parsedFiles.length > 0) {
        setFiles(parsedFiles);
        setSelectedFileId(parsedFiles[0].id);
      }
    } catch (err) {
      console.error('Failed to parse query files:', err);
    } finally {
      setIsUploading(false);
      // Reset input value so re-selecting same files triggers change
      e.target.value = '';
    }
  };

  const handleSubmitAll = async () => {
    if (files.length === 0 || isSubmitting) return;
    setIsSubmitting(true);
    setSubmitStatus(null);

    try {
      await submitCsvFiles(
        files.map((f) => ({ name: f.name, content: f.content || '' }))
      );
      setSubmitStatus({ type: 'success', message: 'Submitted successfully to backend!' });
    } catch (error) {
      setSubmitStatus({ type: 'error', message: error.message || 'Submission failed' });
    } finally {
      setIsSubmitting(false);
    }
  };

  const countLines = (content) => {
    if (!content) return 0;
    return content.trim().split('\n').filter(Boolean).length;
  };

  return (
    <div className="toolbox-section submission-worktree">
      {/* Hidden file inputs for files & folder picker */}
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
              onClick={() => fileInputRef.current?.click()}
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
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploading}
            >
              {isUploading ? 'Parsing Files...' : 'Upload Query Files'}
            </button>
            <button
              type="button"
              className="btn-utility submission-folder-btn"
              onClick={() => folderInputRef.current?.click()}
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
              const isSelected = file.id === selectedFileId;
              const lines = countLines(file.content);
              return (
                <li
                  key={file.id}
                  className={`submission-tree-item ${isSelected ? 'active' : ''}`}
                  onClick={() => setSelectedFileId(file.id)}
                  title={`Selected: ${file.name} (${lines} entries)`}
                >
                  <div className="tree-item-main">
                    <span className="tree-file-icon">📄</span>
                    <span className="tree-file-name">{file.name}</span>
                  </div>
                  <div className="tree-item-meta">
                    <span className={`tree-line-badge ${lines > 0 ? 'has-lines' : ''}`}>
                      {lines > 0 ? `${lines} lines` : 'empty'}
                    </span>
                    <button
                      type="button"
                      className="tree-remove-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(file.id);
                      }}
                      title="Remove file"
                      aria-label={`Remove ${file.name}`}
                    >
                      ✕
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>

          {submitStatus && (
            <div className={`submission-status-banner ${submitStatus.type}`}>
              {submitStatus.message}
            </div>
          )}

          <button
            type="button"
            className="btn-primary submission-submit-btn"
            onClick={handleSubmitAll}
            disabled={isSubmitting || files.length === 0}
          >
            {isSubmitting ? 'Submitting...' : `Submit to Backend (${files.length})`}
          </button>
        </div>
      )}
    </div>
  );
};

export default SubmissionWorktree;
