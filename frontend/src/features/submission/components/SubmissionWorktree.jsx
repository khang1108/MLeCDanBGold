/**
 * Display and create shared submission filenames in Workspace.
 *
 * Editing is delegated to the global SubmissionDialogProvider. This component
 * never owns a second file list, modal, file identity, or browser-storage key.
 */
import React, { useRef, useState } from 'react';
import { useSubmission } from '../contexts/SubmissionContext';
import { useSubmissionDialog } from '../contexts/SubmissionDialogContext';
import { downloadCsvArchive, getNonEmptyCsvFiles } from '../submissionArchive';

const collectDirectoryFiles = async (directoryHandle) => {
  const files = [];
  for await (const entry of directoryHandle.values()) {
    if (entry.kind === 'file') files.push(await entry.getFile());
    else if (entry.kind === 'directory') files.push(...await collectDirectoryFiles(entry));
  }
  return files;
};

const targetNameFor = (fileName) => (
  fileName.toLowerCase().endsWith('.csv')
    ? fileName
    : `${fileName.replace(/\.[^/.]+$/, '')}.csv`
);

export const fileVisualState = (file) => {
  if (!file?.content?.trim()) return 'empty';
  return file.is_validated ? 'validated' : 'filled';
};

const SubmissionWorktree = () => {
  const { files, createFile, connectionError, pendingFileNames } = useSubmission();
  const { openEditor } = useSubmissionDialog();
  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [status, setStatus] = useState(null);

  const processSelectedFiles = async (selectedFiles) => {
    if (!selectedFiles?.length) return;
    setIsUploading(true);
    setStatus(null);
    const existingNames = new Set(files.map((file) => file.name));
    const targets = [];
    const skipped = [];
    selectedFiles.forEach((file) => {
      const name = targetNameFor(file.name);
      if (existingNames.has(name) || targets.some((target) => target === name)) skipped.push(name);
      else targets.push(name);
    });

    const failures = [];
    for (const name of targets) {
      try {
        await createFile({ name, content: '' });
        existingNames.add(name);
      } catch (error) {
        failures.push(`${name}: ${error.message || 'create failed'}`);
      }
    }
    setIsUploading(false);
    if (failures.length || skipped.length) {
      const details = [
        skipped.length ? `Skipped existing names: ${skipped.join(', ')}` : '',
        failures.length ? `Could not create: ${failures.join('; ')}` : '',
      ].filter(Boolean).join(' ');
      setStatus({ type: failures.length ? 'error' : 'success', message: details });
    } else if (targets.length) {
      setStatus({ type: 'success', message: `Created ${targets.length} shared CSV file${targets.length === 1 ? '' : 's'}.` });
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
          accept: { 'text/plain': ['.txt'], 'text/csv': ['.csv'], 'application/json': ['.json'] },
        }],
      });
      await processSelectedFiles(await Promise.all(handles.map((handle) => handle.getFile())));
    } catch (error) {
      if (error.name !== 'AbortError') setStatus({ type: 'error', message: error.message || 'Failed to choose files.' });
    }
  };

  const handleChooseFolder = async () => {
    if (typeof window.showDirectoryPicker !== 'function') {
      folderInputRef.current?.click();
      return;
    }
    try {
      const directoryHandle = await window.showDirectoryPicker({ mode: 'read' });
      await processSelectedFiles(await collectDirectoryFiles(directoryHandle));
    } catch (error) {
      if (error.name !== 'AbortError') setStatus({ type: 'error', message: error.message || 'Failed to choose folder.' });
    }
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
      message: `Downloaded ${nonEmptyFiles.length} non-empty CSV file${nonEmptyFiles.length === 1 ? '' : 's'} as submissions.zip.`,
    });
  };

  const connectionMessage = connectionError && !files.length ? connectionError : null;
  const nonEmptyCount = getNonEmptyCsvFiles(files).length;

  return (
    <section className="toolbox-section submission-worktree" aria-label="Shared submission files">
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
          <button type="button" className="submission-icon-btn" onClick={handleChooseFiles} title="Add more query files" aria-label="Add files">
            +
          </button>
        )}
      </div>

      {connectionMessage && <div className="submission-status-banner error" role="alert">{connectionMessage}</div>}
      {files.length === 0 ? (
        <div className="submission-upload-card">
          <div className="submission-upload-icon" aria-hidden="true">📁</div>
          <p className="submission-upload-title">No Query Files</p>
          <p className="submission-upload-desc">Upload query files (.txt) or folder to generate .csv targets.</p>
          <div className="submission-upload-btn-group">
            <button type="button" className="btn-primary submission-upload-btn" onClick={handleChooseFiles} disabled={isUploading || (!connectionError && pendingFileNames.length > 0)}>
              {isUploading ? 'Parsing Files...' : 'Upload Query Files'}
            </button>
            <button type="button" className="btn-utility submission-folder-btn" onClick={handleChooseFolder} disabled={isUploading}>
              Select Folder
            </button>
          </div>
        </div>
      ) : (
        <div className="submission-tree-container">
          <ul className="submission-tree-list" aria-label="Shared submission filenames">
            {files.map((file) => {
              const visualState = fileVisualState(file);
              return (
                <li
                  key={file.name}
                  className={`submission-tree-item submission-file-row ${visualState}`}
                  onDoubleClick={() => openEditor(file.name)}
                  aria-label={file.name}
                  title="Double-click to edit"
                >
                  <span className="tree-file-name">{file.name}</span>
                </li>
              );
            })}
          </ul>
          {status && <div className={`submission-status-banner ${status.type}`} role="status">{status.message}</div>}
          <button type="button" className="btn-primary submission-submit-btn" onClick={handleDownloadAll} disabled={nonEmptyCount === 0}>
            Download CSV ZIP ({nonEmptyCount})
          </button>
        </div>
      )}
      {!files.length && status && <div className={`submission-status-banner ${status.type}`} role="status">{status.message}</div>}
    </section>
  );
};

export default SubmissionWorktree;
