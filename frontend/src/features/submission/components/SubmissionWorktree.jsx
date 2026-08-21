import React from 'react';
import { useSubmission } from '../contexts/SubmissionContext';
import { submitCsvFiles } from '../../../api/search';

const SubmissionWorktree = () => {
  const { files, selectedFileId, setSelectedFileId, clearFile } = useSubmission();

  const handleSubmitAll = async () => {
    try {
      await submitCsvFiles(
        files.map(f => ({ name: f.name, content: f.content }))
      );
      alert('Submitted successfully to backend!');
    } catch (error) {
      alert(error.message);
    }
  };

  return (
    <div className="toolbox-section submission-worktree" style={{ marginTop: '16px' }}>
      <div className="toolbox-label-row">
        <label className="toolbox-label">Submission Files</label>
      </div>
      <ul style={{ listStyle: 'none', padding: 0, margin: '8px 0', fontSize: '14px' }}>
        {files.map((file) => (
          <li
            key={file.id}
            onClick={() => setSelectedFileId(file.id)}
            style={{
              padding: '6px 8px',
              cursor: 'pointer',
              backgroundColor: file.id === selectedFileId ? 'var(--color-canvas-soft)' : 'transparent',
              color: file.id === selectedFileId ? 'var(--color-primary)' : 'inherit',
              border: file.id === selectedFileId ? '1px solid var(--color-primary)' : '1px solid transparent',
              borderRadius: 'var(--rounded-md)',
              marginBottom: '4px',
              fontWeight: file.id === selectedFileId ? '600' : 'normal',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}
          >
            <span>{file.name}</span>
          </li>
        ))}
      </ul>
      <button className="btn-primary" onClick={handleSubmitAll} style={{ width: '100%', marginTop: '8px' }}>
        Submit to Backend
      </button>
    </div>
  );
};

export default SubmissionWorktree;
