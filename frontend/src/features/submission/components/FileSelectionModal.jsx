import React, { useEffect, useState, useRef } from 'react';
import { useSubmission } from '../contexts/SubmissionContext';

const FileSelectionModal = ({ isOpen, onClose }) => {
  const { files, selectedFileId, setSelectedFileId, updateFileContent } = useSubmission();
  const [focusedIndex, setFocusedIndex] = useState(() => {
    const idx = files.findIndex((f) => f.id === selectedFileId);
    return idx >= 0 ? idx : 0;
  });
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const textareaRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      const idx = files.findIndex((f) => f.id === selectedFileId);
      setFocusedIndex(idx >= 0 ? idx : 0);
      setIsEditing(false);
    }
  }, [isOpen, files, selectedFileId]);

  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [isEditing]);

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (isEditing) {
        if (e.key === 'Escape') {
          e.preventDefault();
          setIsEditing(false); // Cancel edit
        } else if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          updateFileContent(files[focusedIndex].id, editContent);
          setSelectedFileId(files[focusedIndex].id);
          setIsEditing(false);
          onClose();
        }
        return;
      }

      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      } else if (files.length > 0) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          setFocusedIndex((prev) => (prev + 1) % files.length);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          setFocusedIndex((prev) => (prev - 1 + files.length) % files.length);
        } else if (e.key === 'Enter' && files[focusedIndex]) {
          e.preventDefault();
          setEditContent(files[focusedIndex].content);
          setIsEditing(true);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, files, focusedIndex, isEditing, editContent, setSelectedFileId, updateFileContent, onClose]);

  if (!isOpen) return null;

  if (files.length === 0) {
    return (
      <div className="modal-overlay" onClick={onClose} style={{ zIndex: 9999 }}>
        <div 
          className="vim-help-card" 
          onClick={(e) => e.stopPropagation()}
          style={{ width: '400px' }}
        >
          <div className="vim-help-header">
            <div>
              <span className="vim-prompt-tag">SELECT CSV FILE</span>
              <h3 className="vim-help-title">No CSV Files Loaded</h3>
            </div>
            <button type="button" className="modal-close-btn" onClick={onClose} title="Close [Esc]">
              ✕
            </button>
          </div>
          <div className="vim-help-body" style={{ textAlign: 'center', padding: '24px 16px' }}>
            <p style={{ color: 'var(--color-ink-muted)', marginBottom: '16px', fontSize: '13px' }}>
              Please upload query files in the sidebar to generate submission targets.
            </p>
            <button type="button" className="btn-primary" onClick={onClose}>
              Got It
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="modal-overlay" onClick={onClose} style={{ zIndex: 9999 }}>
      <div 
        className="vim-help-card" 
        onClick={(e) => e.stopPropagation()}
        style={{ width: isEditing ? '600px' : '400px' }}
      >
        <div className="vim-help-header">
          <div>
            <span className="vim-prompt-tag">SELECT CSV FILE</span>
            <h3 className="vim-help-title">
              {isEditing ? `Editing: ${files[focusedIndex].name}` : 'Choose file to edit (Enter to edit)'}
            </h3>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} title="Close [Esc]">
            ✕
          </button>
        </div>
        <div className="vim-help-body">
          {isEditing ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <textarea
                ref={textareaRef}
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                style={{
                  width: '100%',
                  height: '300px',
                  backgroundColor: 'var(--color-bg-secondary)',
                  color: 'var(--color-text-primary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: '4px',
                  padding: '8px',
                  fontFamily: 'monospace',
                  resize: 'vertical'
                }}
                placeholder="File is empty..."
              />
              <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                <span><strong>Shift+Enter</strong>: New line</span>
                <span><strong>Enter</strong>: Save & Select</span>
                <span><strong>Esc</strong>: Cancel</span>
              </div>
            </div>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
              {files.map((file, index) => (
                <li
                  key={file.id}
                  onClick={() => {
                    setFocusedIndex(index);
                    setEditContent(file.content);
                    setIsEditing(true);
                  }}
                  style={{
                    padding: '12px 16px',
                    cursor: 'pointer',
                    backgroundColor: index === focusedIndex ? 'var(--color-bg-tertiary)' : 'transparent',
                    color: index === focusedIndex ? 'var(--color-primary-light)' : 'inherit',
                    borderLeft: index === focusedIndex ? '4px solid var(--color-primary-light)' : '4px solid transparent',
                    display: 'flex',
                    justifyContent: 'space-between',
                  }}
                >
                  <span>{file.name}</span>
                  {file.id === selectedFileId && <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)' }}>Current</span>}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default FileSelectionModal;
