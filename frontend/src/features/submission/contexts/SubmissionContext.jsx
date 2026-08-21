import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const SUBMISSION_FILES_KEY = 'hcmai.submission.files';
const SELECTED_FILE_KEY = 'hcmai.submission.selected_file_id';

const SubmissionContext = createContext(null);

export const SubmissionProvider = ({ children }) => {
  const [files, setFiles] = useState(() => {
    const saved = localStorage.getItem(SUBMISSION_FILES_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) return parsed;
      } catch (e) {
        console.error('Failed to parse submission files', e);
      }
    }
    return [];
  });

  const [selectedFileId, setSelectedFileId] = useState(() => {
    const saved = localStorage.getItem(SELECTED_FILE_KEY);
    return saved || (files.length > 0 ? files[0].id : null);
  });

  // Keep selectedFileId valid when files change
  useEffect(() => {
    if (files.length > 0) {
      if (!selectedFileId || !files.some((f) => f.id === selectedFileId)) {
        setSelectedFileId(files[0].id);
      }
    } else {
      setSelectedFileId(null);
    }
  }, [files, selectedFileId]);

  // Keep localStorage in sync
  useEffect(() => {
    localStorage.setItem(SUBMISSION_FILES_KEY, JSON.stringify(files));
  }, [files]);

  useEffect(() => {
    if (selectedFileId) {
      localStorage.setItem(SELECTED_FILE_KEY, selectedFileId);
    } else {
      localStorage.removeItem(SELECTED_FILE_KEY);
    }
  }, [selectedFileId]);

  const addFiles = useCallback((newFiles) => {
    setFiles((prev) => {
      const existingIds = new Set(prev.map((f) => f.id));
      const filtered = newFiles.filter((f) => !existingIds.has(f.id));
      return [...prev, ...filtered];
    });
  }, []);

  const removeFile = useCallback((fileId) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  }, []);

  const clearAllFiles = useCallback(() => {
    setFiles([]);
    setSelectedFileId(null);
    localStorage.removeItem(SUBMISSION_FILES_KEY);
    localStorage.removeItem(SELECTED_FILE_KEY);
  }, []);

  const appendLine = useCallback((line) => {
    setFiles((prevFiles) => prevFiles.map((file) => {
      if (file.id === selectedFileId) {
        const newContent = file.content ? `${file.content}\n${line}` : line;
        return { ...file, content: newContent };
      }
      return file;
    }));
  }, [selectedFileId]);

  const clearFile = useCallback((fileId) => {
    setFiles((prevFiles) => prevFiles.map((file) => {
      if (file.id === fileId) {
        return { ...file, content: '' };
      }
      return file;
    }));
  }, []);

  const updateFileContent = useCallback((fileId, newContent) => {
    setFiles((prevFiles) => prevFiles.map((file) => {
      if (file.id === fileId) {
        return { ...file, content: newContent };
      }
      return file;
    }));
  }, []);

  const value = {
    files,
    setFiles,
    selectedFileId,
    setSelectedFileId,
    addFiles,
    removeFile,
    clearAllFiles,
    appendLine,
    clearFile,
    updateFileContent,
  };

  return (
    <SubmissionContext.Provider value={value}>
      {children}
    </SubmissionContext.Provider>
  );
};

export const useSubmission = () => {
  const context = useContext(SubmissionContext);
  if (!context) {
    return {
      files: [],
      setFiles: () => {},
      selectedFileId: null,
      setSelectedFileId: () => {},
      addFiles: () => {},
      removeFile: () => {},
      clearAllFiles: () => {},
      appendLine: () => {},
      clearFile: () => {},
      updateFileContent: () => {},
    };
  }
  return context;
};
