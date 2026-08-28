import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const SUBMISSION_FILES_KEY = 'hcmai.submission.files';
const EMPTY_SUBMISSION_REQUEST = null;

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
  const [submissionRequest, setSubmissionRequest] = useState(EMPTY_SUBMISSION_REQUEST);

  // Keep localStorage in sync
  useEffect(() => {
    localStorage.setItem(SUBMISSION_FILES_KEY, JSON.stringify(files));
  }, [files]);

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
    localStorage.removeItem(SUBMISSION_FILES_KEY);
  }, []);

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

  const requestSubmission = useCallback((request) => {
    setSubmissionRequest(request);
  }, []);

  const clearSubmissionRequest = useCallback(() => {
    setSubmissionRequest(EMPTY_SUBMISSION_REQUEST);
  }, []);

  const value = {
    files,
    setFiles,
    addFiles,
    removeFile,
    clearAllFiles,
    clearFile,
    updateFileContent,
    submissionRequest,
    requestSubmission,
    clearSubmissionRequest,
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
      addFiles: () => {},
      removeFile: () => {},
      clearAllFiles: () => {},
      clearFile: () => {},
      updateFileContent: () => {},
      submissionRequest: null,
      requestSubmission: () => {},
      clearSubmissionRequest: () => {},
    };
  }
  return context;
};
