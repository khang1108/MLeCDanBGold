import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const SUBMISSION_FILES_KEY = 'hcmai.submission.files';
const SELECTED_FILE_KEY = 'hcmai.submission.selected_file_id';

const INITIAL_FILES = [
  { id: 'query-1-kis.csv', name: 'query-1-kis.csv', content: '' },
  { id: 'query-2-kis.csv', name: 'query-2-kis.csv', content: '' },
  { id: 'query-3-qa.csv', name: 'query-3-qa.csv', content: '' },
  { id: 'query-4-trake.csv', name: 'query-4-trake.csv', content: '' },
];

const SubmissionContext = createContext(null);

export const SubmissionProvider = ({ children }) => {
  const [files, setFiles] = useState(() => {
    const saved = localStorage.getItem(SUBMISSION_FILES_KEY);
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch (e) {
        console.error('Failed to parse submission files', e);
      }
    }
    return INITIAL_FILES;
  });

  const [selectedFileId, setSelectedFileId] = useState(() => {
    const saved = localStorage.getItem(SELECTED_FILE_KEY);
    return saved || INITIAL_FILES[0].id;
  });

  // Keep localStorage in sync
  useEffect(() => {
    localStorage.setItem(SUBMISSION_FILES_KEY, JSON.stringify(files));
  }, [files]);

  useEffect(() => {
    localStorage.setItem(SELECTED_FILE_KEY, selectedFileId);
  }, [selectedFileId]);

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
    selectedFileId,
    setSelectedFileId,
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
    throw new Error('useSubmission must be used within a SubmissionProvider');
  }
  return context;
};
