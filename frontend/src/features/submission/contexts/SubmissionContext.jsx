/**
 * Mirror the shared SQLite submission-file collection in React memory.
 *
 * This provider owns the global WebSocket lifecycle and HTTP hydration. It
 * intentionally has no browser-storage fallback and exposes only filename-
 * keyed, revision-aware mutations.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { getSubmissionFiles, workspaceWebSocketUrl } from '../../../api/workspace';

const OPEN_STATE = 1;
const MIN_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 10000;
const SubmissionContext = createContext(null);

const cancellationError = (message = 'Submission provider was closed') => {
  const error = new Error(message);
  error.name = 'CancellationError';
  return error;
};

const normalizeFile = (file) => {
  if (!file || typeof file !== 'object'
      || typeof file.name !== 'string' || !file.name.trim()
      || typeof file.content !== 'string'
      || typeof file.is_validated !== 'boolean'
      || !Number.isInteger(file.revision) || file.revision < 0) {
    throw new Error('Submission file event has an invalid file contract');
  }
  return {
    name: file.name,
    content: file.content,
    is_validated: file.is_validated,
    revision: file.revision,
  };
};

const normalizeFiles = (files) => {
  if (!Array.isArray(files)) throw new Error('Submission file response must contain files');
  return files.map(normalizeFile);
};

const sortFiles = (files) => [...files].sort((left, right) => left.name.localeCompare(right.name));
const eventName = (payload) => (typeof payload?.type === 'string' ? payload.type : '');
const fileEventTypes = new Set([
  'submission_file.created',
  'submission_file.updated',
  'submission_file.conflict',
]);
const isSocketOpen = (socket) => Boolean(
  socket && (socket.readyState === OPEN_STATE || socket.readyState === socket.OPEN),
);

export const SubmissionProvider = ({ children }) => {
  const [fileState, setFileState] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [pendingFileNames, setPendingFileNames] = useState([]);
  const socketRef = useRef(null);
  const connectRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const hydrationRef = useRef(null);
  const generationRef = useRef(0);
  const pendingRef = useRef(new Map());
  const mountedRef = useRef(false);

  const updatePendingNames = useCallback(() => {
    setPendingFileNames(Array.from(pendingRef.current.keys()).sort());
  }, []);

  const settlePending = useCallback((name, error, file) => {
    const pending = pendingRef.current.get(name);
    if (!pending) return;
    pendingRef.current.delete(name);
    updatePendingNames();
    if (error) pending.reject(error);
    else pending.resolve(file);
  }, [updatePendingNames]);

  const applyEvent = useCallback((payload) => {
    const type = eventName(payload);
    if (fileEventTypes.has(type)) {
      let file;
      try {
        file = normalizeFile(payload.file);
      } catch (error) {
        console.warn(error.message, payload);
        return;
      }

      setFileState((current) => {
        const existing = current.find((item) => item.name === file.name);
        if (type !== 'submission_file.conflict' && existing && file.revision <= existing.revision) return current;
        return sortFiles(existing
          ? current.map((item) => (item.name === file.name ? file : item))
          : [...current, file]);
      });

      const pending = pendingRef.current.get(file.name);
      if (!pending) return;
      const matches = pending.kind === 'create'
        ? type === 'submission_file.created'
        : pending.kind === 'update'
          ? type === 'submission_file.updated'
            && file.revision > pending.expectedRevision
            && file.content === pending.content
          : pending.kind === 'validate'
            ? type === 'submission_file.updated'
              && file.revision > pending.expectedRevision
              && file.is_validated === true
            : false;
      if (matches) settlePending(file.name, null, file);
      if (type === 'submission_file.conflict') {
        const error = new Error(`Submission file ${file.name} changed on the server`);
        error.code = 'REVISION_CONFLICT';
        error.latestFile = file;
        settlePending(file.name, error);
      }
      return;
    }

    if (type === 'submission_file.deleted') {
      if (typeof payload.name !== 'string' || !payload.name.trim()) {
        console.warn('Ignoring malformed submission_file.deleted event', payload);
        return;
      }
      setFileState((current) => current.filter((file) => file.name !== payload.name));
      settlePending(payload.name, null, { name: payload.name });
      return;
    }

    if (type === 'submission_file.error') {
      const name = typeof payload.name === 'string' && payload.name.trim()
        ? payload.name
        : null;
      const message = typeof payload.message === 'string'
        ? payload.message
        : typeof payload.error === 'string'
          ? payload.error
          : 'The submission file operation failed';
      const error = new Error(message);
      error.code = 'SUBMISSION_FILE_ERROR';
      if (name && pendingRef.current.has(name)) settlePending(name, error);
      else setConnectionError(message);
      return;
    }
    console.warn('Ignoring unknown workspace WebSocket event', payload);
  }, [settlePending]);

  const handleSocketMessage = useCallback((messageEvent) => {
    let payload;
    try {
      payload = typeof messageEvent?.data === 'string'
        ? JSON.parse(messageEvent.data)
        : messageEvent?.data || messageEvent;
    } catch (error) {
      console.warn('Ignoring malformed workspace WebSocket JSON', error);
      return;
    }
    if (hydrationRef.current) hydrationRef.current.events.push(payload);
    else applyEvent(payload);
  }, [applyEvent]);

  const hydrate = useCallback(() => {
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    const hydration = { generation, events: [] };
    hydrationRef.current = hydration;
    getSubmissionFiles()
      .then((payload) => {
        if (!mountedRef.current || hydrationRef.current !== hydration) return;
        setFileState(sortFiles(normalizeFiles(payload.files)));
        hydration.events.forEach(applyEvent);
        hydrationRef.current = null;
        setConnectionError(null);
      })
      .catch((error) => {
        if (!mountedRef.current || hydrationRef.current !== hydration) return;
        hydration.events.forEach(applyEvent);
        hydrationRef.current = null;
        setConnectionError(error.message || 'Could not hydrate submission files');
      });
  }, [applyEvent]);

  const scheduleReconnect = useCallback(() => {
    if (!mountedRef.current || reconnectTimerRef.current) return;
    const delay = Math.min(
      MIN_RECONNECT_DELAY_MS * (2 ** reconnectAttemptRef.current),
      MAX_RECONNECT_DELAY_MS,
    );
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      connectRef.current?.();
    }, delay);
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current || socketRef.current) return;
    const WebSocketImpl = window.WebSocket;
    if (typeof WebSocketImpl !== 'function') {
      setConnectionError('WebSocket is unavailable in this browser');
      return;
    }
    let socket;
    try {
      socket = new WebSocketImpl(workspaceWebSocketUrl());
    } catch (error) {
      setConnectionError(error.message || 'Could not open workspace WebSocket');
      scheduleReconnect();
      return;
    }
    socketRef.current = socket;
    socket.onopen = () => {
      if (socketRef.current !== socket) return;
      reconnectAttemptRef.current = 0;
      setIsConnected(true);
      setConnectionError(null);
      hydrate();
    };
    socket.onmessage = handleSocketMessage;
    socket.onerror = () => {
      if (mountedRef.current) setConnectionError('Workspace WebSocket connection error');
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      if (!mountedRef.current) return;
      setIsConnected(false);
      scheduleReconnect();
    };
  }, [handleSocketMessage, hydrate, scheduleReconnect]);

  // The reconnect callback is mutually recursive with connect. A ref keeps
  // the scheduled timer on the latest function without a hook dependency loop.
  connectRef.current = connect;

  const cancelPendingOperations = useCallback(() => {
    pendingRef.current.forEach((pending) => pending.reject(cancellationError()));
    pendingRef.current.clear();
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    hydrate();
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      hydrationRef.current = null;
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close?.();
      }
      cancelPendingOperations();
    };
  }, [cancelPendingOperations, connect]);

  const sendMutation = useCallback(({ name, kind, command, expectedRevision, content }) => {
    const socket = socketRef.current;
    if (!isSocketOpen(socket)) return Promise.reject(new Error('Workspace WebSocket is not connected'));
    if (pendingRef.current.has(name)) {
      return Promise.reject(new Error(`A ${name} operation is already pending`));
    }
    return new Promise((resolve, reject) => {
      pendingRef.current.set(name, { kind, expectedRevision, content, resolve, reject });
      updatePendingNames();
      try {
        socket.send(JSON.stringify(command));
      } catch (error) {
        settlePending(name, error);
      }
    });
  }, [settlePending, updatePendingNames]);

  const createFile = useCallback(({ name, content = '' } = {}) => {
    if (typeof name !== 'string' || !name.trim()) return Promise.reject(new Error('name must be a non-blank string'));
    if (typeof content !== 'string') return Promise.reject(new Error('content must be a string'));
    return sendMutation({ name, kind: 'create', command: { type: 'submission_file.create', name, content } });
  }, [sendMutation]);

  const updateFile = useCallback(({ name, content, expectedRevision } = {}) => {
    if (typeof name !== 'string' || !name.trim()) return Promise.reject(new Error('name must be a non-blank string'));
    if (typeof content !== 'string') return Promise.reject(new Error('content must be a string'));
    if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
      return Promise.reject(new Error('expectedRevision must be a non-negative integer'));
    }
    return sendMutation({
      name,
      kind: 'update',
      expectedRevision,
      content,
      command: { type: 'submission_file.update', name, content, expected_revision: expectedRevision },
    });
  }, [sendMutation]);

  const validateFile = useCallback(({ name, expectedRevision, isValidated = true } = {}) => {
    if (typeof name !== 'string' || !name.trim()) return Promise.reject(new Error('name must be a non-blank string'));
    if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
      return Promise.reject(new Error('expectedRevision must be a non-negative integer'));
    }
    return sendMutation({
      name,
      kind: 'validate',
      expectedRevision,
      command: {
        type: 'submission_file.validate',
        name,
        is_validated: Boolean(isValidated),
        expected_revision: expectedRevision,
      },
    });
  }, [sendMutation]);

  const deleteFile = useCallback(({ name, expectedRevision } = {}) => {
    if (typeof name !== 'string' || !name.trim()) return Promise.reject(new Error('name must be a non-blank string'));
    if (!Number.isInteger(expectedRevision) || expectedRevision < 0) {
      return Promise.reject(new Error('expectedRevision must be a non-negative integer'));
    }
    return sendMutation({
      name,
      kind: 'delete',
      expectedRevision,
      command: { type: 'submission_file.delete', name, expected_revision: expectedRevision },
    });
  }, [sendMutation]);

  const refreshFiles = useCallback(() => {
    if (!isSocketOpen(socketRef.current)) return Promise.reject(new Error('Workspace WebSocket is not connected'));
    hydrate();
    return Promise.resolve();
  }, [hydrate]);

  const files = useMemo(() => sortFiles(fileState), [fileState]);
  const value = useMemo(() => ({
    files,
    isConnected,
    connectionError,
    pendingFileNames,
    createFile,
    updateFile,
    validateFile,
    deleteFile,
    refreshFiles,
  }), [connectionError, createFile, deleteFile, files, isConnected, pendingFileNames, refreshFiles, updateFile, validateFile]);

  return <SubmissionContext.Provider value={value}>{children}</SubmissionContext.Provider>;
};

export const useSubmission = () => {
  const context = useContext(SubmissionContext);
  if (context) return context;
  return {
    files: [], isConnected: false, connectionError: null, pendingFileNames: [],
    createFile: () => Promise.reject(new Error('SubmissionProvider is missing')),
    updateFile: () => Promise.reject(new Error('SubmissionProvider is missing')),
    validateFile: () => Promise.reject(new Error('SubmissionProvider is missing')),
    deleteFile: () => Promise.reject(new Error('SubmissionProvider is missing')),
    refreshFiles: () => Promise.reject(new Error('SubmissionProvider is missing')),
  };
};
