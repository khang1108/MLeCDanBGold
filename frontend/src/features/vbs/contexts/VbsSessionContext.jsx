import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  connectVbsSession,
  disconnectVbsSession,
  getVbsSessionStatus,
} from '../../../api/vbs';
import { DRES_LOG_STATUS_EVENT } from '../../../api/client';

export const VBS_USER_ID_STORAGE_KEY = 'hcmai_user_id';

const VbsSessionContext = createContext(null);

const readStoredUserId = () => {
  try {
    return window.localStorage.getItem(VBS_USER_ID_STORAGE_KEY)?.trim() || '';
  } catch {
    return '';
  }
};

const writeStoredUserId = (value) => {
  try {
    if (value) window.localStorage.setItem(VBS_USER_ID_STORAGE_KEY, value);
    else window.localStorage.removeItem(VBS_USER_ID_STORAGE_KEY);
  } catch {
    // A restricted storage context must not prevent connecting or unlocking.
  }
};

/** Own the browser-safe participant identity and backend session handshake. */
export const VbsSessionProvider = ({ children }) => {
  const [initialUserId] = useState(readStoredUserId);
  const [draftUserId, setDraftUserIdState] = useState(initialUserId);
  const [connectedUserId, setConnectedUserId] = useState('');
  const [connectionState, setConnectionState] = useState(initialUserId ? 'connecting' : 'editing');
  const [error, setError] = useState('');
  const [dresLogStatus, setDresLogStatus] = useState(null);
  const requestGenerationRef = useRef(0);
  const restorePromiseRef = useRef(null);

  const setDraftUserId = useCallback((value) => {
    setDraftUserIdState(String(value ?? ''));
    setError('');
    setConnectionState((current) => (current === 'connected' ? current : 'editing'));
  }, []);

  const connect = useCallback(async () => {
    const userId = draftUserId.trim();
    if (!userId || connectionState === 'connecting') return false;

    const generation = ++requestGenerationRef.current;
    setConnectionState('connecting');
    setError('');
    try {
      const status = await connectVbsSession(userId);
      if (!status.connected) throw new Error('The backend did not establish a DRES session.');
      if (requestGenerationRef.current !== generation) return false;
      setDraftUserIdState(userId);
      setConnectedUserId(userId);
      setConnectionState('connected');
      setDresLogStatus(null);
      writeStoredUserId(userId);
      return true;
    } catch (connectError) {
      if (requestGenerationRef.current === generation) {
        setConnectedUserId('');
        setConnectionState('error');
        setError(connectError.message || 'Could not connect this VBS User ID.');
        if (readStoredUserId() !== userId) writeStoredUserId('');
      }
      return false;
    }
  }, [connectionState, draftUserId]);

  const disconnect = useCallback(async () => {
    const userId = connectedUserId;
    ++requestGenerationRef.current;
    setConnectedUserId('');
    setConnectionState('editing');
    setError('');
    setDresLogStatus(null);
    writeStoredUserId('');
    if (!userId) return;

    try {
      await disconnectVbsSession(userId);
    } catch (disconnectError) {
      setConnectionState('error');
      setError(disconnectError.message || 'Could not clear the backend VBS session.');
    }
  }, [connectedUserId]);

  useEffect(() => {
    const handleDresLogStatus = (event) => {
      const { userId, status } = event.detail || {};
      if (userId !== connectedUserId || (status !== 'sent' && status !== 'failed')) return;
      setDresLogStatus(status);
    };
    window.addEventListener(DRES_LOG_STATUS_EVENT, handleDresLogStatus);
    return () => window.removeEventListener(DRES_LOG_STATUS_EVENT, handleDresLogStatus);
  }, [connectedUserId]);

  useEffect(() => {
    const storedUserId = initialUserId;
    if (!storedUserId) return undefined;

    setDraftUserIdState(storedUserId);
    setConnectionState('connecting');
    const generation = ++requestGenerationRef.current;
    let cancelled = false;

    // React StrictMode can replay mount effects. Share one status/reconnect
    // operation so a disconnected participant is never connected twice.
    if (!restorePromiseRef.current) {
      restorePromiseRef.current = (async () => {
        const status = await getVbsSessionStatus(storedUserId);
        return status.connected ? status : connectVbsSession(storedUserId);
      })();
    }
    restorePromiseRef.current
      .then((status) => {
        if (cancelled || requestGenerationRef.current !== generation) return;
        if (!status.connected) throw new Error('The backend did not restore a DRES session.');
        setDraftUserIdState(storedUserId);
        setConnectedUserId(storedUserId);
        setConnectionState('connected');
        setError('');
        setDresLogStatus(null);
        writeStoredUserId(storedUserId);
      })
      .catch((restoreError) => {
        if (cancelled || requestGenerationRef.current !== generation) return;
        setConnectedUserId('');
        setConnectionState('error');
        setError(restoreError.message || 'Could not reconnect this VBS User ID.');
      });

    return () => { cancelled = true; };
  }, [initialUserId]);

  const value = useMemo(() => ({
    draftUserId,
    setDraftUserId,
    connectedUserId,
    connectionState,
    error,
    dresLogStatus,
    connect,
    disconnect,
  }), [draftUserId, setDraftUserId, connectedUserId, connectionState, error, dresLogStatus, connect, disconnect]);

  return <VbsSessionContext.Provider value={value}>{children}</VbsSessionContext.Provider>;
};

/** Read the current participant draft, handshake state, and connected identity. */
export const useVbsSession = () => {
  const context = useContext(VbsSessionContext);
  if (!context) throw new Error('useVbsSession must be used within VbsSessionProvider');
  return context;
};
