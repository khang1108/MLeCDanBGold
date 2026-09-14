/** Mirror the durable, task-scoped answer workspace and its revisioned mutations. */
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
  answerWorkspaceWebSocketUrl,
  getAnswerWorkspace,
  normalizeAnswerWorkspace,
} from '../../../api/answerWorkspace';

const OPEN_STATE = 1;
const MIN_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 10000;
const MUTATION_TIMEOUT_MS = 15000;
const AnswerWorkspaceContext = createContext(null);

const isSocketOpen = (socket) => Boolean(
  socket && (socket.readyState === OPEN_STATE || socket.readyState === socket.OPEN),
);

const makeError = (message, code) => {
  const error = new Error(message);
  if (code) error.code = code;
  return error;
};

const requireText = (value, field) => {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${field} must be a non-blank string`);
  return value.trim();
};

const requireTimestamp = (value, field) => {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${field} must be a non-negative integer`);
  return value;
};

const rejectMixedInput = (input, forbidden, kind) => {
  if (forbidden.some((field) => input?.[field] != null)) {
    throw new Error(`${kind} answer input cannot contain ${forbidden.join(', ')}`);
  }
};

/** Own the authenticated collaborator's HTTP hydration and WebSocket mirror. */
export const AnswerWorkspaceProvider = ({ children, connectedUserId = '' }) => {
  const normalizedUserId = typeof connectedUserId === 'string' ? connectedUserId.trim() : '';
  const [workspace, setWorkspace] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [pendingAction, setPendingAction] = useState(null);
  const workspaceRef = useRef(null);
  const socketRef = useRef(null);
  const connectRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const hydrationRef = useRef(null);
  const generationRef = useRef(0);
  const pendingRef = useRef(null);
  const mountedRef = useRef(false);

  const replaceWorkspace = useCallback((value, { allowRevisionReset = false } = {}) => {
    let next;
    try {
      next = normalizeAnswerWorkspace(value);
    } catch (error) {
      setConnectionError(error.message || 'Answer workspace event has an invalid contract');
      return null;
    }
    const current = workspaceRef.current;
    if (current && next.revision < current.revision && !allowRevisionReset) return null;
    workspaceRef.current = next;
    setWorkspace(next);
    return next;
  }, []);

  const settlePending = useCallback((error, value) => {
    const pending = pendingRef.current;
    if (!pending) return;
    window.clearTimeout(pending.timeoutId);
    pendingRef.current = null;
    setPendingAction(null);
    if (error) pending.reject(error);
    else pending.resolve(value);
  }, []);

  const eventMatchesPending = useCallback((type, nextWorkspace) => {
    const pending = pendingRef.current;
    if (!pending) return false;
    const candidates = nextWorkspace.candidates;
    if (pending.kind === 'clear-switch' && type === 'answer.task.switched') {
      return nextWorkspace.evaluation_id === pending.targetEvaluationId
        && nextWorkspace.task_scope_key === pending.targetTaskScopeKey
        && candidates.length === 0;
    }
    if (pending.kind === 'add-frame' && type === 'answer.added') {
      const matchesFrame = candidates.some((candidate) => candidate.kind === 'FRAME'
        && candidate.video_id === pending.videoId
        && candidate.timestamp_ms === pending.timestampMs);
      const duplicateAck = nextWorkspace.revision === pending.expectedWorkspaceRevision;
      return matchesFrame && (nextWorkspace.revision > pending.expectedWorkspaceRevision || duplicateAck);
    }
    if (nextWorkspace.revision <= pending.expectedWorkspaceRevision) return false;
    if (pending.kind === 'add-text' && type === 'answer.added') {
      return candidates.some((candidate) => candidate.kind === 'TEXT' && candidate.text === pending.text);
    }
    if (pending.kind === 'update-frame' && type === 'answer.updated') {
      const candidate = candidates.find((item) => item.candidate_id === pending.candidateId);
      return candidate?.kind === 'FRAME'
        && candidate.revision > pending.expectedCandidateRevision
        && candidate.video_id === pending.videoId
        && candidate.timestamp_ms === pending.timestampMs;
    }
    if (pending.kind === 'update-text' && type === 'answer.updated') {
      const candidate = candidates.find((item) => item.candidate_id === pending.candidateId);
      return candidate?.kind === 'TEXT'
        && candidate.revision > pending.expectedCandidateRevision
        && candidate.text === pending.text;
    }
    if (pending.kind === 'remove' && type === 'answer.deleted') {
      return !candidates.some((candidate) => candidate.candidate_id === pending.candidateId);
    }
    if (pending.kind === 'clear' && type === 'answer.cleared') return candidates.length === 0;
    if (pending.kind === 'set-mode' && type === 'answer.mode.changed') {
      return nextWorkspace.avs_enabled === pending.avsEnabled;
    }
    return false;
  }, []);

  const applyEvent = useCallback((payload) => {
    const eventType = typeof payload?.type === 'string' ? payload.type : '';
    if (eventType === 'answer.error') {
      if (payload.workspace) replaceWorkspace(payload.workspace);
      const error = makeError(payload.message || 'Answer workspace mutation failed', payload.code);
      if (pendingRef.current) settlePending(error);
      setConnectionError(error.message);
      return;
    }

    if (!payload?.workspace || typeof payload.workspace !== 'object') return;
    const current = workspaceRef.current;
    const incomingRevision = payload.workspace.revision;
    const isTaskSwitch = eventType === 'answer.task.switched';
    if (!isTaskSwitch && current && Number.isSafeInteger(incomingRevision) && incomingRevision <= current.revision) {
      if (incomingRevision === current.revision) {
        try {
          const unchanged = normalizeAnswerWorkspace(payload.workspace);
          if (eventMatchesPending(eventType, unchanged)) settlePending(null, current);
        } catch (error) {
          setConnectionError(error.message || 'Answer workspace event has an invalid contract');
        }
      }
      return;
    }
    const next = replaceWorkspace(payload.workspace, { allowRevisionReset: isTaskSwitch });
    if (!next) return;

    if (eventType === 'answer.conflict') {
      settlePending(makeError('The answer workspace changed; review the latest shared state.', 'REVISION_CONFLICT'));
      return;
    }
    if (eventMatchesPending(eventType, next)) settlePending(null, next);
  }, [eventMatchesPending, replaceWorkspace, settlePending]);

  const handleSocketMessage = useCallback((messageEvent) => {
    let payload;
    try {
      payload = typeof messageEvent?.data === 'string'
        ? JSON.parse(messageEvent.data)
        : messageEvent?.data || messageEvent;
    } catch (error) {
      setConnectionError('Answer workspace sent malformed WebSocket JSON');
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
    const controller = new AbortController();
    hydration.controller = controller;
    getAnswerWorkspace({ userId: normalizedUserId, signal: controller.signal })
      .then((snapshot) => {
        if (!mountedRef.current || hydrationRef.current !== hydration) return;
        const normalized = normalizeAnswerWorkspace(snapshot);
        const current = workspaceRef.current;
        if (!current || normalized.revision >= current.revision) replaceWorkspace(normalized);
        hydration.events
          .map((event, index) => ({ event, index, revision: Number(event?.workspace?.revision) || -1 }))
          .sort((left, right) => left.revision - right.revision || left.index - right.index)
          .forEach(({ event }) => applyEvent(event));
        hydrationRef.current = null;
        setConnectionError(null);
      })
      .catch((error) => {
        if (!mountedRef.current || hydrationRef.current !== hydration) return;
        hydration.events.forEach(applyEvent);
        hydrationRef.current = null;
        setConnectionError(error.message || 'Could not hydrate the answer workspace');
      });
  }, [applyEvent, normalizedUserId, replaceWorkspace]);

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
    if (!mountedRef.current || !normalizedUserId || socketRef.current) return;
    const WebSocketImpl = window.WebSocket;
    if (typeof WebSocketImpl !== 'function') {
      setConnectionError('WebSocket is unavailable in this browser');
      return;
    }
    let socket;
    try {
      socket = new WebSocketImpl(answerWorkspaceWebSocketUrl(normalizedUserId));
    } catch (error) {
      setConnectionError(error.message || 'Could not open the answer workspace WebSocket');
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
      if (mountedRef.current) setConnectionError('Answer workspace WebSocket connection error');
    };
    socket.onclose = () => {
      if (socketRef.current === socket) socketRef.current = null;
      if (!mountedRef.current) return;
      setIsConnected(false);
      scheduleReconnect();
    };
  }, [handleSocketMessage, hydrate, normalizedUserId, scheduleReconnect]);

  connectRef.current = connect;

  const cancelPending = useCallback(() => {
    if (!pendingRef.current) return;
    settlePending(makeError('Answer workspace provider was closed', 'CANCELLED'));
  }, [settlePending]);

  useEffect(() => {
    mountedRef.current = true;
    workspaceRef.current = null;
    setWorkspace(null);
    setIsConnected(false);
    setConnectionError(null);
    setPendingAction(null);
    if (normalizedUserId) connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (hydrationRef.current?.controller) hydrationRef.current.controller.abort();
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
      setIsConnected(false);
      cancelPending();
    };
  }, [cancelPending, connect, normalizedUserId]);

  const sendMutation = useCallback((kind, command, matchFields = {}, options = {}) => {
    if (!normalizedUserId) return Promise.reject(makeError('Connect a VBS user before changing the answer workspace'));
    const socket = socketRef.current;
    if (!isSocketOpen(socket)) return Promise.reject(makeError('Answer workspace WebSocket is not connected'));
    const current = workspaceRef.current;
    if (!current) return Promise.reject(makeError('Answer workspace has not hydrated yet'));
    if (current.pending_submission) {
      return Promise.reject(makeError('Submission attempt is unresolved', 'SUBMISSION_UNRESOLVED'));
    }
    if (current.task_scope_mismatch && kind !== 'clear-switch') {
      return Promise.reject(makeError('Answer workspace task scope must be resolved first', 'TASK_SCOPE_MISMATCH'));
    }
    if (pendingRef.current) return Promise.reject(makeError('Another answer workspace mutation is pending'));

    const expectedWorkspaceRevision = options.expectedWorkspaceRevision ?? current.revision;
    if (!Number.isSafeInteger(expectedWorkspaceRevision) || expectedWorkspaceRevision < 0) {
      return Promise.reject(makeError('expectedWorkspaceRevision must be a non-negative integer'));
    }
    const message = { ...command, expected_workspace_revision: expectedWorkspaceRevision };
    return new Promise((resolve, reject) => {
      const timeoutId = window.setTimeout(() => {
        settlePending(makeError('Timed out waiting for the answer workspace mutation', 'MUTATION_TIMEOUT'));
      }, MUTATION_TIMEOUT_MS);
      pendingRef.current = {
        kind,
        expectedWorkspaceRevision,
        ...matchFields,
        resolve,
        reject,
        timeoutId,
      };
      setPendingAction(message.type);
      try {
        socket.send(JSON.stringify(message));
      } catch (error) {
        settlePending(error);
      }
    });
  }, [normalizedUserId, settlePending]);

  const addFrame = useCallback((input = {}) => {
    try {
      rejectMixedInput(input, ['text'], 'FRAME');
      const videoId = requireText(input.videoId, 'videoId');
      const timestampMs = requireTimestamp(input.timestampMs, 'timestampMs');
      const sourceFrameId = input.sourceFrameId == null ? null : requireText(input.sourceFrameId, 'sourceFrameId');
      return sendMutation('add-frame', {
        type: 'answer.add_frame',
        video_id: videoId,
        timestamp_ms: timestampMs,
        ...(sourceFrameId ? { source_frame_id: sourceFrameId } : {}),
      }, { videoId, timestampMs });
    } catch (error) {
      return Promise.reject(error);
    }
  }, [sendMutation]);

  const addText = useCallback((input = {}) => {
    try {
      rejectMixedInput(input, ['videoId', 'timestampMs', 'sourceFrameId'], 'TEXT');
      const text = requireText(input.text, 'text');
      return sendMutation('add-text', { type: 'answer.add_text', text }, { text });
    } catch (error) {
      return Promise.reject(error);
    }
  }, [sendMutation]);

  const updateFrame = useCallback((input = {}) => {
    try {
      rejectMixedInput(input, ['text'], 'FRAME');
      const candidateId = requireText(input.candidateId, 'candidateId');
      const expectedCandidateRevision = input.expectedCandidateRevision;
      if (!Number.isSafeInteger(expectedCandidateRevision) || expectedCandidateRevision < 1) {
        throw new Error('expectedCandidateRevision must be a positive integer');
      }
      const videoId = requireText(input.videoId, 'videoId');
      const timestampMs = requireTimestamp(input.timestampMs, 'timestampMs');
      return sendMutation('update-frame', {
        type: 'answer.update_frame',
        candidate_id: candidateId,
        expected_candidate_revision: expectedCandidateRevision,
        video_id: videoId,
        timestamp_ms: timestampMs,
      }, { candidateId, expectedCandidateRevision, videoId, timestampMs });
    } catch (error) {
      return Promise.reject(error);
    }
  }, [sendMutation]);

  const updateText = useCallback((input = {}) => {
    try {
      rejectMixedInput(input, ['videoId', 'timestampMs', 'sourceFrameId'], 'TEXT');
      const candidateId = requireText(input.candidateId, 'candidateId');
      const expectedCandidateRevision = input.expectedCandidateRevision;
      if (!Number.isSafeInteger(expectedCandidateRevision) || expectedCandidateRevision < 1) {
        throw new Error('expectedCandidateRevision must be a positive integer');
      }
      const text = requireText(input.text, 'text');
      return sendMutation('update-text', {
        type: 'answer.update_text',
        candidate_id: candidateId,
        expected_candidate_revision: expectedCandidateRevision,
        text,
      }, { candidateId, expectedCandidateRevision, text });
    } catch (error) {
      return Promise.reject(error);
    }
  }, [sendMutation]);

  const remove = useCallback((input = {}) => {
    try {
      const candidateId = requireText(input.candidateId, 'candidateId');
      const expectedCandidateRevision = input.expectedCandidateRevision;
      if (!Number.isSafeInteger(expectedCandidateRevision) || expectedCandidateRevision < 1) {
        throw new Error('expectedCandidateRevision must be a positive integer');
      }
      return sendMutation('remove', {
        type: 'answer.delete',
        candidate_id: candidateId,
        expected_candidate_revision: expectedCandidateRevision,
      }, { candidateId });
    } catch (error) {
      return Promise.reject(error);
    }
  }, [sendMutation]);

  const clear = useCallback(() => sendMutation('clear', { type: 'answer.clear' }), [sendMutation]);

  const clearAndSwitchTask = useCallback((reviewedScope = {}) => {
    const current = workspaceRef.current;
    if (!current?.task_scope_mismatch) {
      return Promise.reject(makeError('Answer workspace task scope does not need switching'));
    }
    const oldEvaluationId = reviewedScope.oldEvaluationId ?? current.evaluation_id;
    const oldTaskScopeKey = reviewedScope.oldTaskScopeKey ?? current.task_scope_key;
    const targetEvaluationId = reviewedScope.targetEvaluationId ?? current.active_evaluation_id;
    const targetTaskScopeKey = reviewedScope.targetTaskScopeKey ?? current.active_task_scope_key;
    return sendMutation('clear-switch', {
      type: 'answer.task.clear_and_switch',
      expected_old_evaluation_id: oldEvaluationId,
      expected_old_task_scope_key: oldTaskScopeKey,
      target_evaluation_id: targetEvaluationId,
      target_task_scope_key: targetTaskScopeKey,
    }, {
      targetEvaluationId,
      targetTaskScopeKey,
    }, {
      expectedWorkspaceRevision: reviewedScope.expectedWorkspaceRevision ?? current.revision,
    });
  }, [sendMutation]);

  const refreshWorkspace = useCallback(async () => {
    if (!normalizedUserId) throw makeError('Connect a VBS user before loading the answer workspace');
    const snapshot = await getAnswerWorkspace({ userId: normalizedUserId });
    const current = workspaceRef.current;
    if (!current || snapshot.revision >= current.revision
        || snapshot.task_scope_key !== current.task_scope_key || snapshot.evaluation_id !== current.evaluation_id) {
      replaceWorkspace(snapshot, {
        allowRevisionReset: Boolean(current && (snapshot.task_scope_key !== current.task_scope_key
          || snapshot.evaluation_id !== current.evaluation_id)),
      });
    }
    return workspaceRef.current || snapshot;
  }, [normalizedUserId, replaceWorkspace]);

  const setAvsEnabled = useCallback((value) => {
    if (typeof value !== 'boolean') return Promise.reject(new Error('avsEnabled must be a boolean'));
    return sendMutation('set-mode', { type: 'answer.mode.set', avs_enabled: value }, { avsEnabled: value });
  }, [sendMutation]);

  const value = useMemo(() => ({
    connectedUserId: normalizedUserId,
    workspace,
    candidates: workspace?.candidates || [],
    isConnected,
    connectionError,
    pendingAction,
    addFrame,
    addText,
    updateFrame,
    updateText,
    remove,
    clear,
    clearAndSwitchTask,
    setAvsEnabled,
    refreshWorkspace,
  }), [
    normalizedUserId, workspace, isConnected, connectionError, pendingAction,
    addFrame, addText, updateFrame, updateText, remove, clear, clearAndSwitchTask, setAvsEnabled,
    refreshWorkspace,
  ]);

  return <AnswerWorkspaceContext.Provider value={value}>{children}</AnswerWorkspaceContext.Provider>;
};

/** Read current shared answer state and revision-aware mutation actions. */
export const useAnswerWorkspace = () => {
  const context = useContext(AnswerWorkspaceContext);
  if (!context) throw new Error('useAnswerWorkspace must be used within AnswerWorkspaceProvider');
  return context;
};

/** Read answer-workspace state when a shared frame card is outside its provider. */
export const useOptionalAnswerWorkspace = () => useContext(AnswerWorkspaceContext);

export default AnswerWorkspaceProvider;
