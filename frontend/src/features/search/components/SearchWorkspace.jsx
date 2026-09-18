/**
 * Query page orchestration for KIS retrieval and history sessions.
 *
 * Retrieval contracts remain owned by the existing API modules. This module
 * adds only history persistence, canonical activity tracking, and Replay.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { searchKis, uploadKisImage } from '../../../api/kis';
import { openFeedbackSession, sendFeedbackTurn, undoFeedback } from '../../../api/feedback';
import KisPanel from '../../kis/components/KisPanel';
import {
  createInitialKisSessionState,
  setDraft,
  stageImage,
  unstageImage,
  prepareSemanticRequest,
  prepareSearchOnlyRequest,
  commitSearchSuccess,
  commitSearchFailure,
  resetKisSession,
} from '../../kis/session';
import {
  createInitialFeedbackSession,
  initFeedbackSession,
  startFeedbackTurn,
  applyFeedbackSuccess,
  applyFeedbackFailure,
  applyUndoSuccess,
  setSelectedContext,
  clearSelectedContext,
  resetFeedbackSession,
} from '../../kis/feedbackSession';
import { parseComposerDraft } from '../../kis/parser';
import { filterFrames } from '../../../api/filter';
import FilterPagination from '../../filter/components/FilterPagination';
import {
  createQueryHistory,
  markFrameViewed,
  recordQueryInteraction,
} from '../../../api/history';
import FramesBox from '../../frames/components/FramesBox';
import ToolBox from '../../search-controls/components/ToolBox';
import ReplayResults from '../../workspace/components/ReplayResults';
import {
  buildKisSnapshot,
  buildOperationMetadata,
  getSnapshotKind,
  normalizeFrameActivity,
  withViewedFrame,
  activityStateForFrame,
} from '../../workspace/queryHistory';

export const parseRetrievalDescription = (description) => {
  const query = description.trim();
  return query ? { query } : null;
};

const createClientQueryId = () => {
  if (typeof window !== 'undefined' && typeof window.crypto?.randomUUID === 'function') return window.crypto.randomUUID();
  return `query-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const parseObjectInput = (raw) => {
  if (!raw || !raw.trim()) return [];
  return raw.split(',').map((part) => {
    const trimmed = part.trim();
    if (!trimmed) return null;
    if (trimmed.includes(':')) return { value: trimmed };
    return { value: `${trimmed}: 1` };
  }).filter(Boolean);
};

const SearchWorkspace = ({
  topK = 20,
  setTopK,
  queryInputRef,
  onFocusQueryInput,
  onBlurQueryInput,
  renderExtraActions,
  onFrameClick,
  onOpenSubmission,
  isSubmissionOpening = false,
  userId,
  historyUserId,
  onHistoryRefresh,
  onQueryChange,
  replayRequest,
  onReplayHandled,
  isActive = true,
  onExplorationInvalidated,
  onEventTrailInvalidated,
  eventTrailAnnotations = {},
  eventTrail = null,
}) => {
  const notifyEventTrailInvalidated = useCallback(() => {
    onEventTrailInvalidated?.();
    onExplorationInvalidated?.();
  }, [onEventTrailInvalidated, onExplorationInvalidated]);
  const historyIdentity = typeof historyUserId === 'string'
    ? historyUserId.trim()
    : typeof userId === 'string' ? userId.trim() : '';
  const [kisSession, setKisSession] = useState(createInitialKisSessionState);
  const [feedbackSession, setFeedbackSession] = useState(createInitialFeedbackSession);
  const [useDense, setUseDense] = useState(true);
  const [useBm25, setUseBm25] = useState(true);
  const [resultType, setResultType] = useState(null);
  const [frames, setFrames] = useState([]);
  const [kisEvents, setKisEvents] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [activeQuerySession, setActiveQuerySession] = useState(null);
  const [replaySnapshot, setReplaySnapshot] = useState(null);
  const [filterFolderId, setFilterFolderId] = useState('');
  const [filterVideoId, setFilterVideoId] = useState('');
  const [filterTitle, setFilterTitle] = useState('');
  const [filterAsr, setFilterAsr] = useState('');
  const [filterOcr, setFilterOcr] = useState('');
  const [filterObject, setFilterObject] = useState('');
  const [filterPageId, setFilterPageId] = useState(1);
  const [filterTotalPages, setFilterTotalPages] = useState(0);
  const [appliedFilterParams, setAppliedFilterParams] = useState(null);
  const requestRef = useRef(null);
  const viewedPatchRef = useRef(new Set());
  const historyQueuesRef = useRef(new Map());
  const activeQuerySessionRef = useRef(null);
  const historyGenerationRef = useRef(0);
  const lastReplayTokenRef = useRef(null);
  const liveEventTrailContextRef = useRef(null);
  const prevControlsRef = useRef({ topK, useDense, useBm25 });
  const filterFolderIdRef = useRef(null);
  const filterVideoIdRef = useRef(null);
  const filterTitleRef = useRef(null);
  const filterAsrRef = useRef(null);
  const filterOcrRef = useRef(null);
  const filterObjectRef = useRef(null);
  const localQueryInputRef = useRef(null);

  const [gridSize, setGridSize] = useState(() => {
    try {
      return localStorage.getItem('hcmai_grid_size') || 'normal';
    } catch {
      return 'normal';
    }
  });

  const [isOptionsCollapsed, setIsOptionsCollapsed] = useState(() => {
    try {
      return localStorage.getItem('hcmai_options_collapsed') === 'true';
    } catch {
      return false;
    }
  });

  const handleSetGridSize = useCallback((size) => {
    setGridSize(size);
    try {
      localStorage.setItem('hcmai_grid_size', size);
    } catch {
      // ignore
    }
  }, []);

  const handleToggleOptions = useCallback((collapsed) => {
    setIsOptionsCollapsed(collapsed);
    try {
      localStorage.setItem('hcmai_options_collapsed', String(collapsed));
    } catch {
      // ignore
    }
  }, []);

  const [isChatCollapsed, setIsChatCollapsed] = useState(() => {
    try {
      return localStorage.getItem('hcmai_chat_collapsed') === 'true';
    } catch {
      return false;
    }
  });

  const handleToggleChat = useCallback((collapsed) => {
    setIsChatCollapsed(collapsed);
    try {
      localStorage.setItem('hcmai_chat_collapsed', String(collapsed));
    } catch {
      // ignore
    }
  }, []);

  const enqueueHistoryWrite = useCallback((queryId, write) => {
    const prior = historyQueuesRef.current.get(queryId) || Promise.resolve();
    const next = prior.then(write);
    historyQueuesRef.current.set(queryId, next);
    next.catch(() => undefined);
    return next;
  }, []);

  const activateHistorySession = useCallback((session) => {
    const activeSession = {
      ...session,
      generation: historyGenerationRef.current + 1,
    };
    historyGenerationRef.current = activeSession.generation;
    activeQuerySessionRef.current = activeSession;
    setActiveQuerySession(activeSession);
    return activeSession;
  }, []);

  const invalidateHistorySession = useCallback(() => {
    historyGenerationRef.current += 1;
    activeQuerySessionRef.current = null;
    setActiveQuerySession(null);
  }, []);

  const isCurrentHistorySession = useCallback((session) => {
    const current = activeQuerySessionRef.current;
    return current?.queryId === session.queryId
      && current?.generation === session.generation;
  }, []);

  const setQueryTextareaRef = useCallback((node) => {
    if (queryInputRef) queryInputRef.current = node;
    localQueryInputRef.current = node;
  }, [queryInputRef]);

  const focusQueryInput = useCallback(() => {
    const el = queryInputRef?.current || localQueryInputRef.current || document.getElementById('event-query');
    if (el) {
      el.focus();
      el.select?.();
    }
  }, [queryInputRef]);

  const handleAttachImage = useCallback(async (file, targetEventId) => {
    if (!file) return;
    try {
      const assetRef = await uploadKisImage({ imageFile: file });
      const eventId = targetEventId || 'E1';
      setKisSession((prev) => stageImage(prev, eventId, assetRef));
      setError(null);
    } catch (err) {
      setError(err?.message || 'Failed to upload image asset');
    }
  }, []);

  const handleRemoveImage = useCallback((eventId, assetId) => {
    setKisSession((prev) => unstageImage(prev, eventId, assetId));
  }, []);

  const submitFilter = useCallback(async ({ pageId = 1, overrideParams = null } = {}) => {
    const paramsToUse = overrideParams || {
      folderId: filterFolderId.trim(),
      videoId: filterVideoId.trim(),
      filters: {
        title: filterTitle.trim(),
        asr: filterAsr.trim(),
        ocr: filterOcr.trim(),
        caption: '',
        objects: parseObjectInput(filterObject),
      },
    };

    const hasCriteria = paramsToUse.folderId || paramsToUse.videoId
      || Object.values(paramsToUse.filters).some((v) => (Array.isArray(v) ? v.length > 0 : Boolean(v)));
    if (!hasCriteria && !overrideParams) return;

    requestRef.current?.abort();
    liveEventTrailContextRef.current = null;
    notifyEventTrailInvalidated();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setResultType('filter');
    setReplaySnapshot(null);
    invalidateHistorySession();
    lastReplayTokenRef.current = null;

    try {
      const capturedUserId = typeof userId === 'string' ? userId.trim() : '';
      const startTime = performance.now();
      const response = await filterFrames({
        folderId: paramsToUse.folderId,
        videoId: paramsToUse.videoId,
        filters: paramsToUse.filters,
        pageId,
        userId: capturedUserId,
        signal: controller.signal,
      });

      if (controller.signal.aborted) return;
      const elapsed = Math.round(performance.now() - startTime);

      setFrames(response.results || []);
      setKisEvents([]);
      setSearchLatencyMs(elapsed);
      setWarnings(response.warnings || []);
      setFilterPageId(response.page_id || pageId);
      setFilterTotalPages(response.total_pages || 0);
      setAppliedFilterParams(paramsToUse);
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setFrames([]);
      setError(requestError.message || 'Failed to contact filter API');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [
    filterAsr,
    filterFolderId,
    filterObject,
    filterOcr,
    filterTitle,
    filterVideoId,
    invalidateHistorySession,
    notifyEventTrailInvalidated,
    userId,
  ]);

  const handleClearFilter = useCallback(() => {
    setFilterFolderId('');
    setFilterVideoId('');
    setFilterTitle('');
    setFilterAsr('');
    setFilterOcr('');
    setFilterObject('');
    setFilterPageId(1);
    setFilterTotalPages(0);
    setAppliedFilterParams(null);
  }, []);

  const recordViewed = useCallback((frame) => {
    const frameId = frame?.frame_id;
    const session = activeQuerySession;
    if (!frameId || !session?.queryId) return;
    const patchKey = `${session.queryId}:${frameId}`;
    if (viewedPatchRef.current.has(patchKey)) return;
    viewedPatchRef.current.add(patchKey);
    setActiveQuerySession((current) => {
      if (!current || current.queryId !== session.queryId || current.generation !== session.generation) {
        return current;
      }
      const next = { ...current, frameActivity: withViewedFrame(current.frameActivity, frameId) };
      activeQuerySessionRef.current = next;
      return next;
    });

    enqueueHistoryWrite(session.queryId, async () => {
      try {
        await markFrameViewed({ queryId: session.queryId, frameId });
      } catch (patchError) {
        viewedPatchRef.current.delete(patchKey);
        if (isCurrentHistorySession(session)) {
          setWarnings((current) => Array.from(new Set([
            ...current,
            `History view state was not recorded: ${patchError.message || 'request failed'}`,
          ])));
        }
      }
    });
  }, [activeQuerySession, enqueueHistoryWrite, isCurrentHistorySession]);

  const openCanonicalFrame = useCallback((frame) => {
    recordViewed(frame);
    const session = activeQuerySession;
    if (session?.queryId) {
      enqueueHistoryWrite(session.queryId, async () => {
        try {
          await recordQueryInteraction({
            queryId: session.queryId,
            eventType: 'result_open',
            semanticRevision: kisSession.revision ?? 0,
            frameId: frame?.frame_id,
            videoId: frame?.video_id,
            timestampMs: frame?.timestamp_ms,
          });
        } catch {
          // Best-effort research logging
        }
      });
    }
    const trailContext = (
      resultType === 'retrieval'
      && liveEventTrailContextRef.current
      && frame?.result_id
    ) ? {
      snapshotId: liveEventTrailContextRef.current.snapshotId,
      resultId: frame.result_id,
      kisRevision: liveEventTrailContextRef.current.kisRevision,
      events: liveEventTrailContextRef.current.events,
      searchSessionId: liveEventTrailContextRef.current.searchSessionId,
    } : null;

    onFrameClick?.({
      frame,
      ...(trailContext ? { eventTrailContext: trailContext } : {}),
    });
  }, [activeQuerySession, enqueueHistoryWrite, kisSession.revision, onFrameClick, recordViewed, resultType]);

  const openKisFrame = useCallback((frame) => openCanonicalFrame(frame), [openCanonicalFrame]);

  const handleOpenSubmission = useCallback((payload) => {
    const session = activeQuerySession;
    if (session?.queryId) {
      enqueueHistoryWrite(session.queryId, async () => {
        try {
          await recordQueryInteraction({
            queryId: session.queryId,
            eventType: 'submission',
            semanticRevision: kisSession.revision ?? 0,
            videoId: payload?.videoId,
            timestampMs: payload?.startMs ?? payload?.endMs,
          });
        } catch {
          // Best-effort research logging
        }
      });
    }
    onOpenSubmission?.(payload);
  }, [activeQuerySession, enqueueHistoryWrite, kisSession.revision, onOpenSubmission]);

  useEffect(() => {
    const item = replayRequest?.item || replayRequest;
    if (!item || !item.query_id || !item.result_snapshot) return;
    const token = replayRequest?.token || item.query_id;
    if (lastReplayTokenRef.current === token) return;
    lastReplayTokenRef.current = token;

    requestRef.current?.abort();
    liveEventTrailContextRef.current = null;
    notifyEventTrailInvalidated();
    invalidateHistorySession();
    setIsSearching(false);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setSearchLatencyMs(null);
    handleClearFilter();
    setKisSession(createInitialKisSessionState());
    setFeedbackSession(createInitialFeedbackSession());
    onQueryChange?.(item.query_text || '');

    try {
      const kind = getSnapshotKind(item.result_snapshot);
      const normalizedActivity = normalizeFrameActivity(item.frame_activity);
      setReplaySnapshot(item.result_snapshot);
      setResultType(`replay-${kind}`);
      viewedPatchRef.current = new Set();
      activateHistorySession({
        queryId: item.query_id,
        ownerUserId: historyIdentity,
        queryText: item.query_text,
        resultSnapshot: item.result_snapshot,
        frameActivity: normalizedActivity,
        source: 'history-replay',
      });
    } catch (replayError) {
      setReplaySnapshot(null);
      setResultType(null);
      setError(replayError.message);
    }
  }, [
    activateHistorySession,
    handleClearFilter,
    historyIdentity,
    invalidateHistorySession,
    notifyEventTrailInvalidated,
    onQueryChange,
    replayRequest,
  ]);

  const handleFeedbackTurn = useCallback(async (message) => {
    const activeSnapshotId = liveEventTrailContextRef.current?.snapshotId;
    if (!activeSnapshotId) {
      setError('No active search snapshot found. Please search first.');
      return;
    }

    let activeSessionId = feedbackSession.sessionId;
    let activeFeedbackRev = feedbackSession.feedbackRevision;

    // Open feedback session lazily on first turn
    if (!activeSessionId) {
      try {
        setIsSearching(true);
        const openResp = await openFeedbackSession({
          intent: kisSession.currentIntent,
          originalQuery: kisSession.currentIntent?.query_text || message,
          evidenceSnapshotId: activeSnapshotId,
          useDense,
          useBm25,
          topK,
          initialResults: frames,
        });
        activeSessionId = openResp.session_id;
        activeFeedbackRev = openResp.feedback_revision;
        setFeedbackSession((prev) => initFeedbackSession(prev, {
          sessionId: activeSessionId,
          feedbackRevision: activeFeedbackRev,
          state: openResp.state,
          originalQuery: kisSession.currentIntent?.query_text || message,
        }));
      } catch (openErr) {
        setIsSearching(false);
        setError(openErr?.message || 'Failed to open feedback session');
        return;
      }
    }

    const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const selectedCtx = feedbackSession.selectedContext;

    setFeedbackSession((prev) => startFeedbackTurn(prev, {
      requestId,
      message,
      selectedContext: selectedCtx,
    }));
    setKisSession((prev) => setDraft(prev, ''));
    setIsSearching(true);
    setError(null);

    try {
      const turnResp = await sendFeedbackTurn(activeSessionId, {
        requestId,
        expectedFeedbackRevision: activeFeedbackRev,
        expectedKisRevision: kisSession.currentIntent?.revision ?? 1,
        expectedTrailRevision: eventTrail?.session?.trail_revision,
        message,
        selectedResultId: selectedCtx?.resultId,
        selectedEventId: selectedCtx?.eventId,
        selectedFrameId: selectedCtx?.frameId,
      });

      setFeedbackSession((prev) => applyFeedbackSuccess(prev, turnResp, { requestId }));

      if (turnResp.status === 'applied') {
        notifyEventTrailInvalidated();
        if (Array.isArray(turnResp.results)) {
          setFrames([...turnResp.results]);
          if (turnResp.latency) {
            setSearchLatencyMs(turnResp.latency);
          }
          if (historyIdentity) {
            const feedbackQueryId = createClientQueryId();
            const feedbackQueryText = turnResp.intent?.query_text || kisSession.currentIntent?.query_text || message;
            const opMeta = buildOperationMetadata({
              semanticRevision: turnResp.intent?.revision ?? kisSession.currentIntent?.revision ?? 1,
              operationKind: 'chat_feedback',
              action: turnResp.action,
              scope: turnResp.scope,
              affectedEventIds: turnResp.affected_event_ids || [],
              feedbackRevision: turnResp.feedback_revision,
            });
            const historySnapshot = buildKisSnapshot(turnResp.results, {
              intent: turnResp.intent || kisSession.currentIntent,
              latency: turnResp.latency || { total_ms: 0 },
              warnings: [],
              operationMetadata: opMeta,
            });
            activateHistorySession({
              queryId: feedbackQueryId,
              ownerUserId: historyIdentity,
              queryText: feedbackQueryText,
              resultSnapshot: historySnapshot,
              frameActivity: normalizeFrameActivity(),
              source: 'live-search',
            });
            enqueueHistoryWrite(feedbackQueryId, async () => {
              try {
                await createQueryHistory({
                  queryId: feedbackQueryId,
                  userId: historyIdentity,
                  queryText: feedbackQueryText,
                  resultSnapshot: historySnapshot,
                  operationMetadata: opMeta,
                });
                onHistoryRefresh?.();
              } catch (historyErr) {
                // Non-fatal history write
              }
            });
          }
        }
        if (turnResp.intent) {
          setKisSession((prev) => ({
            ...prev,
            currentIntent: turnResp.intent,
            revision: turnResp.intent.revision,
          }));
          const eventTexts = Array.isArray(turnResp.intent?.events)
            ? turnResp.intent.events.map((e) => (typeof e === 'string' ? e : e.text))
            : [];
          setKisEvents(eventTexts);
        }
        if (turnResp.evidence_snapshot_id) {
          if (!liveEventTrailContextRef.current) {
            liveEventTrailContextRef.current = {
              snapshotId: turnResp.evidence_snapshot_id,
              kisRevision: turnResp.intent?.revision ?? 1,
              events: (turnResp.intent?.events || []).map(({ id, text, images }) => ({
                id,
                text,
                images: images || [],
              })),
              searchSessionId: activeQuerySession?.queryId || null,
            };
          } else {
            liveEventTrailContextRef.current.snapshotId = turnResp.evidence_snapshot_id;
            liveEventTrailContextRef.current.kisRevision = turnResp.intent?.revision ?? liveEventTrailContextRef.current.kisRevision;
            if (turnResp.intent?.events) {
              liveEventTrailContextRef.current.events = turnResp.intent.events.map(({ id, text, images }) => ({
                id,
                text,
                images: images || [],
              }));
            }
          }
        }
        if (turnResp.trail && eventTrail?.syncSession) {
          eventTrail.syncSession(turnResp.trail);
        }
      }
    } catch (turnErr) {
      setFeedbackSession((prev) => applyFeedbackFailure(prev, turnErr, { requestId }));
      setError(turnErr?.message || 'Feedback turn failed');
    } finally {
      setIsSearching(false);
    }
  }, [
    feedbackSession,
    kisSession.currentIntent,
    useDense,
    useBm25,
    topK,
    frames,
    eventTrail,
    notifyEventTrailInvalidated,
    historyIdentity,
    activateHistorySession,
    enqueueHistoryWrite,
    onHistoryRefresh,
    activeQuerySession,
  ]);

  const handleUndoFeedback = useCallback(async () => {
    if (!feedbackSession.sessionId || !feedbackSession.canUndo || isSearching) return;

    const requestId = `undo_${Date.now()}`;
    setIsSearching(true);
    try {
      const undoResp = await undoFeedback(feedbackSession.sessionId, {
        requestId,
        expectedFeedbackRevision: feedbackSession.feedbackRevision,
      });

      setFeedbackSession((prev) => applyUndoSuccess(prev, undoResp));

      notifyEventTrailInvalidated();
      if (Array.isArray(undoResp.results)) {
        setFrames([...undoResp.results]);
        if (undoResp.latency) {
          setSearchLatencyMs(undoResp.latency);
        }
        if (historyIdentity) {
          const undoQueryId = createClientQueryId();
          const undoQueryText = undoResp.intent?.query_text || kisSession.currentIntent?.query_text || 'Undo';
          const opMeta = buildOperationMetadata({
            semanticRevision: undoResp.intent?.revision ?? 1,
            operationKind: 'chat_feedback_undo',
            action: 'undo',
            feedbackRevision: undoResp.feedback_revision,
          });
          const historySnapshot = buildKisSnapshot(undoResp.results, {
            intent: undoResp.intent,
            latency: undoResp.latency || { total_ms: 0 },
            warnings: [],
            operationMetadata: opMeta,
          });
          activateHistorySession({
            queryId: undoQueryId,
            ownerUserId: historyIdentity,
            queryText: undoQueryText,
            resultSnapshot: historySnapshot,
            frameActivity: normalizeFrameActivity(),
            source: 'live-search',
          });
          enqueueHistoryWrite(undoQueryId, async () => {
            try {
              await createQueryHistory({
                queryId: undoQueryId,
                userId: historyIdentity,
                queryText: undoQueryText,
                resultSnapshot: historySnapshot,
                operationMetadata: opMeta,
              });
              onHistoryRefresh?.();
            } catch (historyErr) {
              // Non-fatal history write
            }
          });
        }
      }
      if (undoResp.intent) {
        setKisSession((prev) => ({
          ...prev,
          currentIntent: undoResp.intent,
          revision: undoResp.intent.revision,
        }));
        const eventTexts = Array.isArray(undoResp.intent?.events)
          ? undoResp.intent.events.map((e) => (typeof e === 'string' ? e : e.text))
          : [];
        setKisEvents(eventTexts);
      }
      if (undoResp.evidence_snapshot_id) {
        if (!liveEventTrailContextRef.current) {
          liveEventTrailContextRef.current = {
            snapshotId: undoResp.evidence_snapshot_id,
            kisRevision: undoResp.intent?.revision ?? 1,
            events: (undoResp.intent?.events || []).map(({ id, text, images }) => ({
              id,
              text,
              images: images || [],
            })),
            searchSessionId: activeQuerySession?.queryId || null,
          };
        } else {
          liveEventTrailContextRef.current.snapshotId = undoResp.evidence_snapshot_id;
          liveEventTrailContextRef.current.kisRevision = undoResp.intent?.revision ?? liveEventTrailContextRef.current.kisRevision;
          if (undoResp.intent?.events) {
            liveEventTrailContextRef.current.events = undoResp.intent.events.map(({ id, text, images }) => ({
              id,
              text,
              images: images || [],
            }));
          }
        }
      }
      if (undoResp.trail && eventTrail?.syncSession) {
        eventTrail.syncSession(undoResp.trail);
      }
    } catch (undoErr) {
      setError(undoErr?.message || 'Undo failed');
    } finally {
      setIsSearching(false);
    }
  }, [
    feedbackSession,
    isSearching,
    eventTrail,
    notifyEventTrailInvalidated,
    historyIdentity,
    activateHistorySession,
    enqueueHistoryWrite,
    onHistoryRefresh,
    kisSession.currentIntent,
    activeQuerySession,
  ]);

  const handleSelectContext = useCallback((eventId) => {
    setFeedbackSession((prev) => {
      if (prev.selectedContext?.eventId === eventId) {
        return clearSelectedContext(prev);
      }
      return setSelectedContext(prev, { eventId });
    });
  }, []);

  const handleClearContext = useCallback(() => {
    setFeedbackSession((prev) => clearSelectedContext(prev));
  }, []);

  const submit = useCallback(async (event) => {
    event?.preventDefault?.();
    if (isSearching) return;

    let prepared;
    const draftText = (kisSession.draft || '').trim();
    const hasStagedImages = Object.keys(kisSession.stagedImages || {}).length > 0;

    try {
      if (draftText) {
        const preview = parseComposerDraft(draftText, kisSession.currentIntent);
        if (preview.error) {
          setError(preview.error);
          return;
        }
        if (preview.kind === 'feedback' && kisSession.currentIntent) {
          return handleFeedbackTurn(draftText);
        }
        prepared = prepareSemanticRequest(kisSession, preview);
      } else if (hasStagedImages) {
        const preview = kisSession.currentIntent
          ? { kind: 'patch_events', affectedEventIds: Object.keys(kisSession.stagedImages), error: null }
          : { kind: 'initial_resolve', affectedEventIds: [], error: null };
        prepared = prepareSemanticRequest(kisSession, preview);
      } else if (kisSession.currentIntent) {
        prepared = prepareSearchOnlyRequest(kisSession);
      } else {
        return;
      }
    } catch (prepareErr) {
      setError(prepareErr?.message || 'Invalid request');
      return;
    }

    const { nextState, requestPayload } = prepared;
    setKisSession(nextState);

    const capturedUserId = typeof userId === 'string' ? userId.trim() : '';
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const queryId = historyIdentity ? createClientQueryId() : null;

    setIsSearching(true);
    setError(null);
    try {
      const response = await searchKis({
        baseIntent: requestPayload.baseIntent,
        expectedRevision: requestPayload.expectedRevision,
        operation: requestPayload.operation,
        topK,
        useDense,
        useBm25,
        signal: controller.signal,
        userId: capturedUserId,
      });
      if (controller.signal.aborted) return;
      setKisSession((prev) => commitSearchSuccess(prev, response));

      const eventTexts = Array.isArray(response.intent?.events)
        ? response.intent.events.map((e) => (typeof e === 'string' ? e : e.text))
        : [];
      const queryText = response.intent?.query_text || draftText || 'Multimodal search';

      setFeedbackSession({
        ...createInitialFeedbackSession(),
        status: 'applied',
        messages: [
          {
            id: 'msg_initial_query',
            role: 'user',
            text: queryText,
          },
          {
            id: 'msg_initial_response',
            role: 'assistant',
            text: `Analyzed query into ${eventTexts.length} event${eventTexts.length === 1 ? '' : 's'}. You can refine results, reject candidates, or add clues below.`,
            scope: 'all_videos',
            changedEventIds: (response.intent?.events || []).map((e) => e.id),
          },
        ],
        currentIntent: response.intent,
        evidenceSnapshotId: response.evidence_snapshot_id || null,
        results: response.results || [],
        canUndo: false,
      });

      const imageAdded = [];
      const imageRemoved = [];
      if (Array.isArray(requestPayload.operation?.patches)) {
        for (const patch of requestPayload.operation.patches) {
          if (Array.isArray(patch.images_to_add)) {
            imageAdded.push(...patch.images_to_add);
          }
          if (Array.isArray(patch.images_to_remove)) {
            imageRemoved.push(...patch.images_to_remove);
          }
        }
      }
      if (requestPayload.operation?.kind === 'initial_resolve' && Array.isArray(requestPayload.operation.images)) {
        imageAdded.push(...requestPayload.operation.images);
      }

      const operationMetadata = buildOperationMetadata({
        semanticRevision: response.intent?.revision ?? kisSession.revision ?? 0,
        operationKind: response.operation_summary?.operation_kind || requestPayload.operation?.kind || 'initial_resolve',
        affectedEventIds: requestPayload.operation?.patches
          ? requestPayload.operation.patches.map((p) => p.event_id)
          : response.operation_summary?.affected_event_ids || [],
        imageAdded,
        imageRemoved,
        searchOnly: requestPayload.operation?.kind === 'search_only',
      });

      const snapshotOptions = {
        intent: response.intent,
        latency: response.latency,
        warnings: response.warnings || [],
        operationMetadata,
      };
      const historySnapshot = buildKisSnapshot(response.results || [], snapshotOptions);

      notifyEventTrailInvalidated();
      const activeQueryId = (queryId && requestPayload.operation?.kind !== 'search_only')
        ? queryId
        : (activeQuerySession?.queryId || null);

      if (response.evidence_snapshot_id) {
        liveEventTrailContextRef.current = {
          snapshotId: response.evidence_snapshot_id,
          kisRevision: response.intent?.revision ?? 1,
          events: (response.intent?.events || []).map(({ id, text, images }) => ({
            id,
            text,
            images: images || [],
          })),
          searchSessionId: activeQueryId,
        };
      } else {
        liveEventTrailContextRef.current = null;
      }

      setResultType('retrieval');
      setReplaySnapshot(null);
      lastReplayTokenRef.current = null;
      setFrames(response.results || []);
      setKisEvents(eventTexts);
      setSearchLatencyMs(response.latency);
      setWarnings(response.warnings || []);
      onQueryChange?.(queryText);

      if (queryId && requestPayload.operation?.kind !== 'search_only') {
        viewedPatchRef.current = new Set();
        const historySession = activateHistorySession({
          queryId,
          ownerUserId: historyIdentity,
          queryText,
          resultSnapshot: historySnapshot,
          frameActivity: normalizeFrameActivity(),
          source: 'live-search',
        });

        enqueueHistoryWrite(queryId, async () => {
          try {
            await createQueryHistory({
              queryId,
              userId: historyIdentity,
              queryText,
              resultSnapshot: historySnapshot,
              operationMetadata,
              signal: controller.signal,
            });
            if (!controller.signal.aborted && isCurrentHistorySession(historySession)) {
              onHistoryRefresh?.();
            }
          } catch (historyError) {
            if (historyError.name === 'AbortError') throw historyError;
            if (isCurrentHistorySession(historySession)) {
              invalidateHistorySession();
              setWarnings((current) => [...current, `History was not saved: ${historyError.message || 'request failed'}`]);
            }
            throw historyError;
          }
        });
      } else {
        invalidateHistorySession();
      }
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setKisSession((prev) => commitSearchFailure(prev, requestError));
      setError(requestError.message || 'Search failed');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [
    activateHistorySession,
    enqueueHistoryWrite,
    historyIdentity,
    invalidateHistorySession,
    isCurrentHistorySession,
    isSearching,
    kisSession,
    notifyEventTrailInvalidated,
    onHistoryRefresh,
    onQueryChange,
    topK,
    useBm25,
    useDense,
    userId,
    activeQuerySession?.queryId,
    handleFeedbackTurn,
  ]);

  // Step 7: Search-only rerun when only retrieval controls change
  useEffect(() => {
    const prev = prevControlsRef.current;
    const controlsChanged = prev.topK !== topK || prev.useDense !== useDense || prev.useBm25 !== useBm25;
    prevControlsRef.current = { topK, useDense, useBm25 };

    if (controlsChanged && kisSession.currentIntent && !kisSession.draft?.trim() && !isSearching) {
      setFeedbackSession(resetFeedbackSession());
      submit();
    }
  }, [topK, useDense, useBm25, kisSession.currentIntent, kisSession.draft, isSearching, submit]);

  const handleNewSearch = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    liveEventTrailContextRef.current = null;
    notifyEventTrailInvalidated();
    invalidateHistorySession();
    setIsSearching(false);
    setKisSession(resetKisSession());
    setFeedbackSession(resetFeedbackSession());
    handleClearFilter();
    setFrames([]);
    setKisEvents([]);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
    setReplaySnapshot(null);
    lastReplayTokenRef.current = null;
    onQueryChange?.('');
  }, [handleClearFilter, invalidateHistorySession, notifyEventTrailInvalidated, onQueryChange]);

  useEffect(() => {
    if (!isActive) return;

    const handleKeyDown = (event) => {
      // 1. Ctrl + Alt + B -> Toggle Options tab
      if (
        event.ctrlKey &&
        event.altKey &&
        !event.metaKey &&
        (event.key.toLowerCase() === 'b' || event.code === 'KeyB')
      ) {
        event.preventDefault();
        event.stopPropagation();
        handleToggleOptions(!isOptionsCollapsed);
        return;
      }

      // 2. Ctrl + B (without Alt) -> Toggle Chat panel
      if (
        event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        (event.key.toLowerCase() === 'b' || event.code === 'KeyB')
      ) {
        event.preventDefault();
        event.stopPropagation();
        handleToggleChat(!isChatCollapsed);
        return;
      }

      // 3. Ctrl + K -> Focus chat input (expand if collapsed)
      if (
        event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        (event.key.toLowerCase() === 'k' || event.code === 'KeyK')
      ) {
        event.preventDefault();
        event.stopPropagation();
        if (isChatCollapsed) {
          handleToggleChat(false);
        }
        setTimeout(() => {
          focusQueryInput();
        }, 0);
        return;
      }

      // 4. Ctrl + N -> New Search
      if (
        event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        (event.key.toLowerCase() === 'n' || event.code === 'KeyN')
      ) {
        event.preventDefault();
        event.stopPropagation();
        handleNewSearch();
        setTimeout(() => {
          focusQueryInput();
        }, 0);
        return;
      }

      // 5. 'n' alone when not in input/textarea -> New Search
      if (
        !event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        event.key.toLowerCase() === 'n' &&
        event.target.tagName !== 'INPUT' &&
        event.target.tagName !== 'TEXTAREA'
      ) {
        event.preventDefault();
        event.stopPropagation();
        handleNewSearch();
        return;
      }

      // 6. Ctrl + 1..6 -> Focus filter inputs
      if (event.ctrlKey && !event.altKey && !event.metaKey) {
        const key = event.key;
        const code = event.code;
        if (key === '1' || code === 'Digit1') {
          event.preventDefault();
          event.stopPropagation();
          filterFolderIdRef.current?.focus();
          filterFolderIdRef.current?.select?.();
          return;
        }
        if (key === '2' || code === 'Digit2') {
          event.preventDefault();
          event.stopPropagation();
          filterVideoIdRef.current?.focus();
          filterVideoIdRef.current?.select?.();
          return;
        }
        if (key === '3' || code === 'Digit3') {
          event.preventDefault();
          event.stopPropagation();
          filterTitleRef.current?.focus();
          filterTitleRef.current?.select?.();
          return;
        }
        if (key === '4' || code === 'Digit4') {
          event.preventDefault();
          event.stopPropagation();
          filterAsrRef.current?.focus();
          filterAsrRef.current?.select?.();
          return;
        }
        if (key === '5' || code === 'Digit5') {
          event.preventDefault();
          event.stopPropagation();
          filterOcrRef.current?.focus();
          filterOcrRef.current?.select?.();
          return;
        }
        if (key === '6' || code === 'Digit6') {
          event.preventDefault();
          event.stopPropagation();
          filterObjectRef.current?.focus();
          filterObjectRef.current?.select?.();
          return;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown, true);
    return () => window.removeEventListener('keydown', handleKeyDown, true);
  }, [
    isActive,
    isChatCollapsed,
    isOptionsCollapsed,
    handleToggleChat,
    handleToggleOptions,
    handleNewSearch,
    focusQueryInput,
  ]);

  const getFrameClassName = useCallback(
    (frameOrId) => {
      const frameId = typeof frameOrId === 'string' ? frameOrId : frameOrId?.frame_id;
      if (!frameId) return '';
      return activityStateForFrame(frameId, activeQuerySession?.frameActivity);
    },
    [activeQuerySession?.frameActivity],
  );

  const renderResults = () => {
    if (resultType?.startsWith('replay-')) {
      return (
        <ReplayResults
          resultSnapshot={replaySnapshot}
          frameActivity={activeQuerySession?.frameActivity}
          onFrameClick={openKisFrame}
        />
      );
    }
    const getFrameAnnotation = (frame) => {
      const snapshotId = liveEventTrailContextRef.current?.snapshotId;
      if (!frame?.result_id || !snapshotId) return 'unvisited';
      const key = `${snapshotId}:${frame.result_id}`;
      return eventTrailAnnotations?.[key] || 'unvisited';
    };

    return (
      <div className="frames-results-shell">
        <FramesBox
          results={frames}
          isLoading={isSearching}
          error={error}
          latencyMs={searchLatencyMs}
          warnings={warnings}
          events={kisEvents}
          onFrameClick={openKisFrame}
          onOpenSubmission={handleOpenSubmission}
          isSubmissionOpening={isSubmissionOpening}
          getFrameClassName={getFrameClassName}
          getFrameAnnotation={getFrameAnnotation}
          gridSize={gridSize}
        />
        {resultType === 'filter' && filterTotalPages > 1 && (
          <FilterPagination
            currentPage={filterPageId}
            totalPages={filterTotalPages}
            isLoading={isSearching}
            onPageChange={(nextPage) => submitFilter({ pageId: nextPage, overrideParams: appliedFilterParams })}
          />
        )}
      </div>
    );
  };

  const isReplay = resultType?.startsWith('replay-');

  const hasAnyFilterValue = Boolean(
    filterFolderId.trim()
    || filterVideoId.trim()
    || filterTitle.trim()
    || filterAsr.trim()
    || filterOcr.trim()
    || filterObject.trim(),
  );

  return (
    <div className="adhoc-workspace search-workspace">
      <form className="search-query-form" onSubmit={(e) => { e.preventDefault(); submitFilter(); }}>
        <div className="search-filter-row">
          <div className="search-filter-inputs-wrapper">
            <input
              ref={filterFolderIdRef}
              type="text"
              className="input-text search-filter-input"
              placeholder="Folder ID"
              aria-label="Filter Folder ID"
              title="Filter Folder ID (Ctrl+1)"
              value={filterFolderId}
              onChange={(e) => setFilterFolderId(e.target.value)}
              disabled={isSearching}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submitFilter({ pageId: 1 });
                }
              }}
            />
            <input
              ref={filterVideoIdRef}
              type="text"
              className="input-text search-filter-input"
              placeholder="Video ID"
              aria-label="Filter Video ID"
              title="Filter Video ID (Ctrl+2)"
              value={filterVideoId}
              onChange={(e) => setFilterVideoId(e.target.value)}
              disabled={isSearching}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submitFilter({ pageId: 1 });
                }
              }}
            />
            <input
              ref={filterTitleRef}
              type="text"
              className="input-text search-filter-input"
              placeholder="Title"
              aria-label="Filter Title"
              title="Filter Title (Ctrl+3)"
              value={filterTitle}
              onChange={(e) => setFilterTitle(e.target.value)}
              disabled={isSearching}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submitFilter({ pageId: 1 });
                }
              }}
            />
            <input
              ref={filterAsrRef}
              type="text"
              className="input-text search-filter-input"
              placeholder="ASR"
              aria-label="Filter ASR"
              title="Filter ASR (Ctrl+4)"
              value={filterAsr}
              onChange={(e) => setFilterAsr(e.target.value)}
              disabled={isSearching}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submitFilter({ pageId: 1 });
                }
              }}
            />
            <input
              ref={filterOcrRef}
              type="text"
              className="input-text search-filter-input"
              placeholder="OCR"
              aria-label="Filter OCR"
              title="Filter OCR (Ctrl+5)"
              value={filterOcr}
              onChange={(e) => setFilterOcr(e.target.value)}
              disabled={isSearching}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submitFilter({ pageId: 1 });
                }
              }}
            />
            <input
              ref={filterObjectRef}
              type="text"
              className="input-text search-filter-input"
              placeholder="Object (name: count)"
              aria-label="Filter Object"
              title="Filter Object (Ctrl+6)"
              value={filterObject}
              onChange={(e) => setFilterObject(e.target.value)}
              disabled={isSearching}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  submitFilter({ pageId: 1 });
                }
              }}
            />
          </div>
          <div className="search-filter-actions">
            <button
              type="submit"
              className="btn-primary search-filter-btn filter-submit-btn"
              disabled={isSearching || !hasAnyFilterValue}
            >
              Filter
            </button>
            <button
              type="button"
              className="btn-secondary search-filter-clear-btn filter-clear-btn"
              onClick={handleClearFilter}
              disabled={isSearching || !hasAnyFilterValue}
            >
              Clear
            </button>
          </div>
        </div>
      </form>
      <div className="adhoc-workspace-body">
        <aside className={`adhoc-sidebar ${isOptionsCollapsed ? 'collapsed' : ''}`}>
          {isOptionsCollapsed ? (
            <button
              type="button"
              className="adhoc-sidebar-expand-btn"
              onClick={() => handleToggleOptions(false)}
              title="Expand Options panel"
              aria-label="Expand Options panel"
            >
              <span className="sidebar-expand-icon">⚙</span>
              <span className="sidebar-expand-text">Options</span>
              <span className="sidebar-expand-arrow">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="sidebar-arrow-icon"
                  aria-hidden="true"
                >
                  <polyline points="9 18 15 12 9 6" />
                </svg>
              </span>
            </button>
          ) : (
            <>
              <div className="adhoc-sidebar-header">
                <h3 className="adhoc-sidebar-title">Options</h3>
                <button
                  type="button"
                  className="adhoc-sidebar-collapse-btn"
                  onClick={() => handleToggleOptions(true)}
                  title="Collapse Options panel"
                  aria-label="Collapse Options panel"
                >
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="sidebar-arrow-icon"
                    aria-hidden="true"
                  >
                    <polyline points="15 18 9 12 15 6" />
                  </svg>
                </button>
              </div>
              <ToolBox
                topK={topK}
                setTopK={setTopK}
                useDense={useDense}
                setUseDense={setUseDense}
                useBm25={useBm25}
                setUseBm25={setUseBm25}
                gridSize={gridSize}
                setGridSize={handleSetGridSize}
              />
            </>
          )}
        </aside>
        <div className="adhoc-results">
          {renderResults()}
        </div>
        <aside className={`kis-chat-sidebar ${isChatCollapsed ? 'collapsed' : ''}`} aria-label="KIS search">
          {isChatCollapsed ? (
            <button
              type="button"
              className="kis-chat-expand-btn"
              onClick={() => handleToggleChat(false)}
              title="Expand KIS search panel"
              aria-label="Expand KIS search panel"
            >
              <span className="sidebar-expand-icon">🔍</span>
              <span className="sidebar-expand-text">Search</span>
              <span className="sidebar-expand-arrow">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="sidebar-arrow-icon"
                  aria-hidden="true"
                >
                  <polyline points="15 18 9 12 15 6" />
                </svg>
              </span>
            </button>
          ) : (
            <KisPanel
              sessionState={{
                ...kisSession,
                isSearching: isSearching || kisSession.isSearching,
                error: error || kisSession.error,
              }}
              feedbackSession={feedbackSession}
              onUndoFeedback={handleUndoFeedback}
              onSelectContext={handleSelectContext}
              onClearContext={handleClearContext}
              inputRef={setQueryTextareaRef}
              onDraftChange={(val) => {
                setKisSession((prev) => setDraft(prev, val));
              }}
              onSubmit={submit}
              onReset={handleNewSearch}
              onAttachImage={handleAttachImage}
              onRemoveImage={handleRemoveImage}
              disabled={isReplay || isSearching}
              renderExtraActions={renderExtraActions}
              onCollapse={() => handleToggleChat(true)}
            />
          )}
        </aside>
      </div>
    </div>
  );
};

export default SearchWorkspace;
