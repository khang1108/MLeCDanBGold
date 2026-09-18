/**
 * Query page orchestration for KIS retrieval and multimodal search.
 *
 * Retrieval contracts remain owned by the existing API modules.
 */
import React, { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { searchKis, uploadKisImage } from '../../../api/kis';
import { searchAvs } from '../../../api/avs';
import { getCurrentDresTask } from '../../../api/submissions';
import {
  AvsHarvestGrid,
  AvsSelectionBar,
  AvsSelectionDrawer,
  AvsSubmitDialog,
  useAvsSubmission,
  avsSelectionReducer,
  candidateToTemporalAnswer,
  createInitialAvsSelectionState,
} from '../../avs';
import { openFeedbackSession, sendFeedbackTurn, undoFeedback } from '../../../api/feedback';
import {
  previewQueryHypothesis,
  commitQueryHypothesis,
  undoQueryHypothesis,
} from '../../../api/queryHypothesis';
import KisPanel from '../../kis/components/KisPanel';
import {
  createInitialQueryHypothesisState,
  receivePreview,
  clearPreview,
  receiveCommit,
  isResultsStale,
} from '../../kis/queryHypothesisSession';
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
import FramesBox from '../../frames/components/FramesBox';
import HcmusWatermarkBadge from '../../frames/components/HcmusWatermarkBadge';
import GifLoaderOverlay from './GifLoaderOverlay';
import ToolBox from '../../search-controls/components/ToolBox';

export const parseRetrievalDescription = (description) => {
  const query = description.trim();
  return query ? { query } : null;
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
  onQueryChange,
  isActive = true,
  onExplorationInvalidated,
  onEventTrailInvalidated,
  eventTrailAnnotations = {},
  eventTrail = null,
  workspaceMode = 'KIS',
  onToggleMode,
  selectedTask = null,
  setSelectedTask,
  evaluations = [],
  onSessionRejected,
  onAvsSelectionChange,
}) => {
  const notifyEventTrailInvalidated = useCallback(() => {
    onEventTrailInvalidated?.();
    onExplorationInvalidated?.();
  }, [onEventTrailInvalidated, onExplorationInvalidated]);
  const [kisSession, setKisSession] = useState(createInitialKisSessionState);
  const [queryHypothesisState, setQueryHypothesisState] = useState(createInitialQueryHypothesisState);
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
  const liveEventTrailContextRef = useRef(null);
  const prevControlsRef = useRef({ topK, useDense, useBm25 });
  const filterFolderIdRef = useRef(null);
  const filterVideoIdRef = useRef(null);
  const filterTitleRef = useRef(null);
  const filterAsrRef = useRef(null);
  const filterOcrRef = useRef(null);
  const filterObjectRef = useRef(null);
  const localQueryInputRef = useRef(null);

  // AVS State & Submissions
  const [avsResults, setAvsResults] = useState([]);
  const [avsResponse, setAvsResponse] = useState(null);
  const [selectionState, dispatchSelection] = useReducer(
    avsSelectionReducer,
    undefined,
    createInitialAvsSelectionState,
  );
  const [scopeConflict, setScopeConflict] = useState(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isSubmitDialogOpen, setIsSubmitDialogOpen] = useState(false);
  const [activeBatch, setActiveBatch] = useState(null);
  const pendingTaskSwitchRef = useRef(null);

  const avsSubmission = useAvsSubmission({
    userId,
    selectedTask,
    taskScopeKey: selectionState.taskScopeKey,
    onSessionRejected,
  });

  useEffect(() => {
    let isCancelled = false;

    const resolveScope = async () => {
      try {
        const dresTask = await getCurrentDresTask(userId, {
          evaluationId: selectedTask?.evaluationId,
          taskName: selectedTask?.taskName,
        });
        if (isCancelled) return;

        const liveScopeKey = dresTask?.task_scope_key || (selectedTask ? `${selectedTask.evaluationId}:${selectedTask.taskName}` : null);
        if (!liveScopeKey) return;

        if (pendingTaskSwitchRef.current) {
          pendingTaskSwitchRef.current = null;
          dispatchSelection({ type: 'RESET_FOR_SCOPE', taskScopeKey: liveScopeKey });
          setScopeConflict(null);
        } else if (selectionState.taskScopeKey === null) {
          dispatchSelection({ type: 'BIND_SCOPE', taskScopeKey: liveScopeKey });
          setScopeConflict(null);
        } else if (selectionState.taskScopeKey !== liveScopeKey) {
          if (selectionState.pending.size > 0) {
            setScopeConflict(
              `Task scope changed on the server (${liveScopeKey} != ${selectionState.taskScopeKey}). Selections are locked.`
            );
          } else {
            dispatchSelection({ type: 'RESET_FOR_SCOPE', taskScopeKey: liveScopeKey });
            setScopeConflict(null);
          }
        } else {
          setScopeConflict(null);
        }
      } catch (err) {
        if (!isCancelled && err?.status === 401) {
          onSessionRejected?.();
        }
      }
    };

    resolveScope();

    return () => {
      isCancelled = true;
    };
  }, [userId, selectedTask, selectionState.taskScopeKey, selectionState.pending.size, onSessionRejected]);

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

  const openCanonicalFrame = useCallback((frame) => {
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

    const eventText = frame?.eventText || frame?.event_text;
    onFrameClick?.({
      frame,
      ...(eventText ? { query: eventText } : {}),
      ...(trailContext ? { eventTrailContext: trailContext } : {}),
    });
  }, [onFrameClick, resultType]);

  const openKisFrame = useCallback((frame) => openCanonicalFrame(frame), [openCanonicalFrame]);

  const handleOpenSubmission = useCallback((payload) => {
    onOpenSubmission?.(payload);
  }, [onOpenSubmission]);

  const handleToggleCandidate = useCallback((candidate) => {
    dispatchSelection({ type: 'TOGGLE', candidate });
  }, []);

  const handleInspectCandidate = useCallback((candidate) => {
    openCanonicalFrame(candidate);
  }, [openCanonicalFrame]);

  const handleRemoveCandidate = useCallback((candidateId) => {
    dispatchSelection({ type: 'REMOVE', candidateId });
  }, []);

  const handleClearPending = useCallback(() => {
    dispatchSelection({ type: 'CLEAR_PENDING' });
  }, []);

  // Broadcast AVS selection state so App and ImageModal stay in sync
  useEffect(() => {
    onAvsSelectionChange?.({
      pending: selectionState.pending,
      submitted: selectionState.submitted,
      onToggle: handleToggleCandidate,
    });
  }, [selectionState.pending, selectionState.submitted, handleToggleCandidate, onAvsSelectionChange]);

  const handleOpenAvsSubmit = useCallback(() => {
    if (selectionState.pending.size === 0) return;
    setActiveBatch(null);
    setIsSubmitDialogOpen(true);
  }, [selectionState.pending.size]);

  const handleConfirmAvsSubmit = useCallback(async () => {
    const candidates = [...selectionState.pending.values()];
    const candidateIds = candidates.map((c) => c.candidate_id || c.frame_id);
    const answers = candidates.map((c) => candidateToTemporalAnswer(c));
    const batch = { candidateIds, answers };
    setActiveBatch(batch);

    const result = await avsSubmission.submitBatch(batch);

    if (result?.error?.code === 'TASK_SCOPE_MISMATCH' || result?.error?.status === 409) {
      setIsSubmitDialogOpen(false);
      setScopeConflict('Task scope changed on the server. Selections are locked.');
      return;
    }

    if (result?.state === 'RECORDED') {
      dispatchSelection({ type: 'RECORDED', candidateIds });
      setIsSubmitDialogOpen(false);
    } else if (result?.state === 'UNKNOWN') {
      dispatchSelection({ type: 'UNKNOWN', candidateIds });
    }
  }, [selectionState.pending, avsSubmission]);

  const handleRetryUnknown = useCallback(async () => {
    const confirmed = window.confirm('I verified DRES state and want to retry this exact batch.');
    if (confirmed && activeBatch) {
      const result = await avsSubmission.retryUnknown(activeBatch);
      if (result?.state === 'RECORDED') {
        dispatchSelection({ type: 'RECORDED', candidateIds: activeBatch.candidateIds });
        setIsSubmitDialogOpen(false);
      }
    }
  }, [activeBatch, avsSubmission]);

  const handleMarkUnknownRecorded = useCallback(() => {
    dispatchSelection({ type: 'MARK_UNKNOWN_RECORDED' });
    avsSubmission.resetOutcome();
    setIsSubmitDialogOpen(false);
  }, [avsSubmission]);

  const isSelectionDisabled = Boolean(scopeConflict) || Boolean(selectionState.unknownBatch);

  const handleFeedbackTurn = useCallback(async (message) => {
    const activeSnapshotId = liveEventTrailContextRef.current?.snapshotId || feedbackSession.evidenceSnapshotId;
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

      if (turnResp.status === 'proposal' || turnResp.query_proposal) {
        const action = turnResp.query_proposal?.action || turnResp.query_proposal;
        if (action) {
          if (queryHypothesisState.sessionId) {
            try {
              const previewRes = await previewQueryHypothesis(
                queryHypothesisState.sessionId,
                queryHypothesisState.queryRevision,
                action
              );
              setQueryHypothesisState((prev) =>
                receivePreview(prev, { ...previewRes, pendingAction: action })
              );
            } catch (prevErr) {
              if (turnResp.intent) {
                setQueryHypothesisState((prev) =>
                  receivePreview(prev, {
                    base_revision: queryHypothesisState.queryRevision,
                    intent: turnResp.intent,
                    pendingAction: action,
                  })
                );
              }
            }
          } else if (turnResp.intent) {
            setQueryHypothesisState((prev) =>
              receivePreview(prev, {
                base_revision: queryHypothesisState.queryRevision,
                intent: turnResp.intent,
                pendingAction: action,
              })
            );
          }
        }
      } else if (turnResp.status === 'applied') {
        notifyEventTrailInvalidated();
        if (Array.isArray(turnResp.results)) {
          setFrames([...turnResp.results]);
          if (turnResp.latency) {
            setSearchLatencyMs(turnResp.latency);
          }
        }
        if (turnResp.evidence_snapshot_id) {
          if (!liveEventTrailContextRef.current) {
            liveEventTrailContextRef.current = {
              snapshotId: turnResp.evidence_snapshot_id,
              kisRevision: kisSession.currentIntent?.revision ?? 1,
              events: (kisSession.currentIntent?.events || []).map(({ id, text, images }) => ({
                id,
                text,
                images: images || [],
              })),
              searchSessionId: turnResp.evidence_snapshot_id,
            };
          } else {
            liveEventTrailContextRef.current.snapshotId = turnResp.evidence_snapshot_id;
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
    queryHypothesisState.sessionId,
    queryHypothesisState.queryRevision,
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
      }
      if (undoResp.evidence_snapshot_id) {
        if (liveEventTrailContextRef.current) {
          liveEventTrailContextRef.current.snapshotId = undoResp.evidence_snapshot_id;
        }
      }
    } catch (undoErr) {
      setFeedbackSession((prev) => applyFeedbackFailure(prev, undoErr, { requestId }));
      setError(undoErr?.message || 'Feedback undo failed');
    } finally {
      setIsSearching(false);
    }
  }, [feedbackSession.sessionId, feedbackSession.canUndo, feedbackSession.feedbackRevision, isSearching, notifyEventTrailInvalidated]);

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

  const handlePreviewQueryHypothesis = useCallback(async (action) => {
    if (!queryHypothesisState.sessionId) return;
    try {
      const previewRes = await previewQueryHypothesis(
        queryHypothesisState.sessionId,
        queryHypothesisState.queryRevision,
        action,
      );
      setQueryHypothesisState((prev) => receivePreview(prev, { ...previewRes, pendingAction: action }));
    } catch (err) {
      setError(err.message || 'Failed to preview query hypothesis action');
    }
  }, [queryHypothesisState.sessionId, queryHypothesisState.queryRevision]);

  const handleCancelPreviewQueryHypothesis = useCallback(() => {
    setQueryHypothesisState((prev) => clearPreview(prev));
  }, []);

  const handleCommitQueryHypothesis = useCallback(async (actionOrPreview) => {
    if (!queryHypothesisState.sessionId) return;
    const action = actionOrPreview?.pendingAction || actionOrPreview?.action || actionOrPreview;
    try {
      setIsSearching(true);
      const commitRes = await commitQueryHypothesis(
        queryHypothesisState.sessionId,
        queryHypothesisState.queryRevision,
        action,
      );
      setQueryHypothesisState((prev) => receiveCommit(prev, commitRes));
      setKisSession((prev) => ({
        ...prev,
        revision: commitRes.query_revision ?? commitRes.intent?.revision ?? prev.revision,
        currentIntent: commitRes.intent ?? prev.currentIntent,
      }));
    } catch (err) {
      setError(err.message || 'Failed to commit query hypothesis');
    } finally {
      setIsSearching(false);
    }
  }, [queryHypothesisState.sessionId, queryHypothesisState.queryRevision]);

  const handleUndoQueryHypothesis = useCallback(async () => {
    if (!queryHypothesisState.sessionId || !queryHypothesisState.canUndo) return;
    try {
      setIsSearching(true);
      const undoRes = await undoQueryHypothesis(
        queryHypothesisState.sessionId,
        queryHypothesisState.queryRevision,
      );
      setQueryHypothesisState((prev) => receiveCommit(prev, undoRes));
      setKisSession((prev) => ({
        ...prev,
        revision: undoRes.query_revision ?? undoRes.intent?.revision ?? prev.revision,
        currentIntent: undoRes.intent ?? prev.currentIntent,
      }));
    } catch (err) {
      setError(err.message || 'Failed to undo query hypothesis');
    } finally {
      setIsSearching(false);
    }
  }, [queryHypothesisState.sessionId, queryHypothesisState.queryRevision, queryHypothesisState.canUndo]);

  const submit = useCallback(async (event) => {
    event?.preventDefault?.();
    if (isSearching) return;

    const draftText = (kisSession.draft || '').trim();

    if (workspaceMode === 'AVS') {
      if (!draftText) return;

      requestRef.current?.abort();
      const controller = new AbortController();
      requestRef.current = controller;

      setIsSearching(true);
      setError(null);
      setWarnings([]);
      setResultType('avs');
      onQueryChange?.(draftText);

      const startTime = performance.now();
      try {
        const response = await searchAvs({
          query: draftText,
          pageSize: topK || 80,
          userId: typeof userId === 'string' ? userId.trim() : undefined,
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        const elapsed = Math.round(performance.now() - startTime);

        setAvsResults(response.results || []);
        setAvsResponse(response);
        setSearchLatencyMs(elapsed);
        setKisSession((prev) => ({ ...prev, draft: '' }));

        setFeedbackSession({
          ...createInitialFeedbackSession(),
          status: 'applied',
          messages: [
            {
              id: `msg_avs_${Date.now()}`,
              role: 'user',
              text: draftText,
            },
            {
              id: `msg_avs_resp_${Date.now()}`,
              role: 'assistant',
              text: `Found ${response.results?.length || 0} candidate keyframes across ${response.unique_videos || 0} videos. Use checkboxes in the grid to harvest candidates into your basket.`,
            },
          ],
          results: response.results || [],
          canUndo: false,
        });
      } catch (err) {
        if (err.name === 'AbortError') return;
        setAvsResults([]);
        setAvsResponse(null);
        setError(err.message || 'AVS Search failed');
      } finally {
        if (requestRef.current === controller) {
          requestRef.current = null;
          setIsSearching(false);
        }
      }
      return;
    }

    let prepared;
    let queryHypothesisSessionId = queryHypothesisState.sessionId || null;
    const hasStagedImages = Object.keys(kisSession.stagedImages || {}).length > 0;

    const capturedUserId = typeof userId === 'string' ? userId.trim() : '';
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

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
      } else if (queryHypothesisState.sessionId && isResultsStale(queryHypothesisState)) {
        queryHypothesisSessionId = queryHypothesisState.sessionId;
        prepared = {
          nextState: {
            ...kisSession,
            isSearching: true,
            error: null,
            pendingOperation: { kind: 'search_only' },
          },
          requestPayload: {
            baseIntent: null,
            expectedRevision: queryHypothesisState.queryRevision,
            operation: { kind: 'search_only' },
          },
        };
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

    setIsSearching(true);
    setError(null);
    try {
      const response = await searchKis({
        queryHypothesisSessionId,
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
      setQueryHypothesisState((prev) => {
        const nextRev = response.intent?.revision ?? requestPayload.expectedRevision ?? 1;
        return {
          ...prev,
          sessionId: response.query_hypothesis_session_id || prev.sessionId || (response.evidence_snapshot_id ? `qh_${response.evidence_snapshot_id}` : 'qh_1'),
          intent: response.intent || prev.intent,
          queryRevision: nextRev,
          resultsQueryRevision: nextRev,
          preview: null,
          error: null,
        };
      });

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
          if (patch.added_images?.length) {
            imageAdded.push(...patch.added_images);
          }
          if (patch.removed_image_ids?.length) {
            imageRemoved.push(...patch.removed_image_ids);
          }
        }
      }
      if (
        requestPayload.operation?.kind === 'initial_resolve'
        && Array.isArray(requestPayload.operation.events)
      ) {
        for (const ev of requestPayload.operation.events) {
          if (ev.images?.length) {
            imageAdded.push(...ev.images);
          }
        }
      }

      const snapshotId = response.evidence_snapshot_id || response.search_session_id;
      if (snapshotId) {
        liveEventTrailContextRef.current = {
          snapshotId,
          kisRevision: response.intent?.revision ?? 1,
          events: (response.intent?.events || []).map(({ id, text, images }) => ({
            id,
            text,
            images: images || [],
          })),
          searchSessionId: response.search_session_id || snapshotId,
        };
      } else {
        liveEventTrailContextRef.current = null;
      }
      notifyEventTrailInvalidated();

      setFrames(response.results || []);
      setKisEvents(eventTexts);
      setWarnings(response.warnings || []);
      setSearchLatencyMs(response.latency?.total_ms ?? null);
      setResultType('retrieval');
      onQueryChange?.(queryText);
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
    isSearching,
    kisSession,
    queryHypothesisState,
    notifyEventTrailInvalidated,
    onQueryChange,
    topK,
    useBm25,
    useDense,
    userId,
    handleFeedbackTurn,
    workspaceMode,
  ]);

  const handleSearchQueryHypothesis = useCallback(() => {
    submit();
  }, [submit]);

  // Step 7: Search-only rerun when only retrieval controls change
  useEffect(() => {
    const prev = prevControlsRef.current;
    const controlsChanged = prev.topK !== topK || prev.useDense !== useDense || prev.useBm25 !== useBm25;
    prevControlsRef.current = { topK, useDense, useBm25 };

    if (controlsChanged && kisSession.currentIntent && !kisSession.draft?.trim() && !isSearching && workspaceMode === 'KIS') {
      setFeedbackSession(resetFeedbackSession());
      submit();
    }
  }, [topK, useDense, useBm25, kisSession.currentIntent, kisSession.draft, isSearching, submit, workspaceMode]);

  const handleNewSearch = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    liveEventTrailContextRef.current = null;
    notifyEventTrailInvalidated();
    setIsSearching(false);
    setKisSession(resetKisSession());
    setQueryHypothesisState(createInitialQueryHypothesisState());
    setFeedbackSession(resetFeedbackSession());
    handleClearFilter();
    setFrames([]);
    setKisEvents([]);
    setAvsResults([]);
    setAvsResponse(null);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
    onQueryChange?.('');
  }, [handleClearFilter, notifyEventTrailInvalidated, onQueryChange]);

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

      // 6. Ctrl + Shift + 1 / 2 -> Switch search mode (1: KIS, 2: AVS)
      if (event.ctrlKey && event.shiftKey && !event.altKey && !event.metaKey) {
        const key = event.key;
        const code = event.code;
        if (key === '1' || key === '!' || code === 'Digit1' || code === 'Numpad1') {
          event.preventDefault();
          event.stopPropagation();
          onToggleMode?.('KIS');
          return;
        }
        if (key === '2' || key === '@' || code === 'Digit2' || code === 'Numpad2') {
          event.preventDefault();
          event.stopPropagation();
          onToggleMode?.('AVS');
          return;
        }
      }

      // 7. Ctrl + 1..6 -> Focus filter inputs
      if (event.ctrlKey && !event.shiftKey && !event.altKey && !event.metaKey) {
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
    onToggleMode,
  ]);

  const getFrameClassName = useCallback(() => '', []);

  const renderResults = () => {
    if (workspaceMode === 'AVS') {
      const hasSearchedAvs = Boolean(avsResponse || avsResults.length > 0);

      return (
        <div className={`frames-results-shell avs-workspace-results ${isSearching ? 'whip-cursor-mode' : ''}`}>
          {scopeConflict && (
            <div className="avs-status-message error" role="alert">
              {scopeConflict}
            </div>
          )}

          {isSearching ? (
            <div className="avs-loader-container">
              <GifLoaderOverlay isVisible={true} />
            </div>
          ) : error ? (
            <div className="avs-status-message error" role="alert">
              {error}
            </div>
          ) : avsResults.length === 0 ? (
            hasSearchedAvs ? (
              <div className="avs-status-message empty">
                No candidates found matching your query.
              </div>
            ) : (
              <div className="avs-watermark-container">
                <HcmusWatermarkBadge />
              </div>
            )
          ) : (
            <>
              <div className="avs-results-summary">
                Showing {avsResults.length} candidates from{' '}
                {avsResponse?.unique_videos ?? 0} videos (pool size:{' '}
                {avsResponse?.candidate_pool_size ?? avsResults.length})
              </div>
              <AvsHarvestGrid
                candidates={avsResults}
                pending={selectionState.pending}
                submitted={selectionState.submitted}
                selectionDisabled={isSelectionDisabled}
                onToggle={handleToggleCandidate}
                onInspect={handleInspectCandidate}
              />
            </>
          )}

          <AvsSelectionBar
            pending={selectionState.pending}
            onReview={() => setIsDrawerOpen(true)}
            onClear={handleClearPending}
            onSubmit={handleOpenAvsSubmit}
            isSubmitting={avsSubmission.status === 'SUBMITTING'}
            disabled={isSelectionDisabled}
          />

          <AvsSelectionDrawer
            isOpen={isDrawerOpen}
            onClose={() => setIsDrawerOpen(false)}
            pending={selectionState.pending}
            onRemove={handleRemoveCandidate}
            onInspect={handleInspectCandidate}
            disabled={isSelectionDisabled}
          />

          <AvsSubmitDialog
            isOpen={isSubmitDialogOpen}
            onClose={() => {
              setIsSubmitDialogOpen(false);
              avsSubmission.resetOutcome();
            }}
            onConfirm={handleConfirmAvsSubmit}
            candidateCount={activeBatch ? activeBatch.candidateIds.length : selectionState.pending.size}
            uniqueVideoCount={
              activeBatch
                ? new Set(activeBatch.answers.map((a) => a.video_id)).size
                : new Set([...selectionState.pending.values()].map((c) => c.video_id)).size
            }
            isSubmitting={avsSubmission.status === 'SUBMITTING'}
            status={avsSubmission.status}
            outcome={avsSubmission.outcome}
            onRetryUnknown={handleRetryUnknown}
            onMarkUnknownRecorded={handleMarkUnknownRecorded}
          />
        </div>
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
        {isResultsStale(queryHypothesisState) && (
          <div
            className="search-stale-banner alert alert-warning"
            data-testid="stale-results-notice"
            role="alert"
          >
            <span>Query hypothesis has changed since these results were retrieved.</span>
            <button
              type="button"
              className="btn btn-sm btn-warning search-stale-update-btn"
              onClick={() => submit()}
              disabled={isSearching}
            >
              Update Results
            </button>
          </div>
        )}
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
          eventTrail={eventTrail}
          eventTrailContext={liveEventTrailContextRef.current}
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
                onOpenFrame={onFrameClick}
                workspaceMode={workspaceMode}
                onToggleMode={onToggleMode}
                evaluations={evaluations}
                selectedTask={selectedTask}
                setSelectedTask={setSelectedTask}
                connectedUserId={userId}
              />
            </>
          )}
        </aside>
        <div className="adhoc-results">
          {renderResults()}
        </div>
        <aside className={`kis-chat-sidebar ${isChatCollapsed ? 'collapsed' : ''}`} aria-label={workspaceMode === 'AVS' ? 'AVS search' : 'KIS search'}>
          {isChatCollapsed ? (
            <button
              type="button"
              className="kis-chat-expand-btn"
              onClick={() => handleToggleChat(false)}
              title={workspaceMode === 'AVS' ? 'Expand AVS search panel' : 'Expand KIS search panel'}
              aria-label={workspaceMode === 'AVS' ? 'Expand AVS search panel' : 'Expand KIS search panel'}
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
              queryHypothesisState={queryHypothesisState}
              onPreviewQueryHypothesis={handlePreviewQueryHypothesis}
              onCommitQueryHypothesis={handleCommitQueryHypothesis}
              onCancelPreviewQueryHypothesis={handleCancelPreviewQueryHypothesis}
              onUndoQueryHypothesis={handleUndoQueryHypothesis}
              onSearchQueryHypothesis={handleSearchQueryHypothesis}
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
              disabled={isSearching}
              renderExtraActions={renderExtraActions}
              onCollapse={() => handleToggleChat(true)}
              mode={workspaceMode}
            />
          )}
        </aside>
      </div>
    </div>
  );
};

export default SearchWorkspace;
