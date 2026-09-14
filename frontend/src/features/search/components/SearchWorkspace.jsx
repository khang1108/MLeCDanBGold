/**
 * Query page orchestration for KIS retrieval and history sessions.
 *
 * Retrieval contracts remain owned by the existing API modules. This module
 * adds only history persistence, canonical activity tracking, and Replay.
 */
import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { searchFrames } from '../../../api/search';
import {
  createQueryHistory,
  markFrameViewed,
} from '../../../api/history';
import FramesBox from '../../frames/components/FramesBox';
import ToolBox from '../../search-controls/components/ToolBox';
import GifLoaderOverlay from '../../search/components/GifLoaderOverlay';
import ReplayResults from '../../workspace/components/ReplayResults';
import {
  buildKisSnapshot,
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

const SearchWorkspace = ({
  isActive = true,
  topK,
  setTopK,
  onFrameClick,
  onAddCandidate,
  onQueryChange,
  queryInputRef,
  onFocusQueryInput,
  onBlurQueryInput,
  userId,
  historyUserId,
  onHistoryRefresh,
  replayRequest,
  onExplorationInvalidated,
}) => {
  const historyIdentity = typeof historyUserId === 'string'
    ? historyUserId.trim()
    : typeof userId === 'string' ? userId.trim() : '';
  const [eventDescription, setEventDescription] = useState('');
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
  const queryTextareaRef = useRef(null);
  const requestRef = useRef(null);
  const viewedPatchRef = useRef(new Set());
  const lastReplayTokenRef = useRef(null);
  const liveKisSnapshotRef = useRef(null);
  const setQueryTextareaRef = useCallback((node) => {
    queryTextareaRef.current = node;
    if (queryInputRef) queryInputRef.current = node;
  }, [queryInputRef]);

  useLayoutEffect(() => {
    const textarea = queryTextareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.paddingTop = '8px';
    textarea.style.paddingBottom = '8px';
    textarea.style.lineHeight = '1.3';
    const contentHeight = textarea.scrollHeight;
    if (!/\r?\n/.test(textarea.value) && contentHeight <= 42) {
      textarea.style.height = '42px';
      textarea.style.paddingTop = '0px';
      textarea.style.paddingBottom = '0px';
      textarea.style.lineHeight = '42px';
    } else textarea.style.height = `${Math.max(contentHeight, 42)}px`;
  }, [eventDescription]);

  useEffect(() => {
    onQueryChange?.(eventDescription);
  }, [eventDescription, onQueryChange]);

  const recordViewed = useCallback((frame) => {
    const frameId = frame?.frame_id;
    const session = activeQuerySession;
    if (!frameId || !session?.queryId) return;
    const patchKey = `${session.queryId}:${frameId}`;
    if (viewedPatchRef.current.has(patchKey)) return;
    viewedPatchRef.current.add(patchKey);
    setActiveQuerySession((current) => current
      ? { ...current, frameActivity: withViewedFrame(current.frameActivity, frameId) }
      : current);
    markFrameViewed({ queryId: session.queryId, frameId }).catch((patchError) => {
      // Keep the optimistic color, but allow the next open of this frame to
      // retry the failed activity patch without touching submission state.
      viewedPatchRef.current.delete(patchKey);
      setWarnings((current) => Array.from(new Set([
        ...current,
        `History view state was not recorded: ${patchError.message || 'request failed'}`,
      ])));
    });
  }, [activeQuerySession]);

  const openCanonicalFrame = useCallback((frame) => {
    recordViewed(frame);
    onFrameClick?.({
      frame,
      ...(liveKisSnapshotRef.current
        ? { explorationSnapshot: liveKisSnapshotRef.current }
        : {}),
    });
  }, [onFrameClick, recordViewed]);

  const openKisFrame = useCallback((frame) => openCanonicalFrame(frame), [openCanonicalFrame]);

  useEffect(() => () => requestRef.current?.abort(), []);

  useEffect(() => {
    const item = replayRequest?.item || replayRequest;
    if (!item || !item.query_id || !item.result_snapshot) return;
    const token = replayRequest?.token || item.query_id;
    if (lastReplayTokenRef.current === token) return;
    lastReplayTokenRef.current = token;
    requestRef.current?.abort();
    liveKisSnapshotRef.current = null;
    onExplorationInvalidated?.();
    setIsSearching(false);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setSearchLatencyMs(null);
    setEventDescription(item.query_text || '');
    try {
      const kind = getSnapshotKind(item.result_snapshot);
      const normalizedActivity = normalizeFrameActivity(item.frame_activity);
      setReplaySnapshot(item.result_snapshot);
      setResultType(`replay-${kind}`);
      viewedPatchRef.current = new Set();
      setActiveQuerySession({
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
  }, [historyIdentity, onExplorationInvalidated, replayRequest]);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const rawEventText = eventDescription.trim();
    if (!rawEventText || isSearching) return;

    const capturedUserId = typeof userId === 'string' ? userId.trim() : '';
    const retrieval = parseRetrievalDescription(rawEventText);
    requestRef.current?.abort();
    liveKisSnapshotRef.current = null;
    onExplorationInvalidated?.();
    const controller = new AbortController();
    requestRef.current = controller;
    const queryId = historyIdentity ? createClientQueryId() : null;
    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setSearchLatencyMs(null);
    setResultType(null);
    setReplaySnapshot(null);
    setActiveQuerySession(null);
    lastReplayTokenRef.current = null;
    try {
      const response = await searchFrames({
        query: retrieval.query,
        topK,
        useDense,
        useBm25,
        signal: controller.signal,
        userId: capturedUserId,
      });
      if (controller.signal.aborted) return;
      const snapshotOptions = {
        events: response.events || [],
        latency: response.latency,
        warnings: response.warnings || [],
      };
      const historySnapshot = buildKisSnapshot(response.results || [], snapshotOptions);
      // Preserve the scoring-source fields from this response even if the
      // query draft changes before the user opens the frame inspector.
      const explorationSnapshot = {
        ...response,
        query: response.query || rawEventText,
        events: response.events || [],
        dense_events: response.dense_events,
        bm25_caption_events: response.bm25_caption_events,
        use_dense: typeof response.use_dense === 'boolean' ? response.use_dense : useDense,
        use_bm25: typeof response.use_bm25 === 'boolean' ? response.use_bm25 : useBm25,
      };
      // Older response shapes cannot reconstruct source events safely, so
      // they intentionally do not expose temporal exploration.
      const sourcesComplete = Array.isArray(explorationSnapshot.events)
        && explorationSnapshot.events.length > 0
        && (!explorationSnapshot.use_dense || (
          Array.isArray(explorationSnapshot.dense_events)
          && explorationSnapshot.dense_events.length === explorationSnapshot.events.length
        ))
        && (!explorationSnapshot.use_bm25 || (
          Array.isArray(explorationSnapshot.bm25_caption_events)
          && explorationSnapshot.bm25_caption_events.length === explorationSnapshot.events.length
        ));
      liveKisSnapshotRef.current = sourcesComplete ? explorationSnapshot : null;
      setResultType('retrieval');
      setFrames(response.results || []);
      setKisEvents(response.events || []);
      setSearchLatencyMs(response.latency);
      setWarnings(response.warnings || []);
      if (queryId) {
        try {
          await createQueryHistory({
            queryId,
            userId: historyIdentity,
            queryText: rawEventText,
            resultSnapshot: historySnapshot,
            signal: controller.signal,
          });
          if (controller.signal.aborted) return;
          viewedPatchRef.current = new Set();
          setActiveQuerySession({
            queryId,
            ownerUserId: historyIdentity,
            queryText: rawEventText,
            resultSnapshot: historySnapshot,
            frameActivity: normalizeFrameActivity(),
            source: 'live-search',
          });
          onHistoryRefresh?.();
        } catch (historyError) {
          if (historyError.name === 'AbortError') return;
          setWarnings((current) => [...current, `History was not saved: ${historyError.message || 'request failed'}`]);
        }
      }
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setResultType('retrieval');
      setError(requestError.message || 'Failed to contact search API');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [
    eventDescription,
    isSearching,
    onHistoryRefresh,
    onExplorationInvalidated,
    topK,
    useBm25,
    useDense,
    historyIdentity,
    userId,
  ]);

  const handleNewSearch = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    liveKisSnapshotRef.current = null;
    onExplorationInvalidated?.();
    setIsSearching(false);
    setEventDescription('');
    setFrames([]);
    setKisEvents([]);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
    setReplaySnapshot(null);
    setActiveQuerySession(null);
    lastReplayTokenRef.current = null;
  }, [onExplorationInvalidated]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
      if (event.key.toLowerCase() === 'n') {
        event.preventDefault();
        handleNewSearch();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNewSearch]);

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
    return (
      <FramesBox
        results={frames}
        isLoading={false}
        error={error}
        latencyMs={searchLatencyMs}
        warnings={warnings}
        events={kisEvents}
        onFrameClick={openKisFrame}
        onAddCandidate={onAddCandidate}
        getFrameClassName={getFrameClassName}
      />
    );
  };

  return (
    <div className="adhoc-workspace search-workspace">
      <form className="search-query-form" onSubmit={submit}>
        <div className="search-query-row">
          <div className="query-input-wrapper">
            <textarea
              ref={setQueryTextareaRef}
              id="event-query"
              className="input-text query-input-field"
              rows={1}
              value={eventDescription}
              onChange={(event) => setEventDescription(event.target.value)}
              placeholder="Describe a video moment to find (KIS or AVS)"
              onFocus={onFocusQueryInput}
              onBlur={onBlurQueryInput}
              disabled={isSearching}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  submit(event);
                }
              }}
            />
          </div>
          <div className="search-query-actions">
            <button type="submit" className="btn-primary query-submit-btn" disabled={isSearching || !eventDescription.trim()}>{isSearching ? 'Searching…' : 'Search'}</button>
            <button type="button" className="btn-secondary search-action-btn" onClick={handleNewSearch} title="Shortcut: N">New Search</button>
          </div>
        </div>
      </form>
      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox
            topK={topK}
            setTopK={setTopK}
            useDense={useDense}
            setUseDense={setUseDense}
            useBm25={useBm25}
            setUseBm25={setUseBm25}
            isActive={isActive}
          />
        </aside>
        <div className="adhoc-results">
          <GifLoaderOverlay isVisible={isSearching} />
          {!isSearching && renderResults()}
        </div>
      </div>
    </div>
  );
};

export default SearchWorkspace;
