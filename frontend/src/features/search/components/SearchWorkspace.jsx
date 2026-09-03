/**
 * Query page orchestration for KIS/TRAKE retrieval and history sessions.
 *
 * Retrieval contracts remain owned by the existing API modules. This module
 * adds only history persistence, canonical activity tracking, and Replay.
 */
import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { searchFrames, searchTrake } from '../../../api/search';
import {
  createQueryHistory,
  markFrameViewed,
} from '../../../api/workspace';
import FramesBox from '../../frames/components/FramesBox';
import ToolBox from '../../search-controls/components/ToolBox';
import GifLoaderOverlay from '../../search/components/GifLoaderOverlay';
import { displayVideoId } from '../../frames/videoSource';
import { useSubmissionDialog } from '../../submission/contexts/SubmissionDialogContext';
import ReplayResults from '../../workspace/components/ReplayResults';
import {
  buildKisSnapshot,
  buildTrakeSnapshot,
  getSnapshotKind,
  normalizeFrameActivity,
  withViewedFrame,
  withSubmittedFrames,
  activityStateForFrame,
} from '../../workspace/queryHistory';
import TrakeResults from './TrakeResults';

export { TrakeResults };

const TRAKE_EVENT_MARKER = /\bE(\d+)\b\s*:?\s*/gi;

export const parseRetrievalDescription = (description) => {
  const query = description.trim();
  return query ? { query } : null;
};

export const parseTrakeEvents = (description) => {
  const text = description.trim();
  const markers = Array.from(text.matchAll(TRAKE_EVENT_MARKER));
  if (!markers.length || Number(markers[0][1]) !== 1) return null;
  const events = markers.map((marker, index) => {
    if (Number(marker[1]) !== index + 1) return null;
    const start = marker.index + marker[0].length;
    const end = markers[index + 1]?.index ?? text.length;
    return text.slice(start, end).trim() || null;
  });
  return events.some((event) => !event) ? [] : events;
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
  onQueryChange,
  queryInputRef,
  onFocusQueryInput,
  onBlurQueryInput,
  userId,
  onFocusUserId,
  onHistoryRefresh,
  replayRequest,
}) => {
  const [eventDescription, setEventDescription] = useState('');
  const [useDense, setUseDense] = useState(true);
  const [useBm25, setUseBm25] = useState(true);
  const [resultType, setResultType] = useState(null);
  const [frames, setFrames] = useState([]);
  const [kisEvents, setKisEvents] = useState([]);
  const [paths, setPaths] = useState([]);
  const [trakeEvents, setTrakeEvents] = useState([]);
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
  const { requestSubmission } = useSubmissionDialog();

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

  const historyForSession = useCallback((session, frameIds) => {
    if (!session?.queryId) return undefined;
    return { queryId: session.queryId, frameIds };
  }, []);

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

  const openCanonicalFrame = useCallback((frame, submissionMode = 'none') => {
    recordViewed(frame);
    onFrameClick?.({
      frame,
      submissionMode,
      history: historyForSession(activeQuerySession, [frame.frame_id]),
    });
  }, [activeQuerySession, historyForSession, onFrameClick, recordViewed]);

  const openKisFrame = useCallback((frame) => openCanonicalFrame(frame, 'kis'), [openCanonicalFrame]);
  const openTrakeFrame = useCallback((frame) => openCanonicalFrame(frame, 'none'), [openCanonicalFrame]);

  const handleTrakeSubmit = useCallback((path) => {
    const vid = displayVideoId(path.video_id);
    requestSubmission({
      line: `${vid},${path.frame_idxs.join(',')}`,
      source: 'TRAKE path',
      history: historyForSession(activeQuerySession, path.frame_ids),
    });
  }, [activeQuerySession, historyForSession, requestSubmission]);

  const handleFrameSubmit = useCallback((frame) => {
    const vid = displayVideoId(frame.video_id);
    requestSubmission({
      line: `${vid},${frame.frame_idx}`,
      source: 'KIS/TRAKE frame',
      history: historyForSession(activeQuerySession, [frame.frame_id]),
    });
  }, [activeQuerySession, historyForSession, requestSubmission]);

  const handleReplayFrameClick = useCallback((frame, submissionMode = 'kis') => openCanonicalFrame(frame, submissionMode), [openCanonicalFrame]);
  const handleReplayFrameSubmit = useCallback((frame) => handleFrameSubmit(frame), [handleFrameSubmit]);
  const handleReplayPathSubmit = useCallback((path) => handleTrakeSubmit(path), [handleTrakeSubmit]);

  useEffect(() => () => requestRef.current?.abort(), []);

  useEffect(() => {
    const handleHistoryChanged = (event) => {
      const { queryId, frameIds } = event?.detail || {};
      if (!queryId || !Array.isArray(frameIds) || frameIds.length === 0) return;
      setActiveQuerySession((current) => {
        if (!current || current.queryId !== queryId) return current;
        return {
          ...current,
          frameActivity: withSubmittedFrames(current.frameActivity, frameIds),
        };
      });
    };
    window.addEventListener('hcmai:history-changed', handleHistoryChanged);
    return () => window.removeEventListener('hcmai:history-changed', handleHistoryChanged);
  }, []);

  useEffect(() => {
    const item = replayRequest?.item || replayRequest;
    if (!item || !item.query_id || !item.result_snapshot) return;
    const token = replayRequest?.token || item.query_id;
    if (lastReplayTokenRef.current === token) return;
    lastReplayTokenRef.current = token;
    requestRef.current?.abort();
    setIsSearching(false);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setPaths([]);
    setTrakeEvents([]);
    setSearchLatencyMs(null);
    setEventDescription(item.query_text || '');
    try {
      const kind = getSnapshotKind(item.result_snapshot);
      const normalizedActivity = normalizeFrameActivity(item.frame_activity);
      setReplaySnapshot(item.result_snapshot);
      setResultType(kind === 'kis' ? 'replay-kis' : 'replay-trake');
      viewedPatchRef.current = new Set();
      setActiveQuerySession({
        queryId: item.query_id,
        ownerUserId: userId?.trim() || '',
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
  }, [replayRequest, userId]);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const rawEventText = eventDescription.trim();
    if (!rawEventText || isSearching) return;
    if (typeof userId === 'string' && !userId.trim()) {
      setError('Enter a User ID before searching.');
      onFocusUserId?.();
      return;
    }

    const capturedUserId = typeof userId === 'string' ? userId.trim() : null;
    const events = parseTrakeEvents(rawEventText);
    const isTrakeMode = events !== null;
    const retrieval = isTrakeMode ? null : parseRetrievalDescription(rawEventText);
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const queryId = capturedUserId ? createClientQueryId() : null;
    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setPaths([]);
    setTrakeEvents([]);
    setSearchLatencyMs(null);
    setResultType(null);
    setReplaySnapshot(null);
    setActiveQuerySession(null);
    lastReplayTokenRef.current = null;
    try {
      const response = isTrakeMode
        ? await searchTrake({
          events,
          topK,
          useDense,
          useBm25,
          signal: controller.signal,
        })
        : await searchFrames({
          query: retrieval.query,
          topK,
          useDense,
          useBm25,
          signal: controller.signal,
        });
      if (controller.signal.aborted) return;
      const snapshotOptions = {
        events: response.events || events || [],
        latency: response.latency,
        warnings: response.warnings || [],
      };
      const snapshot = isTrakeMode
        ? buildTrakeSnapshot(response.paths || [], snapshotOptions)
        : buildKisSnapshot(response.results || [], snapshotOptions);
      if (isTrakeMode) {
        setResultType('trake');
        setPaths(response.paths || []);
        setTrakeEvents(response.events || events);
      } else {
        setResultType('retrieval');
        setFrames(response.results || []);
        setKisEvents(response.events || []);
        setSearchLatencyMs(response.latency);
      }
      setWarnings(response.warnings || []);
      if (queryId) {
        try {
          await createQueryHistory({
            queryId,
            userId: capturedUserId,
            queryText: rawEventText,
            resultSnapshot: snapshot,
            signal: controller.signal,
          });
          if (controller.signal.aborted) return;
          viewedPatchRef.current = new Set();
          setActiveQuerySession({
            queryId,
            ownerUserId: capturedUserId,
            queryText: rawEventText,
            resultSnapshot: snapshot,
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
      setResultType(isTrakeMode ? 'trake' : 'retrieval');
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
    onFocusUserId,
    onHistoryRefresh,
    topK,
    useBm25,
    useDense,
    userId,
  ]);

  const handleNewSearch = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    setIsSearching(false);
    setEventDescription('');
    setFrames([]);
    setKisEvents([]);
    setPaths([]);
    setTrakeEvents([]);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
    setReplaySnapshot(null);
    setActiveQuerySession(null);
    lastReplayTokenRef.current = null;
  }, []);

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
    if (resultType === 'replay-kis' || resultType === 'replay-trake') {
      return (
        <ReplayResults
          resultSnapshot={replaySnapshot}
          frameActivity={activeQuerySession?.frameActivity}
          onFrameClick={handleReplayFrameClick}
          onFrameSubmit={handleReplayFrameSubmit}
          onPathSubmit={handleReplayPathSubmit}
        />
      );
    }
    if (resultType === 'trake') {
      return (
        <TrakeResults
          events={trakeEvents}
          paths={paths}
          warnings={warnings}
          error={error}
          hasSearched
          onFrameClick={openTrakeFrame}
          onTrakeSubmit={handleTrakeSubmit}
          getFrameClassName={getFrameClassName}
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
        onSubmit={handleFrameSubmit}
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
              placeholder="Describe the event, or add E1, E2, ... for TRAKE"
              onFocus={onFocusQueryInput}
              onBlur={onBlurQueryInput}
              disabled={isSearching}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey && parseTrakeEvents(eventDescription) === null) {
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
            includeSubmissionWorktree={isActive}
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
