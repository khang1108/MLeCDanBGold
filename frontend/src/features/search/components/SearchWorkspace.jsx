/**
 * Query page orchestration for KIS retrieval and history sessions.
 *
 * Retrieval contracts remain owned by the existing API modules. This module
 * adds only history persistence, canonical activity tracking, and Replay.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { searchFramesByImage } from '../../../api/search';
import { searchKis } from '../../../api/kis';
import KisPanel from '../../kis/components/KisPanel';
import {
  createInitialKisSessionState,
  setDraft,
  prepareSearchRequest,
  commitSearchSuccess,
  commitSearchFailure,
  resetKisSession,
} from '../../kis/session';
import { filterFrames } from '../../../api/filter';
import FilterPagination from '../../filter/components/FilterPagination';
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

const formatFileSize = (bytes) => {
  if (!bytes || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
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
  const [kisSession, setKisSession] = useState(createInitialKisSessionState);
  const eventDescription = kisSession.draft;
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
  const [selectedImageFile, setSelectedImageFile] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [isImageDragOver, setIsImageDragOver] = useState(false);
  const [filterFolderId, setFilterFolderId] = useState('');
  const [filterVideoId, setFilterVideoId] = useState('');
  const [filterTitle, setFilterTitle] = useState('');
  const [filterAsr, setFilterAsr] = useState('');
  const [filterOcr, setFilterOcr] = useState('');
  const [filterObject, setFilterObject] = useState('');
  const [filterPageId, setFilterPageId] = useState(1);
  const [filterTotalPages, setFilterTotalPages] = useState(0);
  const [appliedFilterParams, setAppliedFilterParams] = useState(null);
  const imageInputRef = useRef(null);
  const requestRef = useRef(null);
  const viewedPatchRef = useRef(new Set());
  const lastReplayTokenRef = useRef(null);
  const liveKisSnapshotRef = useRef(null);
  const setQueryTextareaRef = useCallback((node) => {
    if (queryInputRef) queryInputRef.current = node;
  }, [queryInputRef]);

  useEffect(() => {
    if (!selectedImageFile) {
      setImagePreviewUrl(null);
      return undefined;
    }
    if (typeof URL.createObjectURL === 'function') {
      const objectUrl = URL.createObjectURL(selectedImageFile);
      setImagePreviewUrl(objectUrl);
      return () => {
        if (typeof URL.revokeObjectURL === 'function') {
          URL.revokeObjectURL(objectUrl);
        }
      };
    }
    return undefined;
  }, [selectedImageFile]);

  const handleImageFileSelect = useCallback((file) => {
    if (!file) return;
    setSelectedImageFile(file);
    setError(null);
  }, []);

  const handleClearImageFile = useCallback((e) => {
    e?.stopPropagation?.();
    setSelectedImageFile(null);
    if (imageInputRef.current) imageInputRef.current.value = '';
  }, []);

  const handleImageDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsImageDragOver(true);
  };

  const handleImageDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsImageDragOver(false);
  };

  const handleImageDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsImageDragOver(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleImageFileSelect(files[0]);
    }
  };

  useEffect(() => {
    if (!isActive) return undefined;

    const handlePaste = (event) => {
      const items = event.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i += 1) {
        const item = items[i];
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            event.preventDefault();
            const ext = file.type.split('/')[1] || 'png';
            const fallbackName = `pasted-image-${Date.now()}.${ext}`;
            const namedFile = file.name && file.name !== 'image.png'
              ? file
              : new File([file], fallbackName, { type: file.type });
            handleImageFileSelect(namedFile);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [handleImageFileSelect, isActive]);

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
    onFrameClick?.({ frame });
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
    setKisSession(setDraft(createInitialKisSessionState(), item.query_text || ''));
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
  }, [replayRequest, historyIdentity, onExplorationInvalidated]);

  const submitImageSearch = useCallback(async (event) => {
    event?.preventDefault?.();
    if (!selectedImageFile || isSearching) return;

    requestRef.current?.abort();
    liveKisSnapshotRef.current = null;
    onExplorationInvalidated?.();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setSearchLatencyMs(null);
    setResultType('image-retrieval');
    setReplaySnapshot(null);
    setActiveQuerySession(null);
    lastReplayTokenRef.current = null;

    try {
      const response = await searchFramesByImage({
        imageFile: selectedImageFile,
        topK,
        signal: controller.signal,
        userId: typeof userId === 'string' ? userId.trim() : '',
      });

      if (controller.signal.aborted) return;

      setFrames(response.results || []);
      setSearchLatencyMs(response.latency);
      setWarnings(response.warnings || []);
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setError(requestError.message || 'Failed to contact image search API');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [isSearching, onExplorationInvalidated, selectedImageFile, topK, userId]);

  const submit = useCallback(async (event) => {
    event?.preventDefault?.();
    if (isSearching) return;

    if (selectedImageFile) {
      return submitImageSearch(event);
    }

    const rawEventText = (kisSession.draft || '').trim();
    if (!rawEventText) return;

    let prepared;
    try {
      prepared = prepareSearchRequest(kisSession);
    } catch {
      return;
    }

    const { nextState, requestPayload } = prepared;
    setKisSession(nextState);

    const capturedUserId = typeof userId === 'string' ? userId.trim() : '';
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
      const response = await searchKis({
        inputs: requestPayload.inputs,
        expectedRevision: requestPayload.expectedRevision,
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
      const queryText = response.intent?.query_text || rawEventText;

      const snapshotOptions = {
        intent: response.intent,
        latency: response.latency,
        warnings: response.warnings || [],
      };
      const historySnapshot = buildKisSnapshot(response.results || [], snapshotOptions);
      // Preserve the scoring-source fields from this response even if the
      // query draft changes before the user opens the frame inspector.
      const explorationSnapshot = {
        ...response,
        query: queryText,
        events: eventTexts,
        dense_events: response.dense_events,
        bm25_caption_events: response.bm25_events,
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
      setKisEvents(eventTexts);
      setSearchLatencyMs(response.latency);
      setWarnings(response.warnings || []);
      if (queryId) {
        try {
          await createQueryHistory({
            queryId,
            userId: historyIdentity,
            queryText,
            resultSnapshot: historySnapshot,
            signal: controller.signal,
          });
          if (controller.signal.aborted) return;
          viewedPatchRef.current = new Set();
          setActiveQuerySession({
            queryId,
            ownerUserId: historyIdentity,
            queryText,
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
      setKisSession((prev) => commitSearchFailure(prev, requestError));
      setResultType('retrieval');
      setError(requestError.message || 'Search failed');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [
    historyIdentity,
    isSearching,
    kisSession,
    onExplorationInvalidated,
    onHistoryRefresh,
    selectedImageFile,
    submitImageSearch,
    topK,
    useBm25,
    useDense,
    userId,
  ]);

  const hasAnyFilterValue = Boolean(
    filterFolderId.trim()
    || filterVideoId.trim()
    || filterTitle.trim()
    || filterAsr.trim()
    || filterOcr.trim()
    || filterObject.trim(),
  );

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
    liveKisSnapshotRef.current = null;
    onExplorationInvalidated?.();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setResultType('filter');
    setReplaySnapshot(null);
    setActiveQuerySession(null);
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
    onExplorationInvalidated,
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

  const handleNewSearch = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    liveKisSnapshotRef.current = null;
    onExplorationInvalidated?.();
    setIsSearching(false);
    setKisSession(resetKisSession());
    setSelectedImageFile(null);
    if (imageInputRef.current) imageInputRef.current.value = '';
    handleClearFilter();
    setFrames([]);
    setKisEvents([]);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
    setReplaySnapshot(null);
    setActiveQuerySession(null);
    lastReplayTokenRef.current = null;
  }, [handleClearFilter, onExplorationInvalidated]);

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
      <div className="frames-results-shell">
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

  return (
    <div className="adhoc-workspace search-workspace">
      <form className="search-query-form" onSubmit={submit}>
        {selectedImageFile ? (
          <div className="search-query-row">
            <div
              className="query-input-wrapper"
              onDragOver={handleImageDragOver}
              onDragLeave={handleImageDragLeave}
              onDrop={handleImageDrop}
            >
              <div className="query-image-preview-bar">
                {imagePreviewUrl && (
                  <img
                    src={imagePreviewUrl}
                    alt="Upload preview"
                    className="query-image-thumb"
                  />
                )}
                <div className="query-image-info">
                  <span className="query-image-name" title={selectedImageFile.name}>{selectedImageFile.name}</span>
                  <span className="query-image-size">{formatFileSize(selectedImageFile.size)}</span>
                </div>
                <button
                  type="button"
                  className="image-clear-btn"
                  onClick={handleClearImageFile}
                  title="Remove image"
                  aria-label="Remove image"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="search-query-actions">
              <input
                ref={imageInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="image-file-hidden-input"
                data-testid="image-search-file-input"
                onChange={(e) => handleImageFileSelect(e.target.files?.[0])}
              />
              <button
                type="submit"
                className="btn-primary query-submit-btn"
                disabled={isSearching}
              >
                {isSearching ? 'Searching…' : 'Search'}
              </button>
              <button
                type="button"
                className="btn-secondary search-action-btn"
                onClick={() => imageInputRef.current?.click()}
                title="Upload image"
              >
                Upload
              </button>
              <button
                type="button"
                className="btn-secondary search-action-btn"
                onClick={handleNewSearch}
                title="Shortcut: N"
              >
                New Search
              </button>
            </div>
          </div>
        ) : (
          <div className="search-query-row search-image-drop-row">
            <div
              className={`query-input-wrapper search-image-drop-target${isImageDragOver ? ' drag-over' : ''}`}
              onDragOver={handleImageDragOver}
              onDragLeave={handleImageDragLeave}
              onDrop={handleImageDrop}
            >
              <span>Drop an image here to search by image.</span>
            </div>
            <div className="search-query-actions">
              <input
                ref={imageInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="image-file-hidden-input"
                data-testid="image-search-file-input"
                onChange={(e) => handleImageFileSelect(e.target.files?.[0])}
              />
              <button
                type="button"
                className="btn-secondary search-action-btn"
                onClick={() => imageInputRef.current?.click()}
                title="Upload image"
              >
                Upload
              </button>
            </div>
          </div>
        )}
        <div className="search-filter-row">
          <div className="search-filter-inputs-wrapper">
            <input
              type="text"
              className="input-text search-filter-input"
              placeholder="Folder ID"
              aria-label="Filter Folder ID"
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
              type="text"
              className="input-text search-filter-input"
              placeholder="Video ID"
              aria-label="Filter Video ID"
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
              type="text"
              className="input-text search-filter-input"
              placeholder="Title"
              aria-label="Filter Title"
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
              type="text"
              className="input-text search-filter-input"
              placeholder="ASR"
              aria-label="Filter ASR"
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
              type="text"
              className="input-text search-filter-input"
              placeholder="OCR"
              aria-label="Filter OCR"
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
              type="text"
              className="input-text search-filter-input"
              placeholder="Object (name: count)"
              aria-label="Filter Object"
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
              type="button"
              className="btn-primary search-filter-btn"
              disabled={isSearching || !hasAnyFilterValue}
              onClick={() => submitFilter({ pageId: 1 })}
            >
              {isSearching && resultType === 'filter' ? 'Filtering…' : 'Filter'}
            </button>
            <button
              type="button"
              className="btn-secondary search-action-btn search-filter-clear-btn"
              onClick={handleClearFilter}
              disabled={isSearching || !hasAnyFilterValue}
            >
              Clear
            </button>
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
        {!selectedImageFile && (
          <aside className="kis-chat-sidebar" aria-label="KIS search">
            <KisPanel
              sessionState={{
                ...kisSession,
                isSearching: isSearching || kisSession.isSearching,
                error: error || kisSession.error,
              }}
              inputRef={setQueryTextareaRef}
              onDraftChange={(val) => {
                setKisSession((prev) => setDraft(prev, val));
                onQueryChange?.(val);
              }}
              onSubmit={submit}
              onReset={handleNewSearch}
              disabled={isSearching}
            />
          </aside>
        )}
      </div>
    </div>
  );
};

export default SearchWorkspace;
