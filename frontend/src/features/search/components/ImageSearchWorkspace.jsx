/**
 * Image Search workspace orchestration for visual nearest-neighbour queries.
 *
 * Reuses the layout, result presentation, and canonical frame submission
 * behaviour of SearchWorkspace while replacing the text input with an image upload.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { searchFramesByImage } from '../../../api/search';
import {
  createQueryHistory,
  markFrameViewed,
} from '../../../api/workspace';
import FramesBox from '../../frames/components/FramesBox';
import ToolBox from '../../search-controls/components/ToolBox';
import GifLoaderOverlay from './GifLoaderOverlay';
import { displayVideoId } from '../../frames/videoSource';
import { useSubmissionDialog } from '../../submission/contexts/SubmissionDialogContext';
import {
  buildKisSnapshot,
  normalizeFrameActivity,
  withViewedFrame,
  withSubmittedFrames,
  activityStateForFrame,
} from '../../workspace/queryHistory';

const createClientQueryId = () => {
  if (typeof window !== 'undefined' && typeof window.crypto?.randomUUID === 'function') {
    return window.crypto.randomUUID();
  }
  return `img-query-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const formatFileSize = (bytes) => {
  if (!bytes || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const ImageSearchWorkspace = ({
  isActive = true,
  topK = 20,
  setTopK,
  onFrameClick,
  userId,
  onFocusUserId,
  onHistoryRefresh,
}) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [frames, setFrames] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [activeQuerySession, setActiveQuerySession] = useState(null);

  const fileInputRef = useRef(null);
  const requestRef = useRef(null);
  const viewedPatchRef = useRef(new Set());
  const { requestSubmission } = useSubmissionDialog();

  // Manage thumbnail object URL
  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return undefined;
    }
    if (typeof URL.createObjectURL === 'function') {
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreviewUrl(objectUrl);
      return () => {
        if (typeof URL.revokeObjectURL === 'function') {
          URL.revokeObjectURL(objectUrl);
        }
      };
    }
    return undefined;
  }, [selectedFile]);

  // Clean up abort controller on unmount
  useEffect(() => () => requestRef.current?.abort(), []);

  // Listen for history submission changes
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
    setActiveQuerySession((current) => (current
      ? { ...current, frameActivity: withViewedFrame(current.frameActivity, frameId) }
      : current));
    markFrameViewed({ queryId: session.queryId, frameId }).catch((patchError) => {
      viewedPatchRef.current.delete(patchKey);
      setWarnings((current) => Array.from(new Set([
        ...current,
        `History view state was not recorded: ${patchError.message || 'request failed'}`,
      ])));
    });
  }, [activeQuerySession]);

  const openCanonicalFrame = useCallback((frame, submissionMode = 'kis') => {
    recordViewed(frame);
    onFrameClick?.({
      frame,
      submissionMode,
      history: historyForSession(activeQuerySession, [frame.frame_id]),
    });
  }, [activeQuerySession, historyForSession, onFrameClick, recordViewed]);

  const handleFrameSubmit = useCallback((frame) => {
    const vid = displayVideoId(frame.video_id);
    requestSubmission({
      line: `${vid},${frame.frame_idx}`,
      source: 'Image search frame',
      history: historyForSession(activeQuerySession, [frame.frame_id]),
    });
  }, [activeQuerySession, historyForSession, requestSubmission]);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    setSelectedFile(file);
    setError(null);
  }, []);

  // Support pasting image from clipboard (Ctrl + V / Cmd + V)
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
            handleFileSelect(namedFile);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [handleFileSelect, isActive]);

  const handleClearFile = (e) => {
    e?.stopPropagation?.();
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const submit = useCallback(async (event) => {
    event?.preventDefault?.();
    if (!selectedFile || isSearching) return;

    if (typeof userId === 'string' && !userId.trim()) {
      setError('Enter a User ID before searching.');
      onFocusUserId?.();
      return;
    }

    const capturedUserId = typeof userId === 'string' ? userId.trim() : null;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const queryId = capturedUserId ? createClientQueryId() : null;

    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setSearchLatencyMs(null);
    setActiveQuerySession(null);

    try {
      const response = await searchFramesByImage({
        imageFile: selectedFile,
        topK,
        signal: controller.signal,
      });

      if (controller.signal.aborted) return;

      const snapshotOptions = {
        events: [],
        latency: response.latency,
        warnings: response.warnings || [],
      };
      const snapshot = buildKisSnapshot(response.results || [], snapshotOptions);

      setFrames(response.results || []);
      setSearchLatencyMs(response.latency);
      setWarnings(response.warnings || []);

      if (queryId) {
        try {
          await createQueryHistory({
            queryId,
            userId: capturedUserId,
            queryText: `[Image] ${selectedFile.name}`,
            resultSnapshot: snapshot,
            signal: controller.signal,
          });
          if (controller.signal.aborted) return;
          viewedPatchRef.current = new Set();
          setActiveQuerySession({
            queryId,
            ownerUserId: capturedUserId,
            queryText: `[Image] ${selectedFile.name}`,
            resultSnapshot: snapshot,
            frameActivity: normalizeFrameActivity(),
            source: 'image-search',
          });
          onHistoryRefresh?.();
        } catch (historyError) {
          if (historyError.name === 'AbortError') return;
          setWarnings((current) => [...current, `History was not saved: ${historyError.message || 'request failed'}`]);
        }
      }
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setError(requestError.message || 'Failed to contact search API');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [isSearching, onFocusUserId, onHistoryRefresh, selectedFile, topK, userId]);

  const handleNewSearch = useCallback(() => {
    requestRef.current?.abort();
    requestRef.current = null;
    setIsSearching(false);
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setFrames([]);
    setWarnings([]);
    setError(null);
    setSearchLatencyMs(null);
    setActiveQuerySession(null);
  }, []);

  const getFrameClassName = useCallback(
    (frameOrId) => {
      const frameId = typeof frameOrId === 'string' ? frameOrId : frameOrId?.frame_id;
      if (!frameId) return '';
      return activityStateForFrame(frameId, activeQuerySession?.frameActivity);
    },
    [activeQuerySession?.frameActivity],
  );

  return (
    <div className="adhoc-workspace search-workspace">
      <form className="search-query-form image-search-form" onSubmit={submit}>
        <div className="search-query-row image-query-row">
          <div
            className="image-dropzone-wrapper"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="image-file-hidden-input"
              onChange={(e) => handleFileSelect(e.target.files?.[0])}
            />

            {!selectedFile ? (
              <button
                type="button"
                className={`image-dropzone-btn ${isDragOver ? 'drag-over' : ''}`}
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="image-dropzone-icon" aria-hidden="true">📁</span>
                <span>Choose, drop, or paste an image (Ctrl + V)</span>
              </button>
            ) : (
              <div className="image-preview-badge">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Upload preview"
                    className="image-preview-thumb"
                  />
                )}
                <div className="image-preview-details">
                  <span className="image-preview-name">{selectedFile.name}</span>
                  <span className="image-preview-size">{formatFileSize(selectedFile.size)}</span>
                </div>
                <button
                  type="button"
                  className="image-clear-btn"
                  onClick={handleClearFile}
                  title="Remove image"
                  aria-label="Remove image"
                >
                  ✕
                </button>
              </div>
            )}
          </div>

          <div className="search-query-actions">
            <button
              type="submit"
              className="btn-primary query-submit-btn"
              disabled={isSearching || !selectedFile}
            >
              {isSearching ? 'Searching…' : 'Search'}
            </button>
            <button
              type="button"
              className="btn-secondary search-action-btn"
              onClick={handleNewSearch}
            >
              New Search
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
            showRetrievalSources={false}
            includeSubmissionWorktree={isActive}
          />
        </aside>

        <div className="adhoc-results">
          <GifLoaderOverlay isVisible={isSearching} />
          {!isSearching && (
            <FramesBox
              results={frames}
              isLoading={false}
              error={error}
              latencyMs={searchLatencyMs}
              warnings={warnings}
              events={[]}
              onFrameClick={openCanonicalFrame}
              onSubmit={handleFrameSubmit}
              getFrameClassName={getFrameClassName}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageSearchWorkspace;
