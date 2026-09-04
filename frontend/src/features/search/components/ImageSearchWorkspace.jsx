/**
 * Image Search workspace orchestration for visual nearest-neighbour queries.
 *
 * Reuses the layout, result presentation, and canonical frame submission
 * behaviour of SearchWorkspace while replacing the text input with an image upload.
 * It intentionally does not create or update query-history records.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { searchFramesByImage } from '../../../api/search';
import FramesBox from '../../frames/components/FramesBox';
import ToolBox from '../../search-controls/components/ToolBox';
import GifLoaderOverlay from './GifLoaderOverlay';
import { displayVideoId } from '../../frames/videoSource';
import { useSubmissionDialog } from '../../submission/contexts/SubmissionDialogContext';

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
}) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [frames, setFrames] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  const fileInputRef = useRef(null);
  const requestRef = useRef(null);
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

  const openCanonicalFrame = useCallback((frame, submissionMode = 'kis') => {
    onFrameClick?.({
      frame,
      submissionMode,
    });
  }, [onFrameClick]);

  const handleFrameSubmit = useCallback((frame) => {
    const vid = displayVideoId(frame.video_id);
    requestSubmission({
      line: `${vid},${frame.frame_idx}`,
      source: 'Image search frame',
    });
  }, [requestSubmission]);

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

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;

    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setSearchLatencyMs(null);

    try {
      const response = await searchFramesByImage({
        imageFile: selectedFile,
        topK,
        signal: controller.signal,
      });

      if (controller.signal.aborted) return;

      setFrames(response.results || []);
      setSearchLatencyMs(response.latency);
      setWarnings(response.warnings || []);
    } catch (requestError) {
      if (requestError.name === 'AbortError') return;
      setError(requestError.message || 'Failed to contact search API');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [isSearching, selectedFile, topK]);

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
  }, []);

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
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageSearchWorkspace;
