import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import VideoTimeline from "./VideoTimeline";
import EventTrailPanel from "../../event-trail/components/EventTrailPanel";
import { resolveFrameAtTimestamp } from "../../../api/frames";
import {
  displayVideoId,
  getStreamVideoUrl,
} from "../videoSource";
import { keyframeUrl } from "../../../api/keyframes";

// The player page endpoint returns HTML, so the inspector uses the raw MP4
// stream and seeks native media time to the selected canonical timestamp.
const ImageModal = ({
  frame = {},
  events = [],
  initialTimestampMs,
  query,
  onClose,
  eventTrail,
  onOpenSubmission,
  isSubmissionOpening = false,
  isAvsMode = false,
  isCandidateSelected = false,
  onToggleCandidateSelection,
}) => {
  const modalCardRef = React.useRef(null);
  const videoRef = React.useRef(null);
  const [videoError, setVideoError] = useState(null);
  const [videoDuration, setVideoDuration] = useState(0);
  const targetTime = useMemo(
    () => {
      const requestedTimestamp = Number(initialTimestampMs);
      const timestampMs = Number.isInteger(requestedTimestamp) && requestedTimestamp >= 0
        ? requestedTimestamp
        : Number(frame.timestamp_ms);
      return Number.isInteger(timestampMs) && timestampMs >= 0
        ? timestampMs / 1000
        : null;
    },
    [frame.timestamp_ms, initialTimestampMs],
  );
  const streamUrl = useMemo(
    () => getStreamVideoUrl(frame.video_id, frame.timestamp_ms),
    [frame.timestamp_ms, frame.video_id],
  );
  const [playbackTime, setPlaybackTime] = useState(targetTime);
  const [activeFrameId, setActiveFrameId] = useState(frame.frame_id);
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [fitMode, setFitMode] = useState('contain'); // 'contain' | 'cover'
  const [isFullscreen, setIsFullscreen] = useState(false);
  const viewerColumnRef = useRef(null);
  const videoLabel = displayVideoId(frame.video_id);

  const toggleTheater = useCallback(() => {
    setIsTheaterMode((prev) => !prev);
  }, []);

  const toggleFitMode = useCallback(() => {
    setFitMode((prev) => (prev === 'contain' ? 'cover' : 'contain'));
  }, []);

  const toggleFullscreen = useCallback(() => {
    const el = viewerColumnRef.current;
    if (!el) return;
    if (document.fullscreenElement) {
      document.exitFullscreen?.().catch(() => {});
    } else {
      el.requestFullscreen?.().catch(() => {});
    }
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  useEffect(() => {
    setPlaybackTime(targetTime);
    setActiveFrameId(frame.frame_id);
    setVideoError(null);
    setVideoDuration(0);
  }, [streamUrl, targetTime, frame.frame_id]);

  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === "Escape" && !event.defaultPrevented) {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", closeOnEscape);
    modalCardRef.current?.focus();
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  const updatePlaybackTime = useCallback((sourceTime) => {
    const value = Number(sourceTime);
    if (Number.isFinite(value) && value >= 0) {
      setPlaybackTime(value);
    }
  }, []);

  const handleVideoLoadedMetadata = useCallback((event) => {
    const video = event.currentTarget;
    const duration = Number(video.duration);
    setVideoDuration(Number.isFinite(duration) && duration > 0 ? duration : 0);
    if (targetTime === null) return;

    const seekTime = Number.isFinite(duration) && duration >= 0
      ? Math.min(targetTime, duration)
      : targetTime;
    video.currentTime = seekTime;
    updatePlaybackTime(seekTime);
  }, [targetTime, updatePlaybackTime]);



  const handleVideoTimeUpdate = useCallback((event) => {
    updatePlaybackTime(event.currentTarget.currentTime);
  }, [updatePlaybackTime]);

  const handleVideoSeek = useCallback((nextTime) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(nextTime)) return;
    video.currentTime = nextTime;
    updatePlaybackTime(nextTime);
  }, [updatePlaybackTime]);

  const handleSeekFromTimestamp = useCallback((timestampMs, candidateFrameId) => {
    if (!Number.isFinite(timestampMs) || timestampMs < 0) return;
    const timeSec = timestampMs / 1000;
    handleVideoSeek(timeSec);
    updatePlaybackTime(timeSec);
    if (candidateFrameId) {
      setActiveFrameId(candidateFrameId);
    }
  }, [handleVideoSeek, updatePlaybackTime]);


  const effectiveTrailEvents = useMemo(() => {
    if (Array.isArray(eventTrail?.context?.events) && eventTrail.context.events.length > 0) {
      return eventTrail.context.events;
    }
    if (Array.isArray(events) && events.length > 0) {
      return events.map((e, idx) => (
        typeof e === 'string'
          ? { id: `E${idx + 1}`, text: e }
          : { id: e.id || `E${idx + 1}`, text: e.text || e.canonical_text || '' }
      ));
    }
    return [];
  }, [eventTrail?.context?.events, events]);

  const [selectedEventId, setSelectedEventId] = useState(null);
  const lastActionRef = useRef(null);
  const prevTrailRevRef = useRef(eventTrail?.state?.trail_revision);

  const eventLabel = useMemo(() => {
    if (selectedEventId) return selectedEventId;
    return frame.eventLabel || frame.event_label || null;
  }, [selectedEventId, frame.eventLabel, frame.event_label]);

  const displayQuery = useMemo(() => {
    if (selectedEventId && effectiveTrailEvents?.length) {
      const match = effectiveTrailEvents.find(
        (e) => (typeof e === 'object' && e?.id === selectedEventId)
      );
      if (match?.text) return match.text;
    }
    if (frame?.eventText) return frame.eventText;
    if (frame?.event_text) return frame.event_text;
    if (Number.isInteger(frame?.eventIndex) && events?.[frame.eventIndex]) {
      const ev = events[frame.eventIndex];
      return typeof ev === 'string' ? ev : ev?.text || ev?.canonical_text || '';
    }
    if (Number.isInteger(frame?.event_index) && events?.[frame.event_index]) {
      const ev = events[frame.event_index];
      return typeof ev === 'string' ? ev : ev?.text || ev?.canonical_text || '';
    }
    return query || '';
  }, [
    selectedEventId,
    effectiveTrailEvents,
    frame?.eventText,
    frame?.event_text,
    frame?.eventIndex,
    frame?.event_index,
    events,
    query,
  ]);

  useEffect(() => {
    const currentState = eventTrail?.state;
    if (!currentState) {
      prevTrailRevRef.current = undefined;
      return;
    }
    const prevRev = prevTrailRevRef.current;
    prevTrailRevRef.current = currentState.trail_revision;

    if (prevRev !== undefined && currentState.trail_revision !== prevRev) {
      const lastAction = lastActionRef.current;
      if (lastAction?.type === 'decline' && currentState.status === 'active') {
        const actedEventId = currentState.transition?.action_event_id || lastAction.eventId;
        const candidate = currentState.path?.find((c) => c.event_id === actedEventId);
        if (candidate && Number.isFinite(candidate.timestamp_ms)) {
          handleSeekFromTimestamp(candidate.timestamp_ms, candidate.frame_id);
        }
      }
      lastActionRef.current = null;
    }
  }, [eventTrail?.state, handleSeekFromTimestamp]);

  const handleUse = useCallback(async (eventId) => {
    if (!eventId || !eventTrail?.act) return;
    const video = videoRef.current;
    const currentTime = video?.currentTime ?? playbackTime ?? 0;
    const timestampMs = Math.max(0, Math.round(currentTime * 1000));
    const videoId = frame.video_id;
    try {
      const resolved = await resolveFrameAtTimestamp({ videoId, timestampMs });
      if (resolved?.frame_id && resolved?.video_id === videoId) {
        lastActionRef.current = { type: 'use_frame', eventId };
        await eventTrail.act({
          type: 'use_frame',
          event_id: eventId,
          frame_id: resolved.frame_id,
        });
      }
    } catch (err) {
      console.error('Failed to resolve canonical frame for Use:', err);
    }
  }, [eventTrail, frame.video_id, playbackTime]);

  const handleApprove = useCallback(async (eventId) => {
    if (!eventId || !eventTrail?.act) return;
    lastActionRef.current = { type: 'approve', eventId };
    await eventTrail.act({ type: 'approve', event_id: eventId });
  }, [eventTrail]);

  const handleDecline = useCallback(async (eventId) => {
    if (!eventId || !eventTrail?.act) return;
    lastActionRef.current = { type: 'decline', eventId };
    await eventTrail.act({ type: 'decline', event_id: eventId });
  }, [eventTrail]);

  const handleClearAnchor = useCallback(async (eventId) => {
    if (!eventId || !eventTrail?.act) return;
    lastActionRef.current = { type: 'clear_anchor', eventId };
    await eventTrail.act({ type: 'clear_anchor', event_id: eventId });
  }, [eventTrail]);

  const handleUndo = useCallback(async () => {
    if (!eventTrail?.undo) return;
    lastActionRef.current = { type: 'undo' };
    await eventTrail.undo();
  }, [eventTrail]);

  const handleSetWindow = useCallback(async (action) => {
    if (!eventTrail?.act) return;
    lastActionRef.current = { type: 'set_window' };
    await eventTrail.act(action);
  }, [eventTrail]);

  const handleClearWindow = useCallback(async () => {
    if (!eventTrail?.act) return;
    lastActionRef.current = { type: 'clear_window' };
    await eventTrail.act({ type: 'clear_window' });
  }, [eventTrail]);

  const handleExitTrail = useCallback(async () => {
    if (eventTrail?.close) {
      await eventTrail.close({ suppressError: true });
    }
  }, [eventTrail]);

  const handleBack = useCallback(async () => {
    if (eventTrail?.back) {
      await eventTrail.back();
    }
    onClose();
  }, [eventTrail, onClose]);

  const handleSubmitFromTrail = useCallback((selection) => {
    if (!selection || !Number.isFinite(selection.timestamp_ms)) return;
    if (typeof onOpenSubmission === 'function') {
      const videoId = eventTrail?.state?.video_id || frame.video_id;
      onOpenSubmission({
        videoId,
        startMs: selection.timestamp_ms,
        endMs: selection.timestamp_ms,
      });
    }
  }, [eventTrail?.state?.video_id, frame.video_id, onOpenSubmission]);

  const currentTimestampMs = useMemo(() => {
    const video = videoRef.current;
    const currentTime = Number.isFinite(video?.currentTime)
      ? video.currentTime
      : playbackTime;
    return Number.isFinite(currentTime) && currentTime >= 0
      ? Math.round(currentTime * 1000)
      : (Number.isFinite(frame.timestamp_ms) ? frame.timestamp_ms : 0);
  }, [playbackTime, frame.timestamp_ms]);

  const handleDirectSubmit = useCallback(() => {
    if (typeof onOpenSubmission !== 'function') return;
    const video = videoRef.current;
    const currentTime = Number.isFinite(video?.currentTime)
      ? video.currentTime
      : playbackTime;
    const timestampMs = Number.isFinite(currentTime) && currentTime >= 0
      ? Math.round(currentTime * 1000)
      : (Number.isFinite(frame.timestamp_ms) ? frame.timestamp_ms : 0);
    const videoId = typeof frame.video_id === 'string' ? frame.video_id.trim() : '';

    if (!videoId) return;

    onOpenSubmission({
      videoId,
      startMs: timestampMs,
      endMs: timestampMs,
    });
  }, [frame.video_id, frame.timestamp_ms, onOpenSubmission, playbackTime]);

  const canSubmitFrame = typeof onOpenSubmission === 'function'
    && typeof frame.video_id === 'string'
    && frame.video_id.trim().length > 0;

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused || video.ended) {
      video.play?.().catch?.(() => undefined);
    } else {
      video.pause?.();
    }
  }, []);

  const seekBy = useCallback((offsetSeconds) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(video.currentTime)) return;
    const duration = Number(video.duration);
    const maximum = Number.isFinite(duration) && duration >= 0 ? duration : Infinity;
    const nextTime = Math.min(Math.max(video.currentTime + offsetSeconds, 0), maximum);
    video.currentTime = nextTime;
    updatePlaybackTime(nextTime);
  }, [updatePlaybackTime]);

  const handleModalKeyDown = useCallback((event) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }

    const targetTag = event.target?.tagName;
    if (['BUTTON', 'INPUT', 'TEXTAREA', 'SELECT'].includes(targetTag)) return;

    if (event.key === ' ' || event.key === 'Spacebar' || event.key.toLowerCase() === 'k') {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      seekBy(-5);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      seekBy(5);
    } else if (event.key.toLowerCase() === 't' || event.key.toLowerCase() === 'e') {
      event.preventDefault();
      toggleTheater();
    } else if (event.key.toLowerCase() === 'f') {
      event.preventDefault();
      toggleFullscreen();
    } else if (event.key.toLowerCase() === 'c') {
      event.preventDefault();
      toggleFitMode();
    }

  }, [onClose, seekBy, togglePlayback, toggleTheater, toggleFullscreen, toggleFitMode]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-frame-stack" onClick={(event) => event.stopPropagation()}>
        <div
          ref={modalCardRef}
          className={`modal-card split-layout ${isTheaterMode ? 'is-theater-mode' : ''}`}
          onKeyDown={handleModalKeyDown}
          tabIndex={-1}
        >
          <div className="modal-main-stage">
            <div
              ref={viewerColumnRef}
              className={`modal-viewer-column ${isTheaterMode ? 'is-theater' : ''} fit-${fitMode}`}
            >
              {/* Floating Top Control Bar */}
              <div className="modal-viewer-top-bar">
                <div className="modal-viewer-top-left">
                  <span className="modal-viewer-badge">
                    #{activeFrameId || frame.frame_id}
                  </span>
                </div>
                <div className="modal-viewer-top-right">
                  {videoError && (
                    <>
                      {canSubmitFrame && !eventTrail?.state && (
                        <button
                          type="button"
                          className="modal-viewer-action-btn submit-btn"
                          onClick={handleDirectSubmit}
                          disabled={isSubmissionOpening}
                          aria-label="Submit this keyframe"
                        >
                          ↗ Submit
                        </button>
                      )}
                      {isAvsMode && onToggleCandidateSelection && (
                        <button
                          type="button"
                          className={`modal-viewer-action-btn select-btn ${isCandidateSelected ? 'is-selected' : ''}`}
                          onClick={onToggleCandidateSelection}
                          aria-label={isCandidateSelected ? "Deselect candidate from AVS" : "Select candidate for AVS"}
                        >
                          {isCandidateSelected ? "✓ Selected" : "+ Select"}
                        </button>
                      )}
                    </>
                  )}
                  <button
                    type="button"
                    className={`modal-viewer-icon-btn ${fitMode === 'cover' ? 'active' : ''}`}
                    onClick={toggleFitMode}
                    title={fitMode === 'contain' ? "Fill frame (eliminate black borders) [C]" : "Fit entire video/frame [C]"}
                    aria-label={fitMode === 'contain' ? "Fill frame" : "Fit frame"}
                  >
                    {fitMode === 'cover' ? '⤢ Fit' : '⇲ Fill'}
                  </button>
                  <button
                    type="button"
                    className={`modal-viewer-icon-btn ${isTheaterMode ? 'active' : ''}`}
                    onClick={toggleTheater}
                    title={isTheaterMode ? "Standard view [T]" : "Expand player / Theater mode [T]"}
                    aria-label={isTheaterMode ? "Exit theater mode" : "Expand player"}
                  >
                    {isTheaterMode ? '⤡ Standard' : '⤢ Expand'}
                  </button>
                  <button
                    type="button"
                    className="modal-viewer-icon-btn"
                    onClick={toggleFullscreen}
                    title={isFullscreen ? "Exit fullscreen [F]" : "Fullscreen [F]"}
                    aria-label={isFullscreen ? "Exit viewer fullscreen" : "Viewer fullscreen"}
                  >
                    ⛶
                  </button>
                </div>
              </div>

              {isTheaterMode && (
                <button
                  type="button"
                  className="modal-restore-inspector-btn"
                  onClick={toggleTheater}
                  title="Restore Inspector [T]"
                  aria-label="Restore inspector"
                >
                  ◂ Inspector
                </button>
              )}

              {streamUrl && targetTime !== null && !videoError ? (
                <div className="modal-video-shell">
                  <video
                    ref={videoRef}
                    className={`modal-viewer-video fit-${fitMode}`}
                    src={streamUrl}
                    autoPlay
                    muted
                    preload="metadata"
                    playsInline
                    aria-label={`Video for ${videoLabel}`}
                    onLoadedMetadata={handleVideoLoadedMetadata}
                    onTimeUpdate={handleVideoTimeUpdate}
                    onError={() => setVideoError('The MP4 stream could not be loaded or decoded.')}
                    onDoubleClick={toggleTheater}
                  />
                  <VideoTimeline
                    videoId={frame.video_id}
                    videoRef={videoRef}
                    currentTime={playbackTime}
                    duration={videoDuration}
                    onSeek={handleVideoSeek}
                    onTogglePlayback={togglePlayback}
                    onSubmit={canSubmitFrame && !eventTrail?.state ? handleDirectSubmit : undefined}
                    isSubmitting={isSubmissionOpening}
                    onSelectCandidate={isAvsMode && onToggleCandidateSelection ? onToggleCandidateSelection : undefined}
                    isCandidateSelected={isCandidateSelected}
                    onToggleTheater={toggleTheater}
                    isTheaterMode={isTheaterMode}
                    onToggleFitMode={toggleFitMode}
                    fitMode={fitMode}
                  />
                </div>
              ) : videoError && (activeFrameId || frame.frame_id) ? (
                <div className="modal-fallback-viewer" onDoubleClick={toggleTheater}>
                  <img
                    src={keyframeUrl(activeFrameId || frame.frame_id)}
                    alt={`Frame ${activeFrameId || frame.frame_id}`}
                    className={`modal-viewer-fallback-image fit-${fitMode}`}
                  />
                  <div className="modal-video-fallback-notice">
                    <span>
                      Video stream unavailable &bull; Showing keyframe preview #{activeFrameId || frame.frame_id}
                      {Number.isFinite(playbackTime) && ` (${playbackTime.toFixed(1)}s)`}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="frame-image-placeholder">
                  <p>
                    Video playback is unavailable. {videoError || (
                      targetTime === null
                        ? 'The backend response is missing timestamp_ms.'
                        : 'The backend response is missing a canonical video_id.'
                    )}
                  </p>
                </div>
              )}
            </div>
            <div className={`modal-inspector-column ${eventTrail?.state ? 'trail-active' : 'kis-mode'}`}>
              <div className="inspector-header">
                <span className="inspector-title">
                  {eventTrail?.state ? 'EventTrail Exploration' : 'Frame Inspector'}
                </span>
                <div className="inspector-header-actions">
                  <button
                    type="button"
                    className={`inspector-header-theater-btn ${isTheaterMode ? 'is-expanded' : ''}`}
                    onClick={toggleTheater}
                    title={isTheaterMode ? "Restore inspector (T)" : "Expand video player (T)"}
                    aria-label={isTheaterMode ? "Restore inspector" : "Expand video player"}
                  >
                    {isTheaterMode ? '◂ Show' : '⤢ Expand'}
                  </button>
                  {isAvsMode && onToggleCandidateSelection && (
                    <button
                      type="button"
                      className={`inspector-header-select-btn ${isCandidateSelected ? 'is-selected' : ''}`}
                      onClick={onToggleCandidateSelection}
                      aria-label={isCandidateSelected ? "Deselect candidate from AVS" : "Select candidate for AVS"}
                      title={isCandidateSelected ? "Deselect candidate from AVS basket" : "Select candidate for AVS batch"}
                    >
                      <span className="select-btn-check" aria-hidden="true">{isCandidateSelected ? '✓' : '+'}</span>
                      <span>{isCandidateSelected ? 'Selected' : 'Select'}</span>
                    </button>
                  )}
                  {canSubmitFrame && !eventTrail?.state && (
                    <button
                      type="button"
                      className="frame-submit-button inspector-header-submit-btn"
                      disabled={isSubmissionOpening}
                      onClick={handleDirectSubmit}
                      aria-label="Submit this frame to DRES"
                      title={`Submit ${frame.video_id} at ${currentTimestampMs} ms to DRES`}
                    >
                      <span className="submit-arrow-icon" aria-hidden="true">↗</span>
                      <span>Submit</span>
                    </button>
                  )}
                  <button
                    type="button"
                    className="inspector-close-btn"
                    onClick={onClose}
                    aria-label="Close popup"
                  >
                    ×
                  </button>
                </div>
              </div>
              {displayQuery?.trim() && (
                <div className="modal-query-context" role="status" aria-label="Current query">
                  {eventLabel && (
                    <span className="frame-event-badge">{eventLabel}</span>
                  )}
                  <span className="query-context-label">Query:</span>
                  <p className="modal-query-text">{displayQuery.trim()}</p>
                </div>
              )}
              {eventTrail?.state ? (
                <div className="inspector-content">
                  <EventTrailPanel
                    events={effectiveTrailEvents}
                    state={eventTrail.state}
                    pending={eventTrail.pending}
                    error={eventTrail.error}
                    selectedEventId={selectedEventId}
                    onSelectEvent={setSelectedEventId}
                    onUse={handleUse}
                    onKeep={handleApprove}
                    onApprove={handleApprove}
                    onRejectMode={handleDecline}
                    onDecline={handleDecline}
                    onClearAnchor={handleClearAnchor}
                    onUndo={handleUndo}
                    onSetWindow={handleSetWindow}
                    onClearWindow={handleClearWindow}
                    onBack={handleBack}
                    onExitTrail={handleExitTrail}
                    onSubmit={handleSubmitFromTrail}
                  />
                </div>
              ) : (
                <div className="inspector-content">
                  <FrameMetadata frame={frame} playbackTime={playbackTime} />
                </div>
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default ImageModal;
