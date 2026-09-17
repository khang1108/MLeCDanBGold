import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import VideoTimeline from "./VideoTimeline";
import AlignmentAccordion from "../../alignment/components/AlignmentAccordion";
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
  const videoLabel = displayVideoId(frame.video_id);

  useEffect(() => {
    setPlaybackTime(targetTime);
    setVideoError(null);
    setVideoDuration(0);
  }, [streamUrl, targetTime]);

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

  const handleSeekFromTimestamp = useCallback((timestampMs) => {
    if (!Number.isFinite(timestampMs) || timestampMs < 0) return;
    handleVideoSeek(timestampMs / 1000);
  }, [handleVideoSeek]);

  const resolvedEvents = useMemo(() => {
    if (Array.isArray(events) && events.length > 0) return events;
    if (Array.isArray(frame.events) && frame.events.length > 0) return frame.events;
    if (Array.isArray(frame.aligned_events) && frame.aligned_events.length > 0) return frame.aligned_events;
    return [];
  }, [events, frame.events, frame.aligned_events]);

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

  const frameIds = useMemo(
    () => frame.frame_ids || frame.aligned_frame_ids || [],
    [frame.frame_ids, frame.aligned_frame_ids],
  );

  const timestampsMs = useMemo(
    () => frame.timestamps_ms || frame.aligned_timestamps_ms || [],
    [frame.timestamps_ms, frame.aligned_timestamps_ms],
  );

  const effectiveEvents = useMemo(() => {
    if (resolvedEvents.length > 0) return resolvedEvents;
    if (frameIds.length > 1 && timestampsMs.length === frameIds.length) {
      return frameIds.map((_, idx) => `Event ${idx + 1}`);
    }
    return [];
  }, [resolvedEvents, frameIds, timestampsMs]);

  const [selectedEventId, setSelectedEventId] = useState(null);
  const lastActionRef = useRef(null);
  const prevTrailRevRef = useRef(eventTrail?.state?.trail_revision);

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
          handleSeekFromTimestamp(candidate.timestamp_ms);
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
  const hasAlignment = useMemo(() => (
    Array.isArray(effectiveEvents)
    && effectiveEvents.length > 0
    && effectiveEvents.length === frameIds?.length
    && effectiveEvents.length === timestampsMs?.length
  ), [effectiveEvents, frameIds, timestampsMs]);

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
    }

  }, [onClose, seekBy, togglePlayback]);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-frame-stack" onClick={(event) => event.stopPropagation()}>
        <div
          ref={modalCardRef}
          className="modal-card split-layout"
          onKeyDown={handleModalKeyDown}
          tabIndex={-1}
        >
<<<<<<< HEAD
        <div className="modal-viewer-column">
          {streamUrl && targetTime !== null && !videoError ? (
            <div className="modal-video-shell">
              <video
                ref={videoRef}
                className="modal-viewer-video"
                src={streamUrl}
                autoPlay
                muted
                preload="metadata"
                playsInline
                aria-label={`Video for ${videoLabel}`}
                onLoadedMetadata={handleVideoLoadedMetadata}
                onTimeUpdate={handleVideoTimeUpdate}
                onError={() => setVideoError('The MP4 stream could not be loaded or decoded.')}
              />
              <VideoTimeline
                videoId={frame.video_id}
                videoRef={videoRef}
                currentTime={playbackTime}
                duration={videoDuration}
                onSeek={handleVideoSeek}
                onTogglePlayback={togglePlayback}
              />
            </div>
          ) : videoError && frame.frame_id ? (
            <div className="modal-fallback-viewer">
              <img
                src={keyframeUrl(frame.frame_id)}
                alt={`Frame ${frame.frame_id}`}
                className="modal-viewer-fallback-image"
              />
              <div className="modal-video-fallback-notice">
                <span>Video stream unavailable &bull; Showing keyframe preview</span>
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
        <div className="modal-inspector-column">
          <div className="inspector-header">
            <span className="inspector-title">
              {Number.isFinite(frame.timestamp_ms) ? `${videoLabel} · ${frame.timestamp_ms} ms` : videoLabel}
            </span>
            <div className="inspector-header-actions">
              {typeof onOpenSubmission === 'function' && !eventTrail?.state && (
                <button
                  type="button"
                  className="inspector-submit-answer-button"
                  onClick={openCurrentVideoMoment}
                  disabled={!isVideoReady || Boolean(videoError) || isSubmissionOpening}
                  aria-label="Submit current video moment to DRES"
                  title={isSubmissionOpening ? 'Loading the current DRES task' : 'Prepare this exact player time for DRES'}
                >
                  ↗ Submit
                </button>
=======
          <div className="modal-main-stage">
            <div className="modal-viewer-column">
              {streamUrl && targetTime !== null && !videoError ? (
                <div className="modal-video-shell">
                  <video
                    ref={videoRef}
                    className="modal-viewer-video"
                    src={streamUrl}
                    autoPlay
                    muted
                    preload="metadata"
                    playsInline
                    aria-label={`Video for ${videoLabel}`}
                    onLoadedMetadata={handleVideoLoadedMetadata}
                    onTimeUpdate={handleVideoTimeUpdate}
                    onError={() => setVideoError('The MP4 stream could not be loaded or decoded.')}
                  />
                  <VideoTimeline
                    videoId={frame.video_id}
                    videoRef={videoRef}
                    currentTime={playbackTime}
                    duration={videoDuration}
                    onSeek={handleVideoSeek}
                    onTogglePlayback={togglePlayback}
                  />
                </div>
              ) : videoError && frame.frame_id ? (
                <div className="modal-fallback-viewer">
                  <img
                    src={keyframeUrl(frame.frame_id)}
                    alt={`Frame ${frame.frame_id}`}
                    className="modal-viewer-fallback-image"
                  />
                  <div className="modal-video-fallback-notice">
                    <span>Video stream unavailable &bull; Showing keyframe preview</span>
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
>>>>>>> origin/feat/ui-vbs
              )}
            </div>
            <div className="modal-inspector-column">
              <div className="inspector-header">
                <span className="inspector-title">Frame Inspector</span>
                <div className="inspector-header-actions">
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
              {query?.trim() && (
                <div className="modal-query-context" role="status" aria-label="Current query">
                  <span className="query-context-label">Query:</span>
                  <p className="modal-query-text">{query.trim()}</p>
                </div>
              )}
              <div className="inspector-content">
                <FrameMetadata frame={frame} playbackTime={playbackTime} />
              </div>
            </div>
          </div>
          {hasAlignment && (
            <div className="modal-bottom-alignment-section">
              <AlignmentAccordion
                events={effectiveEvents}
                frameIds={frameIds}
                timestampsMs={timestampsMs}
                onSeek={handleSeekFromTimestamp}
                collapsible={false}
              />
            </div>
          )}
<<<<<<< HEAD
          {eventTrail?.state ? (
            <div className="inspector-content">
              <EventTrailPanel
                events={effectiveTrailEvents}
                state={eventTrail.state}
                pending={eventTrail.pending}
                error={eventTrail.error}
                selectedEventId={selectedEventId}
                onSelectEvent={setSelectedEventId}
                onExplore={(candidate) => handleSeekFromTimestamp(candidate.timestamp_ms)}
                onUse={handleUse}
                onApprove={handleApprove}
                onDecline={handleDecline}
                onClearAnchor={handleClearAnchor}
                onUndo={handleUndo}
                onSetWindow={handleSetWindow}
                onClearWindow={handleClearWindow}
                onBack={handleBack}
                onSubmit={handleSubmitFromTrail}
              />
            </div>
          ) : (
            <div className="inspector-content">
              <FrameMetadata frame={frame} playbackTime={playbackTime} />
              <AlignmentAccordion
                events={effectiveEvents}
                frameIds={frameIds}
                timestampsMs={timestampsMs}
                onSeek={handleSeekFromTimestamp}
                collapsible={false}
              />
              {eventTrail?.context && (
                <div className="event-trail-launch-card">
                  <button
                    type="button"
                    className="btn-primary event-trail-open-btn"
                    onClick={() => eventTrail.open(eventTrail.context)}
                    disabled={eventTrail.pending}
                  >
                    {eventTrail.pending ? 'Opening EventTrail…' : 'Open EventTrail'}
                  </button>
                </div>
              )}
              <div className="inspector-shortcuts-card">
                <span className="shortcuts-card-title">Video Controls</span>
                <div className="shortcuts-row">
                  <kbd>Space</kbd> / <kbd>K</kbd> <span>Play / Pause</span>
                </div>
                <div className="shortcuts-row">
                  <kbd>←</kbd> <kbd>→</kbd> <span>Seek ±5s</span>
                </div>
                <div className="shortcuts-row">
                  <kbd>Esc</kbd> <span>Close</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  </div>
=======
        </div>
      </div>
    </div>
>>>>>>> origin/feat/ui-vbs
  );
};

export default ImageModal;
