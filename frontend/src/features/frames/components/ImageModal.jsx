import React, { useCallback, useEffect, useMemo, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import VideoTimeline from "./VideoTimeline";
import AlignmentAccordion from "../../alignment/components/AlignmentAccordion";
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
  exploration,
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
    if (Array.isArray(exploration?.events) && exploration.events.length > 0) return exploration.events;
    return [];
  }, [events, frame.events, frame.aligned_events, exploration?.events]);

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
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
