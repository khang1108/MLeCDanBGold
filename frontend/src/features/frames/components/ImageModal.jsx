import React, { useCallback, useEffect, useMemo, useState } from "react";
import FrameMetadata from "./FrameMetadata";
import VideoTimeline from "./VideoTimeline";
import ExplorationPanel from "../../alignment/components/ExplorationPanel";
import {
  displayVideoId,
  getStreamVideoUrl,
} from "../videoSource";

// The player page endpoint returns HTML, so the inspector uses the raw MP4
// stream and seeks native media time to the selected canonical timestamp.
const ImageModal = ({
  frame = {},
  initialTimestampMs,
  query,
  onClose,
  exploration,
  workspaceAction,
  onAddCandidate,
}) => {
  const modalCardRef = React.useRef(null);
  const videoRef = React.useRef(null);
  const [videoError, setVideoError] = useState(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
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
    setIsVideoReady(false);
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
    setIsVideoReady(true);
    const duration = Number(video.duration);
    setVideoDuration(Number.isFinite(duration) && duration > 0 ? duration : 0);
    if (targetTime === null) return;

    const seekTime = Number.isFinite(duration) && duration >= 0
      ? Math.min(targetTime, duration)
      : targetTime;
    video.currentTime = seekTime;
    updatePlaybackTime(seekTime);
  }, [targetTime, updatePlaybackTime]);

  const addCurrentVideoMoment = useCallback(() => {
    const video = videoRef.current;
    if (!video || typeof onAddCandidate !== 'function') return;
    const currentTime = video.currentTime;
    if (!Number.isFinite(currentTime) || currentTime < 0) return;
    const timestampMs = Math.round(video.currentTime * 1000);
    onAddCandidate({ kind: 'FRAME', videoId: frame.video_id, timestampMs });
  }, [frame.video_id, onAddCandidate]);

  const handleVideoTimeUpdate = useCallback((event) => {
    updatePlaybackTime(event.currentTarget.currentTime);
  }, [updatePlaybackTime]);

  const handleVideoSeek = useCallback((nextTime) => {
    const video = videoRef.current;
    if (!video || !Number.isFinite(nextTime)) return;
    video.currentTime = nextTime;
    updatePlaybackTime(nextTime);
  }, [updatePlaybackTime]);

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
        {query?.trim() && (
          <div className="modal-query-context" role="status" aria-label="Current query">
            <p className="modal-query-text">{query.trim()}</p>
          </div>
        )}
        <div
          ref={modalCardRef}
          className="modal-card split-layout"
          onKeyDown={handleModalKeyDown}
          tabIndex={-1}
        >
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
      {videoLabel} · {Number.isFinite(frame.frame_idx) ? frame.frame_idx : `${frame.timestamp_ms} ms`}
            </span>
            <div className="inspector-header-actions">
              {workspaceAction === 'add-candidate' && (
                <button
                  type="button"
                  className="inspector-add-answer-button"
                  onClick={addCurrentVideoMoment}
                  disabled={!isVideoReady || Boolean(videoError)}
                  aria-label="Add current video moment to answer workspace"
                  title="Add the current video time to the answer workspace"
                >
                  ＋ Answer
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
          <div className="inspector-content">
            <FrameMetadata frame={frame} playbackTime={playbackTime} />
            {exploration && (
              <>
                {!exploration.session && (
                  <div className="exploration-launch">
                    <button type="button" disabled={!videoDuration || exploration.pending} onClick={() => exploration.open(videoDuration)}>
                      {exploration.pending ? "Opening…" : "Explore"}
                    </button>
                    {exploration.error && <p role="alert">{exploration.error}</p>}
                  </div>
                )}
              <ExplorationPanel
                events={exploration.events || exploration.session?.events || exploration.session?.view?.events || []}
                session={exploration.session}
                pending={exploration.pending}
                error={exploration.error}
                unsynced={exploration.unsynced}
                onRefresh={exploration.refresh}
                readCurrentTimeMs={() => {
                  const seconds = Number(videoRef.current?.currentTime);
                  return Number.isFinite(seconds) && seconds >= 0 ? Math.round(seconds * 1000) : null;
                }}
                onApprove={(payload) => exploration.act?.({ action: "confirm", ...payload })}
                onDecline={(payload) => exploration.act?.({ action: "reject", ...payload })}
                onSearchRange={(payload) => exploration.act?.({ action: "window", ...payload })}
                onUndo={exploration.undo}
                onBack={exploration.onBack}
                onSeek={(timestampMs) => {
                  const milliseconds = Number(timestampMs);
                  if (Number.isFinite(milliseconds) && milliseconds >= 0) handleVideoSeek(milliseconds / 1000);
                }}
              />
              </>
            )}
          </div>
        </div>
        </div>
      </div>
    </div>
  );
};

export default ImageModal;
