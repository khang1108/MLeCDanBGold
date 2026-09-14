/**
 * Inline video player for the Workspace page.
 * Displays video playback with timeline controls without opening a modal popup.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import VideoTimeline from '../../frames/components/VideoTimeline';
import { displayVideoId, getStreamVideoUrl } from '../../frames/videoSource';

const WorkspaceInlinePlayer = ({
  frame = {},
  initialTimestampMs,
  onClose,
}) => {
  const videoRef = useRef(null);
  const [videoError, setVideoError] = useState(null);
  const [videoDuration, setVideoDuration] = useState(0);

  const targetTime = useMemo(() => {
    const requestedTimestamp = Number(initialTimestampMs);
    const timestampMs = Number.isInteger(requestedTimestamp) && requestedTimestamp >= 0
      ? requestedTimestamp
      : Number(frame.timestamp_ms);
    return Number.isInteger(timestampMs) && timestampMs >= 0
      ? timestampMs / 1000
      : null;
  }, [frame.timestamp_ms, initialTimestampMs]);

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

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused || video.ended) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  }, []);

  return (
    <section className="workspace-inline-player" aria-label={`Inline player for ${videoLabel}`}>
      <div className="workspace-inline-player-header">
        <div className="workspace-inline-player-title">
          <span className="workspace-inline-player-name">{videoLabel}</span>
          {Number.isFinite(frame.timestamp_ms) && (
            <span className="workspace-inline-player-time">{frame.timestamp_ms} ms</span>
          )}
        </div>
        {typeof onClose === 'function' && (
          <button
            type="button"
            className="workspace-inline-player-close"
            onClick={onClose}
            aria-label="Close video player"
            title="Close video"
          >
            ×
          </button>
        )}
      </div>

      <div className="workspace-inline-player-body">
        {streamUrl && targetTime !== null && !videoError ? (
          <div className="modal-video-shell workspace-inline-video-shell">
            <video
              ref={videoRef}
              className="modal-viewer-video workspace-inline-video"
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
          <div className="workspace-inline-player-error">
            <p>
              Video playback is unavailable. {videoError || (
                targetTime === null
                  ? 'The frame is missing timestamp_ms.'
                  : 'The frame is missing video_id.'
              )}
            </p>
          </div>
        )}
      </div>
    </section>
  );
};

export default WorkspaceInlinePlayer;
