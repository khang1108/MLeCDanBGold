import React, { useEffect, useRef, useState } from 'react';
import { resolveFrameAtTimestamp } from '../../../api/frames';

/**
 * Form to resolve and open a video moment directly in the viewer popup modal.
 */
export const ManualVideoOpener = ({ onOpenFrame }) => {
  const [videoId, setVideoId] = useState('');
  const [timestampText, setTimestampText] = useState('');
  const [videoError, setVideoError] = useState(null);
  const [isOpeningVideo, setIsOpeningVideo] = useState(false);
  const viewerRequestRef = useRef(null);

  useEffect(() => () => {
    viewerRequestRef.current?.abort();
    viewerRequestRef.current = null;
  }, []);

  const handleManualVideoSubmit = async (event) => {
    event.preventDefault();
    if (isOpeningVideo) return;

    const trimmedVideoId = videoId.trim();
    const timestamp = Number(timestampText.trim());
    if (!trimmedVideoId) {
      setVideoError('Enter a video_id.');
      return;
    }
    if (!/^\d+$/.test(timestampText.trim()) || !Number.isSafeInteger(timestamp) || timestamp < 0) {
      setVideoError('timestamp_ms must be a non-negative base-10 integer.');
      return;
    }
    setVideoError(null);
    viewerRequestRef.current?.abort();
    const controller = new AbortController();
    viewerRequestRef.current = controller;
    setIsOpeningVideo(true);

    try {
      const frame = await resolveFrameAtTimestamp({
        videoId: trimmedVideoId,
        timestampMs: timestamp,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      const requestedTimestampMs = frame.requested_timestamp_ms ?? timestamp;
      onOpenFrame?.({
        frame,
        initialTimestampMs: requestedTimestampMs,
      });
    } catch (error) {
      if (error.name !== 'AbortError') {
        setVideoError(error.message || 'Could not resolve canonical frame metadata.');
      }
    } finally {
      if (viewerRequestRef.current === controller) {
        viewerRequestRef.current = null;
        setIsOpeningVideo(false);
      }
    }
  };

  return (
    <div className="toolbox-section toolbox-video-opener-section">
      <div className="toolbox-label-row">
        <span className="toolbox-label">Open Video Moment</span>
      </div>
      <form className="toolbox-video-opener-form" onSubmit={handleManualVideoSubmit}>
        <div className="toolbox-video-field">
          <label htmlFor="toolbox-video-id" className="toolbox-field-caption">
            video_id
          </label>
          <input
            id="toolbox-video-id"
            className="input-text toolbox-video-input"
            value={videoId}
            onChange={(event) => setVideoId(event.target.value)}
            placeholder="L21_V001"
            autoComplete="off"
            aria-label="video_id"
          />
        </div>
        <div className="toolbox-video-field">
          <label htmlFor="toolbox-timestamp-ms" className="toolbox-field-caption">
            timestamp_ms
          </label>
          <input
            id="toolbox-timestamp-ms"
            className="input-text toolbox-video-input"
            value={timestampText}
            onChange={(event) => setTimestampText(event.target.value)}
            placeholder="12000"
            inputMode="numeric"
            aria-label="timestamp_ms"
          />
        </div>
        {videoError && (
          <p className="toolbox-form-error" role="alert">
            {videoError}
          </p>
        )}
        <button
          type="submit"
          className="btn-primary toolbox-video-submit-btn"
          disabled={isOpeningVideo}
        >
          {isOpeningVideo ? 'Opening…' : 'Open in viewer'}
        </button>
      </form>
    </div>
  );
};

export default ManualVideoOpener;
