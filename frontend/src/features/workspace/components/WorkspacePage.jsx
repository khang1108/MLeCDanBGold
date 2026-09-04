/**
 * Workspace page for persisted Query history, manual video inspection, and
 * the shared submission filename worktree.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getQueryHistory } from '../../../api/workspace';
import { resolveFrameAtTimestamp } from '../../../api/frames';
import SubmissionWorktree from '../../submission/components/SubmissionWorktree';

const WorkspacePage = ({
  isActive = false,
  userId = '',
  historyRefreshToken = 0,
  onReplay,
  onOpenManualVideo,
}) => {
  const [historyItems, setHistoryItems] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [videoId, setVideoId] = useState('');
  const [timestampText, setTimestampText] = useState('');
  const [videoError, setVideoError] = useState(null);
  const [isOpeningVideo, setIsOpeningVideo] = useState(false);
  const [eventRefreshToken, setEventRefreshToken] = useState(0);
  const viewerRequestRef = useRef(null);

  const loadHistory = useCallback((signal) => {
    const trimmedUserId = userId.trim();
    if (!isActive || !trimmedUserId) return Promise.resolve();
    setIsLoadingHistory(true);
    setHistoryError(null);
    return getQueryHistory({ userId: trimmedUserId, signal })
      .then((response) => setHistoryItems(response.items))
      .catch((error) => {
        if (error.name !== 'AbortError') setHistoryError(error.message || 'Could not load Query history');
      })
      .finally(() => {
        if (!signal?.aborted) setIsLoadingHistory(false);
      });
  }, [isActive, userId]);

  useEffect(() => {
    if (!isActive || !userId.trim()) return undefined;
    const controller = new AbortController();
    loadHistory(controller.signal);
    return () => controller.abort();
  }, [eventRefreshToken, historyRefreshToken, isActive, loadHistory, userId]);

  useEffect(() => {
    const refresh = () => setEventRefreshToken((token) => token + 1);
    window.addEventListener('hcmai:history-changed', refresh);
    return () => window.removeEventListener('hcmai:history-changed', refresh);
  }, []);

  useEffect(() => {
    if (!userId.trim()) {
      setHistoryItems([]);
      setHistoryError(null);
    }
  }, [userId]);

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
      onOpenManualVideo?.({
        frame,
        requestedTimestampMs: frame.requested_timestamp_ms,
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
    <div className="workspace-page" aria-label="Workspace">
      <div className="workspace-columns">
        <section className="workspace-history-panel" aria-labelledby="workspace-history-title">
          <div className="workspace-panel-heading">
            <div>
              <h2 id="workspace-history-title">History</h2>
            </div>
            {isLoadingHistory && <span className="workspace-loading">Loading…</span>}
          </div>
          {!userId.trim() ? (
            <p className="workspace-empty-copy">Enter a User ID in the header to load Query history.</p>
          ) : historyError ? (
            <div className="workspace-error" role="alert">{historyError}</div>
          ) : historyItems.length === 0 && !isLoadingHistory ? (
            <p className="workspace-empty-copy">No saved queries for this User ID yet.</p>
          ) : (
            <div className="workspace-history-list">
              {historyItems.map((item) => (
                <article className="workspace-history-row" key={item.query_id}>
                  <div className="workspace-history-copy">
                    <p className="workspace-history-query">{item.query_text}</p>
                    {item.submission_files?.length > 0 && (
                      <div className="workspace-history-files" aria-label="Submitted files">
                        {item.submission_files.map((fileName) => (
                          <span className="workspace-history-file" key={fileName}>{fileName}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    className="btn-secondary workspace-replay-button"
                    onClick={() => onReplay?.(item)}
                  >
                    Replay in Query
                  </button>
                </article>
              ))}
            </div>
          )}
        </section>

        <aside className="workspace-tools-column">
          <section className="workspace-video-panel" aria-labelledby="workspace-video-title">
            <div className="workspace-panel-heading">
              <div>
                <h2 id="workspace-video-title">Open video moment</h2>
              </div>
            </div>
            <form className="workspace-video-form" onSubmit={handleManualVideoSubmit}>
              <label className="workspace-form-field" htmlFor="workspace-video-id">
                <span>video_id</span>
                <input
                  id="workspace-video-id"
                  className="input-text"
                  value={videoId}
                  onChange={(event) => setVideoId(event.target.value)}
                  placeholder="L21_V001"
                  autoComplete="off"
                />
              </label>
              <label className="workspace-form-field" htmlFor="workspace-timestamp-ms">
                <span>timestamp_ms</span>
                <input
                  id="workspace-timestamp-ms"
                  className="input-text"
                  value={timestampText}
                  onChange={(event) => setTimestampText(event.target.value)}
                  placeholder="12000"
                  inputMode="numeric"
                />
              </label>
              {videoError && <p className="workspace-form-error" role="alert">{videoError}</p>}
              <button type="submit" className="btn-primary" disabled={isOpeningVideo}>
                {isOpeningVideo ? 'Opening…' : 'Open in viewer'}
              </button>
            </form>
          </section>
          {isActive && <SubmissionWorktree />}
        </aside>
      </div>
    </div>
  );
};

export default WorkspacePage;
