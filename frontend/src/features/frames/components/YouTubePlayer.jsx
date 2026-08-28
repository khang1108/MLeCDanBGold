import React, { useEffect, useRef, useState } from 'react';
import { getYouTubeVideoId } from '../videoSource';

const YOUTUBE_IFRAME_API_SRC = 'https://www.youtube.com/iframe_api';
const CURRENT_TIME_POLL_MS = 100;
let playerIdSequence = 0;
let youtubeApiPromise;

const loadYouTubeIframeApi = () => {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (youtubeApiPromise) return youtubeApiPromise;

  youtubeApiPromise = new Promise((resolve, reject) => {
    let settled = false;
    let timeoutId;
    const previousReadyHandler = window.onYouTubeIframeAPIReady;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timeoutId);
      if (error) {
        reject(error);
      } else if (window.YT?.Player) {
        resolve(window.YT);
      } else {
        reject(new Error('YouTube IFrame API loaded without a player.'));
      }
    };

    window.onYouTubeIframeAPIReady = () => {
      try {
        previousReadyHandler?.();
      } catch (error) {
        console.error('[HCMAI YouTube] previous API callback failed:', error);
      }
      finish();
    };

    let script = document.querySelector(`script[src="${YOUTUBE_IFRAME_API_SRC}"]`);
    if (!script) {
      script = document.createElement('script');
      script.src = YOUTUBE_IFRAME_API_SRC;
      script.async = true;
      script.addEventListener(
        'error',
        () => finish(new Error('Could not load YouTube IFrame API.')),
        { once: true },
      );
      document.head.appendChild(script);
    }

    timeoutId = window.setTimeout(
      () => finish(new Error('Timed out loading YouTube IFrame API.')),
      15_000,
    );
  }).catch((error) => {
    youtubeApiPromise = undefined;
    throw error;
  });

  return youtubeApiPromise;
};

const directEmbedUrl = (embedUrl, targetTime) => {
  try {
    const url = new URL(embedUrl);
    if (Number.isFinite(targetTime) && targetTime > 0) {
      url.searchParams.set('start', String(Math.floor(targetTime)));
    }
    return url.toString();
  } catch {
    return embedUrl;
  }
};

const DirectFallback = ({ embedUrl, targetTime, title, onTimeUpdate }) => (
  <iframe
    title={title}
    className="modal-viewer-video modal-viewer-youtube"
    src={directEmbedUrl(embedUrl, targetTime)}
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowFullScreen
    referrerPolicy="strict-origin-when-cross-origin"
    tabIndex={0}
    onLoad={() => onTimeUpdate?.(targetTime)}
  />
);

/**
 * Create a YouTube IFrame API player and report its real playback clock.
 *
 * The placeholder ID and player options intentionally mirror the official API
 * example. If the API script cannot initialize, a direct iframe remains
 * available so a control-plane failure does not hide the video itself.
 */
const YouTubePlayer = ({
  embedUrl,
  targetTime,
  title,
  onTimeUpdate,
  onIframeFocus,
  onPlayerReady,
}) => {
  const playerIdRef = useRef(null);
  const playerRef = useRef(null);
  const targetTimeRef = useRef(targetTime);
  const onTimeUpdateRef = useRef(onTimeUpdate);
  const onIframeFocusRef = useRef(onIframeFocus);
  const onPlayerReadyRef = useRef(onPlayerReady);
  const [apiFailed, setApiFailed] = useState(false);

  if (!playerIdRef.current) {
    playerIdSequence += 1;
    playerIdRef.current = `hcmai-youtube-player-${playerIdSequence}`;
  }

  targetTimeRef.current = targetTime;
  onTimeUpdateRef.current = onTimeUpdate;
  onIframeFocusRef.current = onIframeFocus;
  onPlayerReadyRef.current = onPlayerReady;

  useEffect(() => {
    let disposed = false;
    let pollId = null;

    const stopPolling = () => {
      if (pollId !== null) window.clearInterval(pollId);
      pollId = null;
    };

    const reportCurrentTime = (player, fallback = targetTimeRef.current) => {
      try {
        const currentTime = Number(player.getCurrentTime?.());
        if (Number.isFinite(currentTime) && currentTime >= 0) {
          onTimeUpdateRef.current?.(currentTime);
          return currentTime;
        }
      } catch {
        // The player can be between states while buffering.
      }
      if (Number.isFinite(fallback) && fallback >= 0) {
        onTimeUpdateRef.current?.(fallback);
        return fallback;
      }
      return null;
    };

    const startPolling = (player) => {
      stopPolling();
      reportCurrentTime(player);
      pollId = window.setInterval(() => reportCurrentTime(player), CURRENT_TIME_POLL_MS);
    };

    const seekToTarget = (player) => {
      const requestedTime = Number(targetTimeRef.current);
      if (!Number.isFinite(requestedTime) || requestedTime < 0) return;

      let seekTime = requestedTime;
      try {
        const duration = Number(player.getDuration?.());
        if (Number.isFinite(duration) && duration > 0) {
          seekTime = Math.min(requestedTime, duration);
        }
        player.seekTo?.(seekTime, true);
        onTimeUpdateRef.current?.(seekTime);
      } catch (error) {
        console.error('[HCMAI YouTube] seek failed:', error);
      }
    };

    loadYouTubeIframeApi()
      .then((YT) => {
        if (disposed) return;

        const videoId = getYouTubeVideoId(embedUrl);
        if (!videoId) throw new Error(`Cannot extract YouTube video ID from ${embedUrl}`);

        const playingState = YT.PlayerState?.PLAYING ?? 1;
        const player = new YT.Player(playerIdRef.current, {
          width: '640',
          height: '360',
          videoId,
          playerVars: {
            autoplay: 0,
            origin: window.location.origin,
            playsinline: 1,
            rel: 0,
            ...(Number.isFinite(targetTimeRef.current) && targetTimeRef.current > 0
              ? { start: Math.floor(targetTimeRef.current) }
              : {}),
          },
          events: {
            onReady: (event) => {
              if (disposed) return;
              const iframe = event.target.getIframe?.();
              if (iframe) {
                iframe.title = title;
                iframe.className = 'modal-viewer-video modal-viewer-youtube';
                iframe.tabIndex = 0;
                if (onIframeFocusRef.current) {
                  iframe.addEventListener('focus', onIframeFocusRef.current);
                }
              }

              console.log('[HCMAI YouTube] READY');
              console.log('[HCMAI YouTube] state:', event.target.getPlayerState?.());
              seekToTarget(event.target);
              onPlayerReadyRef.current?.({
                seekBy: (seconds) => {
                  const currentTime = Number(event.target.getCurrentTime?.());
                  if (!Number.isFinite(currentTime)) return;
                  const duration = Number(event.target.getDuration?.());
                  const requestedTime = Math.max(0, currentTime + seconds);
                  const nextTime = Number.isFinite(duration) && duration > 0
                    ? Math.min(requestedTime, duration)
                    : requestedTime;
                  event.target.seekTo?.(nextTime, true);
                  onTimeUpdateRef.current?.(nextTime);
                },
                togglePlayPause: () => {
                  if (event.target.getPlayerState?.() === playingState) {
                    event.target.pauseVideo?.();
                  } else {
                    event.target.playVideo?.();
                  }
                },
              });
            },
            onStateChange: (event) => {
              if (disposed) return;
              const currentTime = reportCurrentTime(event.target);
              console.log('[HCMAI YouTube] STATE:', event.data);
              console.log('[HCMAI YouTube] time:', currentTime);
              if (event.data === playingState) startPolling(event.target);
              else stopPolling();
            },
            onError: (event) => {
              console.error('[HCMAI YouTube] YOUTUBE ERROR:', event.data, {
                videoId,
                embedUrl,
              });
            },
          },
        });
        playerRef.current = player;
      })
      .catch((error) => {
        if (disposed) return;
        console.error('[HCMAI YouTube] API initialization failed:', error);
        setApiFailed(true);
      });

    return () => {
      disposed = true;
      stopPolling();
      try {
        playerRef.current?.destroy?.();
      } catch {
        // The modal may already have removed the player host.
      }
      playerRef.current = null;
    };
  }, [embedUrl, title]);

  useEffect(() => {
    const player = playerRef.current;
    if (!player || !Number.isFinite(targetTime) || targetTime < 0) return;
    try {
      player.seekTo?.(targetTime, true);
      onTimeUpdate?.(targetTime);
    } catch (error) {
      console.error('[HCMAI YouTube] target seek failed:', error);
    }
  }, [targetTime, onTimeUpdate]);

  if (apiFailed) {
    return (
      <DirectFallback
        embedUrl={embedUrl}
        targetTime={targetTime}
        title={title}
        onTimeUpdate={onTimeUpdate}
      />
    );
  }

  return (
    <div className="modal-youtube-host" aria-label={title}>
      <div id={playerIdRef.current} />
    </div>
  );
};

export default YouTubePlayer;
