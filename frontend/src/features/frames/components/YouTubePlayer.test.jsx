import React from 'react';
import { act, render, screen, waitFor } from '@testing-library/react';
import YouTubePlayer from './YouTubePlayer';

let fakePlayer;
let fakePlayerOptions;
let fakePlayerTarget;
let logSpy;
let errorSpy;

beforeEach(() => {
  fakePlayer = {
    getCurrentTime: jest.fn(() => 5.2),
    getDuration: jest.fn(() => 20),
    getPlayerState: jest.fn(() => 2),
    seekTo: jest.fn(),
    pauseVideo: jest.fn(),
    playVideo: jest.fn(),
    destroy: jest.fn(),
  };
  fakePlayerOptions = null;
  fakePlayerTarget = null;
  logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
  errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
  window.YT = {
    PlayerState: { PLAYING: 1 },
    Player: jest.fn((element, options) => {
      fakePlayerTarget = element;
      fakePlayerOptions = options;
      return fakePlayer;
    }),
  };
});

afterEach(() => {
  delete window.YT;
  logSpy.mockRestore();
  errorSpy.mockRestore();
});

const renderPlayer = (props = {}) => render(
  <YouTubePlayer
    embedUrl="https://www.youtube.com/embed/Rzpw5WR7nAY?enablejsapi=1"
    targetTime={5.2}
    title="Video for L21_V001"
    {...props}
  />,
);

test('creates a YT.Player host with the requested video and origin', async () => {
  renderPlayer();

  await waitFor(() => expect(fakePlayerOptions).not.toBeNull());

  expect(fakePlayerTarget).toMatch(/^hcmai-youtube-player-/);
  expect(document.getElementById(fakePlayerTarget)).toBeTruthy();
  expect(fakePlayerOptions.videoId).toBe('Rzpw5WR7nAY');
  expect(fakePlayerOptions.playerVars).toEqual(expect.objectContaining({
    autoplay: 0,
    origin: window.location.origin,
    playsinline: 1,
  }));
});

test('logs READY and seeks to the selected timestamp', async () => {
  const onTimeUpdate = jest.fn();
  renderPlayer({ onTimeUpdate });

  await waitFor(() => expect(fakePlayerOptions).not.toBeNull());
  act(() => fakePlayerOptions.events.onReady({ target: fakePlayer }));

  expect(logSpy).toHaveBeenCalledWith('[HCMAI YouTube] READY');
  expect(logSpy).toHaveBeenCalledWith('[HCMAI YouTube] state:', 2);
  expect(fakePlayer.seekTo).toHaveBeenCalledWith(5.2, true);
  expect(fakePlayer.pauseVideo).not.toHaveBeenCalled();
  expect(onTimeUpdate).toHaveBeenCalledWith(5.2);
});

test('logs state, time, and YouTube errors for browser debugging', async () => {
  renderPlayer();

  await waitFor(() => expect(fakePlayerOptions).not.toBeNull());
  act(() => fakePlayerOptions.events.onStateChange({ data: 1, target: fakePlayer }));
  act(() => fakePlayerOptions.events.onError({ data: 150 }));

  expect(logSpy).toHaveBeenCalledWith('[HCMAI YouTube] STATE:', 1);
  expect(logSpy).toHaveBeenCalledWith('[HCMAI YouTube] time:', 5.2);
  expect(errorSpy).toHaveBeenCalledWith(
    '[HCMAI YouTube] YOUTUBE ERROR:',
    150,
    expect.objectContaining({ videoId: 'Rzpw5WR7nAY' }),
  );
});
