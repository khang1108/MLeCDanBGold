import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import VideoTimeline from "./VideoTimeline";

const setTrackBounds = (track, { left = 0, width = 200 } = {}) => {
  Object.defineProperty(track, "getBoundingClientRect", {
    configurable: true,
    value: () => ({ left, width, top: 0, right: left + width, bottom: 10, height: 10 }),
  });
};

const createVideoRef = ({ paused = true } = {}) => ({
  current: {
    paused,
    ended: false,
    muted: true,
    volume: 1,
    play: jest.fn(() => Promise.resolve()),
    pause: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
  },
});

test("shows the deterministic 1fps thumbnail while hovering the timeline", () => {
  const videoRef = createVideoRef();
  const onSeek = jest.fn();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={2}
      duration={10}
      onSeek={onSeek}
    />,
  );

  const track = screen.getByTestId("video-timeline-track");
  setTrackBounds(track);
  fireEvent.mouseMove(track, { clientX: 100 });

  expect(screen.getByAltText("Preview at 00:05").getAttribute("src"))
    .toContain("/api/v1/keyframes/L28_V001_raw1fps_000000005");
  expect(onSeek).not.toHaveBeenCalled();
});

test("seeking changes playback only after the range is changed", () => {
  const videoRef = createVideoRef();
  const onSeek = jest.fn();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={2}
      duration={10}
      onSeek={onSeek}
    />,
  );

  fireEvent.change(screen.getByRole("slider", { name: "Video timeline" }), {
    target: { value: "7" },
  });

  expect(onSeek).toHaveBeenCalledWith(7);
});

test("toggles the actual video element", () => {
  const videoRef = createVideoRef();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={0}
      duration={10}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "Play video" }));
  expect(videoRef.current.play).toHaveBeenCalledTimes(1);
});

test("controls mute state and volume on the actual video element", () => {
  const videoRef = createVideoRef();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={0}
      duration={10}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "Unmute video" }));
  expect(videoRef.current.muted).toBe(false);

  fireEvent.change(screen.getByRole("slider", { name: "Video volume" }), {
    target: { value: "0.4" },
  });
  expect(videoRef.current.volume).toBe(0.4);
  expect(videoRef.current.muted).toBe(false);

  fireEvent.click(screen.getByRole("button", { name: "Mute video" }));
  expect(videoRef.current.muted).toBe(true);
});

test("renders submit button and triggers onSubmit when provided", () => {
  const videoRef = createVideoRef();
  const onSubmit = jest.fn();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={3.5}
      duration={10}
      onSubmit={onSubmit}
    />,
  );

  const submitBtn = screen.getByRole("button", { name: /submit/i });
  expect(submitBtn).toBeTruthy();
  fireEvent.click(submitBtn);
  expect(onSubmit).toHaveBeenCalledTimes(1);
});

test("steps backward and forward by frames using step buttons", () => {
  const videoRef = createVideoRef();
  videoRef.current.currentTime = 5.0;
  const onSeek = jest.fn();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={5.0}
      duration={10}
      onSeek={onSeek}
    />,
  );

  const prevFrameBtn = screen.getByRole("button", { name: "Previous frame" });
  fireEvent.click(prevFrameBtn);
  expect(onSeek).toHaveBeenCalledWith(4.96);

  const nextFrameBtn = screen.getByRole("button", { name: "Next frame" });
  fireEvent.click(nextFrameBtn);
  expect(onSeek).toHaveBeenLastCalledWith(5);
});

test("opens playback speed menu and sets video playback rate", () => {
  const videoRef = createVideoRef();
  videoRef.current.playbackRate = 1;
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={2}
      duration={10}
    />,
  );

  const speedBtn = screen.getByRole("button", { name: /playback speed: 1x/i });
  fireEvent.click(speedBtn);

  const speed15x = screen.getByText("1.5x");
  fireEvent.click(speed15x);

  expect(videoRef.current.playbackRate).toBe(1.5);
});

test("toggles between elapsed and remaining time on click", () => {
  const videoRef = createVideoRef();
  render(
    <VideoTimeline
      videoId="L28_V001"
      videoRef={videoRef}
      currentTime={3}
      duration={10}
    />,
  );

  const timeReadout = screen.getByText("00:03 / 00:10");
  fireEvent.click(timeReadout);
  expect(screen.getByText("-00:07 / 00:10")).toBeTruthy();

  fireEvent.click(timeReadout);
  expect(screen.getByText("00:03 / 00:10")).toBeTruthy();
});

