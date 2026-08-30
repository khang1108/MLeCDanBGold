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
