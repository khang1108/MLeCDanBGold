import React from "react";
import { render, screen } from "@testing-library/react";
import FramesBox from "./FramesBox";

jest.mock("./FrameCard", () => ({ frame }) => (
  <div data-testid="frame-card">{frame.frame_ids.join(" → ")}</div>
));

test("renders distinct DP paths that share a representative frame without duplicate keys", () => {
  const consoleError = jest.spyOn(console, "error").mockImplementation(() => {});
  const shared = {
    frame_id: "f5",
    video_id: "V01",
    frame_idx: 5,
  };

  render(
    <FramesBox
      results={[
        { ...shared, frame_ids: ["f1", "f5", "f9"] },
        { ...shared, frame_ids: ["f2", "f5", "f10"] },
      ]}
      isLoading={false}
      error={null}
      latencyMs={1}
    />,
  );

  expect(screen.getAllByTestId("frame-card")).toHaveLength(6);
  expect(consoleError.mock.calls.flat().join(" ")).not.toContain(
    "same key",
  );
  consoleError.mockRestore();
});

test("rounds latency display values in seconds in summary and stages", () => {
  render(
    <FramesBox
      results={[{ frame_id: "f1", video_id: "V01", frame_idx: 1, frame_ids: ["f1"] }]}
      isLoading={false}
      error={null}
      latencyMs={{
        query_ms: 12.345,
        retrieval_ms: 45.678,
        alignment_ms: 3.456,
        materialization_ms: 1.234,
        total_ms: 62.713,
      }}
    />,
  );

  expect(screen.getByText("0.06s")).toBeTruthy();
  expect(screen.getByText("Query: 0.01s")).toBeTruthy();
  expect(screen.getByText("Retrieval: 0.05s")).toBeTruthy();
  expect(screen.getByText("Alignment: 0.00s")).toBeTruthy();
  expect(screen.getByText("Materialize: 0.00s")).toBeTruthy();
});

test("renders GifLoaderOverlay when isLoading is true and results is empty", () => {
  render(
    <FramesBox
      results={[]}
      isLoading={true}
      error={null}
      latencyMs={null}
    />,
  );

  expect(screen.getByTestId("gif-loader")).toBeTruthy();
  expect(screen.queryByTestId("hcmus-copyright-badge")).toBeNull();
});

test("renders copyright empty state when not loading and no search has occurred", () => {
  render(
    <FramesBox
      results={[]}
      isLoading={false}
      error={null}
      latencyMs={null}
    />,
  );

  expect(screen.getByTestId("hcmus-copyright-badge")).toBeTruthy();
  expect(screen.getByRole("heading", { name: /MLeCDanBGold/ })).toBeTruthy();
  expect(screen.queryByTestId("gif-loader")).toBeNull();
});

test("renders no-frames empty state when not loading and search returned empty", () => {
  render(
    <FramesBox
      results={[]}
      isLoading={false}
      error={null}
      latencyMs={100}
    />,
  );

  expect(screen.getByText("No frames found matching your query")).toBeTruthy();
  expect(screen.queryByTestId("gif-loader")).toBeNull();
});

test("renders EventTrail start button when context and result_id exist", () => {
  const openTrail = jest.fn();
  const eventTrailContext = {
    snapshotId: "snap-123",
    kisRevision: 1,
    events: [{ id: "E1", text: "event 1" }],
    searchSessionId: "snap-123",
  };

  render(
    <FramesBox
      results={[
        {
          result_id: "res-1",
          frame_id: "f1",
          video_id: "V01",
          frame_idx: 1,
          frame_ids: ["f1"],
          timestamp_ms: 1000,
        },
      ]}
      isLoading={false}
      error={null}
      latencyMs={50}
      eventTrailContext={eventTrailContext}
      eventTrail={{ open: openTrail, session: null, pending: false }}
    />,
  );

  const startBtn = screen.getByRole("button", { name: /eventtrail/i });
  expect(startBtn).toBeTruthy();
});

test("renders active trail revision, undo and exit buttons when row trail is active", () => {
  const undo = jest.fn();
  const close = jest.fn();
  const eventTrailContext = {
    snapshotId: "snap-123",
    kisRevision: 1,
    events: [{ id: "E1", text: "event 1" }],
    searchSessionId: "snap-123",
  };
  const activeTrailSession = {
    session_id: "trail-sess-1",
    result_id: "res-1",
    trail_revision: 2,
    status: "active",
    path: [{ event_id: "E1", frame_id: "f1", frame_idx: 1, timestamp_ms: 1000 }],
    approved_event_ids: ["E1"],
    rejected_counts: {},
  };

  render(
    <FramesBox
      results={[
        {
          result_id: "res-1",
          frame_id: "f1",
          video_id: "V01",
          frame_idx: 1,
          frame_ids: ["f1"],
          timestamp_ms: 1000,
        },
      ]}
      isLoading={false}
      error={null}
      latencyMs={50}
      eventTrailContext={eventTrailContext}
      eventTrail={{
        open: jest.fn(),
        undo,
        close,
        session: activeTrailSession,
        pending: false,
      }}
    />,
  );

  expect(screen.getByText(/Trail Rev 2/i)).toBeTruthy();
  expect(screen.getByRole("button", { name: /undo/i })).toBeTruthy();
  expect(screen.getByRole("button", { name: /exit trail/i })).toBeTruthy();
});

test("renders updated candidate frames from activeTrailSession.path when row trail is active", () => {
  const eventTrailContext = {
    snapshotId: "snap-123",
    kisRevision: 1,
    events: [{ id: "E1", text: "event 1" }],
    searchSessionId: "snap-123",
  };
  const activeTrailSession = {
    session_id: "trail-sess-1",
    result_id: "res-1",
    trail_revision: 2,
    status: "active",
    path: [{ event_id: "E1", frame_id: "f2_new", frame_idx: 2, timestamp_ms: 2000 }],
    approved_event_ids: [],
    rejected_counts: { E1: 1 },
  };

  render(
    <FramesBox
      results={[
        {
          result_id: "res-1",
          frame_id: "f1_old",
          video_id: "V01",
          frame_idx: 1,
          frame_ids: ["f1_old"],
          timestamp_ms: 1000,
        },
      ]}
      isLoading={false}
      error={null}
      latencyMs={50}
      eventTrailContext={eventTrailContext}
      eventTrail={{
        open: jest.fn(),
        undo: jest.fn(),
        close: jest.fn(),
        session: activeTrailSession,
        pending: false,
      }}
    />,
  );

  expect(screen.getByTestId("frame-card").textContent).toBe("f2_new");
});


