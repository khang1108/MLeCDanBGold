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

  expect(screen.getAllByTestId("frame-card")).toHaveLength(2);
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
  expect(screen.queryByText("Welcome to HCMAI Frame Search")).toBeNull();
});

test("renders welcome empty state when not loading and no search has occurred", () => {
  render(
    <FramesBox
      results={[]}
      isLoading={false}
      error={null}
      latencyMs={null}
    />,
  );

  expect(screen.getByText("Welcome to HCMAI Frame Search")).toBeTruthy();
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

