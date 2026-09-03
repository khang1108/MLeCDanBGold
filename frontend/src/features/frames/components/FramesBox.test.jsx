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

test("rounds latency display values to 2 decimal places in summary and stages", () => {
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

  expect(screen.getByText("62.71ms")).toBeTruthy();
  expect(screen.getByText("Query: 12.35ms")).toBeTruthy();
  expect(screen.getByText("Retrieval: 45.68ms")).toBeTruthy();
  expect(screen.getByText("Alignment: 3.46ms")).toBeTruthy();
  expect(screen.getByText("Materialize: 1.23ms")).toBeTruthy();
});
