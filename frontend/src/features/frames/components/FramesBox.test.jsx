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
