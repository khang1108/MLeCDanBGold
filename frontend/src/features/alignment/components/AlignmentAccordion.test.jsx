import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import AlignmentAccordion, { formatTimestampMs } from "./AlignmentAccordion";
import { keyframeUrl } from "../../../api/keyframes";

test("reveals aligned events with their canonical timestamps and keyframes", () => {
  render(
    <AlignmentAccordion
      events={["hold", "roll"]}
      frameIds={["f1", "f2"]}
      timestampsMs={[1200, 2400]}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: /alignment/i }));

  expect(screen.getByText("hold")).toBeTruthy();
  expect(screen.getByText("00:01.200")).toBeTruthy();
  expect(screen.getByAltText(/f1/i).getAttribute("src"))
    .toBe(keyframeUrl("f1"));
});

test("formats sub-hour and hour-long timestamps consistently", () => {
  expect(formatTimestampMs(1200)).toBe("00:01.200");
  expect(formatTimestampMs(3_661_200)).toBe("01:01:01.200");
});

test("opens non-destructive preview popup on thumbnail click and closes on close button or escape", () => {
  const handleSeek = jest.fn();
  render(
    <AlignmentAccordion
      events={["person walking in park"]}
      frameIds={["frame_001"]}
      timestampsMs={[3500]}
      onSeek={handleSeek}
      collapsible={false}
    />,
  );

  // Preview overlay should not exist initially
  expect(screen.queryByRole("dialog", { name: /keyframe preview/i })).toBeNull();

  // Click on thumbnail wrapper to open preview
  const thumbnail = screen.getByAltText(/aligned frame frame_001/i);
  fireEvent.click(thumbnail);

  // Dialog is visible
  const dialog = screen.getByRole("dialog", { name: /keyframe preview/i });
  expect(dialog).toBeTruthy();
  expect(within(dialog).getByText("Keyframe Preview")).toBeTruthy();
  expect(within(dialog).getByText("person walking in park")).toBeTruthy();
  expect(within(dialog).getByAltText(/keyframe preview frame_001/i).getAttribute("src"))
    .toBe(keyframeUrl("frame_001"));

  // Seek button inside preview works
  const seekBtn = screen.getByRole("button", { name: /tua video/i });
  expect(seekBtn).toBeTruthy();

  // Close dialog via close button
  const closeBtn = screen.getByRole("button", { name: /close preview/i });
  fireEvent.click(closeBtn);
  expect(screen.queryByRole("dialog", { name: /keyframe preview/i })).toBeNull();

  // Open again and close with Escape
  fireEvent.click(screen.getByAltText(/aligned frame frame_001/i));
  expect(screen.getByRole("dialog", { name: /keyframe preview/i })).toBeTruthy();
  fireEvent.keyDown(window, { key: "Escape" });
  expect(screen.queryByRole("dialog", { name: /keyframe preview/i })).toBeNull();
});
