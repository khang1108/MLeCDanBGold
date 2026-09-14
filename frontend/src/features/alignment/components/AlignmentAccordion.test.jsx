import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
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
