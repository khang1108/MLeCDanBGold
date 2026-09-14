import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import ExplorationPanel from "./ExplorationPanel";

const session = {
  view: {
    video_id: "V01",
    events: ["first event", "second event"],
    paths: [{ frame_ids: ["f1", "f2"], timestamps_ms: [1000, 2000] }],
    conditions: { window: [0, 10_000] },
    can_undo: true,
  },
};

test("selects E2, captures a valid interval, approves, and supports undo", () => {
  const onApprove = jest.fn();
  const onUndo = jest.fn();
  const readCurrentTimeMs = jest.fn()
    .mockReturnValueOnce(1234)
    .mockReturnValueOnce(5678);

  const { rerender } = render(
    <ExplorationPanel
      events={session.view.events}
      session={session}
      pending={false}
      onApprove={onApprove}
      onUndo={onUndo}
      readCurrentTimeMs={readCurrentTimeMs}
    />,
  );

  fireEvent.click(screen.getByRole("button", { name: "E2" }));
  fireEvent.click(screen.getByRole("button", { name: "Use current time as start" }));
  fireEvent.click(screen.getByRole("button", { name: "Use current time as end" }));
  expect(onApprove).not.toHaveBeenCalled();
  rerender(
    <ExplorationPanel
      events={session.view.events}
      session={{ ...session, view: { ...session.view, revision: 2, status: "no_valid_path", paths: [] } }}
      pending={false}
      onApprove={onApprove}
      onUndo={onUndo}
      readCurrentTimeMs={readCurrentTimeMs}
    />,
  );
  expect(screen.getByText("Event: second event")).toBeTruthy();
  expect(screen.getByRole("status").textContent).toContain("no valid aligned path");
  fireEvent.click(screen.getByRole("button", { name: "Approve" }));

  expect(onApprove).toHaveBeenCalledWith({
    event_index: 1,
    interval: [1234, 5678],
  });
  fireEvent.click(screen.getByRole("button", { name: "Undo" }));
  expect(onUndo).toHaveBeenCalledTimes(1);
});
