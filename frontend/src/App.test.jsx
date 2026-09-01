import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("./features/search/components/SearchWorkspace", () => (
  function FakeUnifiedWorkspace({ onFrameClick }) {
    const frame = {
      frame_id: "f1",
      video_id: "V01",
      frame_idx: 125,
      timestamp_ms: 5000,
    };
    return (
      <div>
        Unified search workspace
        <button
          type="button"
          onClick={() => onFrameClick({ frame, submissionMode: "kis" })}
        >
          Open KIS inspector
        </button>
        <button
          type="button"
          onClick={() => onFrameClick({ frame, submissionMode: "none" })}
        >
          Open TRAKE inspector
        </button>
      </div>
    );
  }
));
jest.mock("./features/health/hooks/useHealthCheck", () => ({
  useHealthCheck: () => ({ isHealthy: true, healthData: {} }),
}));
jest.mock("./features/vim/hooks/useVimMode", () => ({
  useVimMode: () => ({
    mode: "NORMAL",
    enterInsertMode: jest.fn(),
    enterNormalMode: jest.fn(),
    setMode: jest.fn(),
    isTopKOpen: false,
    setIsTopKOpen: jest.fn(),
    isHelpOpen: false,
    setIsHelpOpen: jest.fn(),
  }),
}));

test("mounts one shared search workspace without task tabs", () => {
  render(<App />);

  expect(screen.getByText("Unified search workspace")).toBeTruthy();
  expect(screen.queryByRole("navigation", { name: "Task selection" })).toBeNull();
});

test("only KIS frame selections expose inspector submission", () => {
  render(<App />);

  fireEvent.click(screen.getByRole("button", { name: "Open TRAKE inspector" }));
  expect(screen.queryByRole("button", { name: /submit current frame/i })).toBeNull();

  fireEvent.click(screen.getByRole("button", { name: "Close popup" }));
  fireEvent.click(screen.getByRole("button", { name: "Open KIS inspector" }));
  expect(screen.getByRole("button", { name: /submit current frame/i })).toBeTruthy();
});
