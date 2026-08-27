import React from "react";
import { render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("./features/search/components/SearchWorkspace", () => (
  function FakeUnifiedWorkspace() {
    return <div>Unified search workspace</div>;
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
