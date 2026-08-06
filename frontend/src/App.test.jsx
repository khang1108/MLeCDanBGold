import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import App from "./App";

jest.mock("./features/search/components/AdHocSearchWorkspace", () => (
  function FakeKisWorkspace() {
    return <div>KIS workspace</div>;
  }
));
jest.mock("./features/vqa/components/VqaSearchWorkspace", () => (
  function FakeVqaWorkspace() {
    return <div>QA workspace</div>;
  }
));
jest.mock("./features/health/hooks/useHealthCheck", () => ({
  useHealthCheck: () => ({ isHealthy: true, healthData: {}, isChecking: false }),
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

afterEach(() => {
  window.history.replaceState({}, "", "/");
});

test("mounts QA at /qa and lets the user switch back to KIS", () => {
  window.history.replaceState({}, "", "/qa");
  render(<App />);

  expect(screen.getByText("QA workspace")).toBeTruthy();
  expect(screen.getByRole("link", { name: "QA" }).getAttribute("aria-current"))
    .toBe("page");

  fireEvent.click(screen.getByRole("link", { name: "KIS" }));

  expect(window.location.pathname).toBe("/");
  expect(screen.getByText("KIS workspace")).toBeTruthy();
});

test("task selector navigates from KIS to /qa", () => {
  render(<App />);

  fireEvent.click(screen.getByRole("link", { name: "QA" }));

  expect(window.location.pathname).toBe("/qa");
  expect(screen.getByText("QA workspace")).toBeTruthy();
});
