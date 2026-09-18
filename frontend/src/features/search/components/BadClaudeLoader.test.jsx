import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import BadClaudeLoader from "./BadClaudeLoader";

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    clearRect: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    bezierCurveTo: jest.fn(),
    lineTo: jest.fn(),
    stroke: jest.fn(),
    arc: jest.fn(),
    fill: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
  }));

  window.HTMLMediaElement.prototype.play = jest.fn(() => Promise.resolve());
  window.HTMLMediaElement.prototype.pause = jest.fn();
});

test("renders BadClaudeLoader with Searching animation and strike counter", () => {
  render(<BadClaudeLoader isVisible={true} />);

  expect(screen.getByTestId("gif-loader")).toBeTruthy();
  expect(screen.getByText(/Searching/i)).toBeTruthy();
  expect(screen.getByText(/Whip Strikes: 1/i)).toBeTruthy();
});

test("clicking anywhere cracks the whip, increments strikes, and triggers burst", () => {
  render(<BadClaudeLoader isVisible={true} />);

  const initialCount = screen.getByText(/Whip Strikes: 1/i);
  expect(initialCount).toBeTruthy();

  const loader = screen.getByTestId("gif-loader");
  fireEvent.click(loader);

  expect(screen.getByText(/Whip Strikes: 2/i)).toBeTruthy();
  const burstSvg = loader.querySelector(".badclaude-swoosh-svg");
  expect(burstSvg).toBeTruthy();

  const popText = loader.querySelector(".badclaude-pop-text");
  expect(popText).toBeTruthy();
  expect(popText.textContent.length).toBeGreaterThan(0);
});

test("returns null when isVisible is false", () => {
  const { container } = render(<BadClaudeLoader isVisible={false} />);
  expect(container.firstChild).toBeNull();
});
