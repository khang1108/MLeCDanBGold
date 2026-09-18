import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import BadClaudeLoader from "./BadClaudeLoader";

// Mock HTMLCanvasElement getContext
beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = jest.fn(() => ({
    clearRect: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    stroke: jest.fn(),
    arc: jest.fn(),
    fill: jest.fn(),
    save: jest.fn(),
    restore: jest.fn(),
    createLinearGradient: jest.fn(() => ({
      addColorStop: jest.fn(),
    })),
  }));
});

test("renders BadClaudeLoader with slogans and whip count", () => {
  render(<BadClaudeLoader isVisible={true} />);

  expect(screen.getByTestId("gif-loader")).toBeTruthy();
  expect(screen.getByText(/BadClaude Accelerator/i)).toBeTruthy();
  expect(screen.getByText(/Whip Strikes:/i)).toBeTruthy();
  expect(screen.getByText(/WORK FASTER!/i)).toBeTruthy();
});

test("clicking anywhere cracks the whip and increments whip strikes", () => {
  render(<BadClaudeLoader isVisible={true} />);

  const initialCount = screen.getByText(/Whip Strikes: 1/i);
  expect(initialCount).toBeTruthy();

  const overlay = screen.getByTestId("gif-loader");
  fireEvent.click(overlay);

  expect(screen.getByText(/Whip Strikes: 2/i)).toBeTruthy();
});

test("returns null when isVisible is false", () => {
  const { container } = render(<BadClaudeLoader isVisible={false} />);
  expect(container.firstChild).toBeNull();
});
