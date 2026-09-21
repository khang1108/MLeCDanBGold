import React from "react";
import { render, screen } from "@testing-library/react";
import BadClaudeLoader from "./BadClaudeLoader";

test("renders BadClaudeLoader with Searching animation", () => {
  render(<BadClaudeLoader isVisible={true} />);

  expect(screen.getByTestId("gif-loader")).toBeTruthy();
  expect(screen.getByText(/Searching/i)).toBeTruthy();
});

test("returns null when isVisible is false", () => {
  const { container } = render(<BadClaudeLoader isVisible={false} />);
  expect(container.firstChild).toBeNull();
});
