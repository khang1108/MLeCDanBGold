import React from "react";

/**
 * Clean, lightweight search loader with a smooth spinner.
 * Replaces the previous interactive whip physics and audio with a clean, focused indicator.
 */
const BadClaudeLoader = ({ isVisible = true }) => {
  if (!isVisible) return null;

  return (
    <div
      className="search-loader-container"
      data-testid="gif-loader"
      role="status"
      aria-label="Searching"
    >
      <div className="search-loader-spinner" aria-hidden="true" />
      <span className="search-loader-title">Searching…</span>
      <p className="search-loader-subtext">Retrieving multimodal video frame evidence</p>
    </div>
  );
};

export default BadClaudeLoader;
