import React from "react";
import BadClaudeLoader from "./BadClaudeLoader";

// Backward-compatible export redirecting to BadClaudeLoader
const GifLoaderOverlay = (props) => {
  return <BadClaudeLoader {...props} />;
};

export default GifLoaderOverlay;
