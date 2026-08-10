import React, { useCallback, useEffect, useRef, useState } from "react";
import { searchFrames } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "./GifLoaderOverlay";

const QUERY_PREFIX = /^\/(kis|vkis)\b\s*/i;

// Standalone competition search workspace with frame results.
const AdHocSearchWorkspace = ({
  topK,
  setTopK,
  onFrameClick,
  queryInputRef,
  onFocusQueryInput,
  onBlurQueryInput,
}) => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [latencyMs, setLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => () => requestRef.current?.abort(), []);
  const handleSubmit = useCallback(
    async (event) => {
      event?.preventDefault();
      const trimmed = query.trim();
      if (!trimmed || isSearching) return;

      const prefixMatch = trimmed.match(QUERY_PREFIX);
      if (!prefixMatch) {
        setError(
          "Start your frame query with /kis or /vkis.",
        );
        return;
      }

      const queryType = prefixMatch[1].toLowerCase();
      const searchQuery = trimmed.slice(prefixMatch[0].length).trim();
      if (!searchQuery) {
        setError(`Enter a query after /${queryType}.`);
        return;
      }

      requestRef.current?.abort();
      const controller = new AbortController();
      requestRef.current = controller;
      setIsSearching(true);
      setError(null);

      try {
        const response = await searchFrames({
          query: searchQuery,
          topK,
          queryType,
          signal: controller.signal,
        });
        setResults(response.results || []);
        setWarnings(response.warnings || []);
        setLatencyMs(response.latency_ms || null);
      } catch (requestError) {
        if (requestError?.name === "AbortError") return;
        setError(requestError.message || "Failed to contact search API");
      } finally {
        if (requestRef.current === controller) {
          requestRef.current = null;
          setIsSearching(false);
        }
      }
    },
    [isSearching, query, topK],
  );

  const handleResetOptions = useCallback(() => {
    setTopK(20);
  }, [setTopK]);

  return (
    <div className="adhoc-workspace">
      <form className="adhoc-query-bar" onSubmit={handleSubmit}>
        <div className="query-input-wrapper">
          <svg
            className="query-search-icon"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.8}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.603 10.602Z"
            />
          </svg>
          <input
            ref={queryInputRef}
            type="text"
            className="input-text query-input-field"
            placeholder="Start with /kis or /vkis"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onFocus={onFocusQueryInput}
            onBlur={onBlurQueryInput}
            disabled={isSearching}
          />
        </div>
        <button
          type="submit"
          className="btn-primary query-submit-btn"
          disabled={isSearching || !query.trim()}
        >
          {isSearching ? "Searching..." : "Search"}
        </button>
      </form>

      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox
            topK={topK}
            setTopK={setTopK}
            onReset={handleResetOptions}
          />
        </aside>

        <section className="adhoc-results">
          <GifLoaderOverlay isVisible={isSearching} />
          <FramesBox
            results={results}
            isLoading={isSearching}
            error={error}
            latencyMs={latencyMs}
            warnings={warnings}
            onFrameClick={onFrameClick}
          />
        </section>
      </div>
    </div>
  );
};

export default AdHocSearchWorkspace;
