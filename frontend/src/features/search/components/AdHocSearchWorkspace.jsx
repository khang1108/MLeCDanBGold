import React, { useCallback, useEffect, useRef, useState } from "react";
import { searchFrames } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "./GifLoaderOverlay";

const QUERY_PREFIX = /^\/(kis)\b\s*/i;
const SEARCH_ID_KEY = "hcmai.progressive.kis.search_id";

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
  const [searchId, setSearchId] = useState(
    () => window.sessionStorage.getItem(SEARCH_ID_KEY),
  );
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
          "Start your frame query with /kis.",
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
          searchId,
          signal: controller.signal,
        });
        if (response.search_id) {
          setSearchId(response.search_id);
          window.sessionStorage.setItem(SEARCH_ID_KEY, response.search_id);
        }
        setResults(response.results || []);
        setWarnings(response.warnings || []);
        setLatencyMs(response.latency_ms || null);
      } catch (requestError) {
        if (requestError?.name === "AbortError") return;
        if (requestError?.status === 410) {
          setSearchId(null);
          window.sessionStorage.removeItem(SEARCH_ID_KEY);
        }
        setError(requestError.message || "Failed to contact search API");
      } finally {
        if (requestRef.current === controller) {
          requestRef.current = null;
          setIsSearching(false);
        }
      }
    },
    [isSearching, query, searchId, topK],
  );

  const handleNewQuestion = useCallback(() => {
    setSearchId(null);
    window.sessionStorage.removeItem(SEARCH_ID_KEY);
    setQuery("");
    setResults([]);
    setWarnings([]);
    setLatencyMs(null);
    setError(null);
  }, []);

  const handleResetOptions = useCallback(() => {
    setTopK(20);
  }, [setTopK]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (
        event.key.toLowerCase() === 'n' &&
        event.target.tagName !== 'INPUT' &&
        event.target.tagName !== 'TEXTAREA'
      ) {
        event.preventDefault();
        handleNewQuestion();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNewQuestion]);

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
            placeholder="Start with /kis"
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
        <button type="button" className="btn-secondary" onClick={handleNewQuestion} title="Shortcut: N">
          New Question
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
