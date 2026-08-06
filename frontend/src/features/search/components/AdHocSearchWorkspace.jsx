import React, { useCallback, useEffect, useRef, useState } from "react";
import { fetchQuerySuggestions } from "../../../api/querySuggestions";
import { searchFrames } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "./GifLoaderOverlay";
import MiniChallengePanel from "../../minichallenge/components/MiniChallengePanel";
import { useMiniChallenge } from "../../minichallenge/hooks/useMiniChallenge";
import QuerySuggestions from "./QuerySuggestions";

const QUERY_PREFIX = /^\/(kis|kisc|vkis|vqa|trake)\b\s*/i;

// Standalone competition search workspace with query suggestions and frame results.
const AdHocSearchWorkspace = ({
  topK,
  setTopK,
  suggestionCount,
  setSuggestionCount,
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
  const challenge = useMiniChallenge();

  useEffect(() => () => requestRef.current?.abort(), []);
  const [suggestions, setSuggestions] = useState([]);
  const [suggestionError, setSuggestionError] = useState(null);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestionQueryType, setSuggestionQueryType] = useState("kis");
  const suggestionSequence = useRef(0);
  const suggestionAbortController = useRef(null);

  useEffect(
    () => () => suggestionAbortController.current?.abort(),
    [],
  );

  const loadSuggestions = useCallback(
    (searchQuery, requestSequence) => {
      const controller = new AbortController();
      suggestionAbortController.current = controller;
      setIsSuggesting(true);

      fetchQuerySuggestions({
        query: searchQuery,
        count: suggestionCount,
        signal: controller.signal,
      })
        .then((response) => {
          if (suggestionSequence.current !== requestSequence) return;
          setSuggestions(response.suggestions);
          setSuggestionError(null);
        })
        .catch((requestError) => {
          if (
            requestError?.name === "AbortError" ||
            suggestionSequence.current !== requestSequence
          ) {
            return;
          }
          setSuggestions([]);
          setSuggestionError(
            requestError.message || "Failed to generate query suggestions",
          );
        })
        .finally(() => {
          if (suggestionSequence.current === requestSequence) {
            setIsSuggesting(false);
          }
        });
    },
    [suggestionCount],
  );

  const handleSubmit = useCallback(
    async (event) => {
      event?.preventDefault();
      const trimmed = query.trim();
      if (!trimmed || isSearching) return;

      const prefixMatch = trimmed.match(QUERY_PREFIX);
      if (!prefixMatch) {
        setError(
          "Start your query with /kis, /kisc, /vkis, /vqa, or /trake.",
        );
        return;
      }

      const queryType = prefixMatch[1].toLowerCase();
      const searchQuery = trimmed.slice(prefixMatch[0].length).trim();
      if (!searchQuery) {
        setError(`Enter a query after /${queryType}.`);
        return;
      }

      setIsSearching(true);
      setError(null);

      try {
        const response = await searchFrames({
          query: searchQuery,
          topK,
          queryType,
        });
        setResults(response.results || []);
        setWarnings(response.warnings || []);
        setLatencyMs(response.latency_ms || null);
      } catch (requestError) {
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

  const handleSuggest = useCallback(() => {
    const trimmed = query.trim();
    if (!trimmed || isSuggesting || isSearching) return;

    const prefixMatch = trimmed.match(QUERY_PREFIX);
    if (!prefixMatch) {
      setSuggestions([]);
      setSuggestionError(
        "Start your query with /kis, /kisc, /vkis, /vqa, or /trake.",
      );
      return;
    }

    const queryType = prefixMatch[1].toLowerCase();
    const searchQuery = trimmed.slice(prefixMatch[0].length).trim();
    if (!searchQuery) {
      setSuggestions([]);
      setSuggestionError(`Enter a query after /${queryType}.`);
      return;
    }

    suggestionAbortController.current?.abort();
    const requestSequence = suggestionSequence.current + 1;
    suggestionSequence.current = requestSequence;
    setSuggestionQueryType(queryType);
    setSuggestions([]);
    setSuggestionError(null);
    loadSuggestions(searchQuery, requestSequence);
  }, [isSearching, isSuggesting, loadSuggestions, query]);
  const handleSuggestionSelect = useCallback(
    (suggestedQuery) => {
      const trimmed = suggestedQuery.trim();
      setQuery(
        QUERY_PREFIX.test(trimmed)
          ? trimmed
          : `/${suggestionQueryType} ${trimmed}`,
      );
      window.requestAnimationFrame(() => {
        queryInputRef.current?.focus();
        queryInputRef.current?.setSelectionRange(
          queryInputRef.current.value.length,
          queryInputRef.current.value.length,
        );
      });
    },
    [queryInputRef, suggestionQueryType],
  );

  const handleResetOptions = useCallback(() => {
    setTopK(20);
    setSuggestionCount(5);
  }, [setSuggestionCount, setTopK]);

  const handleChallengeSubmit = useCallback((frame) => {
    const taskName = challenge.currentTask?.name;
    if (!taskName) return;
    const confirmed = window.confirm(
      `Submit ${frame.video_id} to “${taskName}”?`,
    );
    if (confirmed) challenge.submitFrame(frame);
  }, [challenge]);

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
            placeholder="Start with /kis, /kisc, /vkis, /vqa, or /trake"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onFocus={onFocusQueryInput}
            onBlur={onBlurQueryInput}
            disabled={isSearching}
          />
        </div>
        <button
          type="button"
          className="btn-utility query-suggest-btn"
          disabled={isSearching || isSuggesting || !query.trim()}
          onClick={handleSuggest}
        >
          {isSuggesting ? "Suggesting..." : "Suggest"}
        </button>
        <button
          type="submit"
          className="btn-primary query-submit-btn"
          disabled={isSearching || !query.trim()}
        >
          {isSearching ? "Searching..." : "Search"}
        </button>
      </form>

      <QuerySuggestions
        suggestions={suggestions}
        isLoading={isSuggesting}
        error={suggestionError}
        onSelect={handleSuggestionSelect}
      />

      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox
            topK={topK}
            setTopK={setTopK}
            suggestionCount={suggestionCount}
            setSuggestionCount={setSuggestionCount}
            onReset={handleResetOptions}
          />
          <MiniChallengePanel challenge={challenge} />
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
            onChallengeSubmit={challenge.currentTask ? handleChallengeSubmit : null}
            submittingFrameId={challenge.submittingFrameId}
          />
        </section>
      </div>
    </div>
  );
};

export default AdHocSearchWorkspace;