import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { searchFrames, searchTrake, suggestQueries } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import QuerySuggestionsPanel from "../../search-controls/components/QuerySuggestionsPanel";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import { displayVideoId } from "../../frames/videoSource";
import { useSubmission } from "../../submission/contexts/SubmissionContext";
import TrakePathCard from "./TrakePathCard";

const TRAKE_EVENT_MARKER = /\bE(\d+)\b\s*:?\s*/gi;
const SESSION_FINGERPRINT_KEY = "hcmai.session.fingerprint";
const SEARCH_ID_PREFIX = "hcmai.progressive.search_id";
const PROGRESSIVE_TASKS = ["kis"];

export const parseRetrievalDescription = (description) => {
  const query = description.trim();
  return query ? { queryType: "kis", query } : null;
};

export const parseTrakeEvents = (description) => {
  const text = description.trim();
  const markers = Array.from(text.matchAll(TRAKE_EVENT_MARKER));
  if (!markers.length || Number(markers[0][1]) !== 1) return null;
  if (markers.length < 2) return [];

  const events = markers.map((marker, index) => {
    if (Number(marker[1]) !== index + 1) return null;
    const start = marker.index + marker[0].length;
    const end = markers[index + 1]?.index ?? text.length;
    const event = text.slice(start, end).trim();
    return event || null;
  });

  return events.some((event) => !event) ? [] : events;
};

const sessionFingerprint = () => {
  let fingerprint = window.sessionStorage.getItem(SESSION_FINGERPRINT_KEY);
  if (!fingerprint) {
    fingerprint = window.crypto?.randomUUID?.()
      || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    window.sessionStorage.setItem(SESSION_FINGERPRINT_KEY, fingerprint);
  }
  return fingerprint;
};

export const progressiveSearchIdKey = (task) => (
  `${SEARCH_ID_PREFIX}.${task}.${sessionFingerprint()}`
);

export const TrakeResults = ({ events, paths, warnings, error, hasSearched, onFrameClick, onTrakeSubmit }) => (
  <section className="frames-container" aria-label="TRAKE ordered paths">
    {error && (
      <div className="error-alert" role="alert">
        <div className="error-details">
          <h4 className="error-title">TRAKE Search Error</h4>
          <p className="error-message">{error}</p>
        </div>
      </div>
    )}
    {!error && warnings.length > 0 && (
      <div className="search-warning" role="status">
        <span>Server note:</span>
        <ul>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
      </div>
    )}
    {!error && paths.length > 0 && (
      <div className="frames-scroll-region">
        {paths.map((path, index) => (
          <TrakePathCard
            key={`${path.video_id}-${index}`}
            events={events}
            path={path}
            onFrameClick={onFrameClick}
            onSubmit={onTrakeSubmit}
          />
        ))}
      </div>
    )}
    {!error && hasSearched && paths.length === 0 && (
      <div className="frames-empty-state">
        <p className="body-md frames-empty-text">No ordered TRAKE paths found</p>
      </div>
    )}
  </section>
);

const SearchWorkspace = ({
  topK,
  setTopK,
  onFrameClick,
  onQueryChange,
  queryInputRef,
  onFocusQueryInput,
  onBlurQueryInput,
}) => {
  const [eventDescription, setEventDescription] = useState("");
  const [resultType, setResultType] = useState(null);
  const [frames, setFrames] = useState([]);
  const [kisEvents, setKisEvents] = useState([]);
  const [paths, setPaths] = useState([]);
  const [trakeEvents, setTrakeEvents] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestError, setSuggestError] = useState(null);
  const queryTextareaRef = useRef(null);
  const { requestSubmission } = useSubmission();

  const setQueryTextareaRef = useCallback((node) => {
    queryTextareaRef.current = node;
    if (queryInputRef) queryInputRef.current = node;
  }, [queryInputRef]);

  useLayoutEffect(() => {
    const textarea = queryTextareaRef.current;
    if (!textarea) return;

    textarea.style.height = "0px";
    textarea.style.paddingTop = "8px";
    textarea.style.paddingBottom = "8px";
    textarea.style.lineHeight = "1.3";
    const contentHeight = textarea.scrollHeight;
    const isSingleLine = !/\r?\n/.test(textarea.value) && contentHeight <= 42;

    if (isSingleLine) {
      textarea.style.height = "42px";
      textarea.style.paddingTop = "0px";
      textarea.style.paddingBottom = "0px";
      textarea.style.lineHeight = "42px";
      return;
    }

    textarea.style.height = `${Math.max(contentHeight, 42)}px`;
  }, [eventDescription]);

  useEffect(() => {
    onQueryChange?.(eventDescription);
  }, [eventDescription, onQueryChange]);

  const handleSuggestQuery = useCallback(async () => {
    if (isSuggesting) return;
    setIsSuggesting(true);
    setSuggestError(null);
    try {
      const list = await suggestQueries({ count: 5, query: eventDescription });
      setSuggestions(list || []);
    } catch (err) {
      setSuggestError(err.message || 'Failed to fetch query suggestions');
    } finally {
      setIsSuggesting(false);
    }
  }, [eventDescription, isSuggesting]);

  const handleSelectSuggestion = useCallback((suggestionText) => {
    const trimmed = (suggestionText || "").trim();
    setEventDescription(trimmed);
    if (queryInputRef?.current) {
      queryInputRef.current.focus();
    }
  }, [queryInputRef]);

  const handleTrakeSubmit = (path) => {
    const vid = displayVideoId(path.video_id);
    const framesStr = path.frame_idxs.join(',');
    requestSubmission({
      line: `${vid},${framesStr}`,
      source: "TRAKE path",
    });
  };

  const handleFrameSubmit = (frame) => {
    const vid = displayVideoId(frame.video_id);
    requestSubmission({
      line: `${vid},${frame.frame_idx}`,
      source: "KIS/TRAKE frame",
    });
  };
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const rawEventText = eventDescription.trim();
    if (!rawEventText || isSearching) return;

    const events = parseTrakeEvents(rawEventText);
    const isTrakeMode = events !== null;
    let retrieval = null;

    if (isTrakeMode) {
      if (events.length < 2) {
        setResultType("trake");
        setError("TRAKE requires at least two ordered events labeled E1, E2, ... .");
        return;
      }
    } else {
      retrieval = parseRetrievalDescription(rawEventText);
    }

    const task = isTrakeMode ? "trake" : "kis";
    const searchKey = task === "trake" ? null : progressiveSearchIdKey(task);
    const searchId = searchKey ? window.sessionStorage.getItem(searchKey) : null;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setKisEvents([]);
    setPaths([]);
    setTrakeEvents([]);
    setSearchLatencyMs(null);

    try {
      let response;
      if (isTrakeMode) {
        response = await searchTrake({ events, topK, signal: controller.signal });
      } else {
        response = await searchFrames({
          query: retrieval.query,
          queryType: retrieval.queryType,
          topK,
          searchId,
          signal: controller.signal,
        });
      }

      if (searchKey && response.search_id) {
        window.sessionStorage.setItem(searchKey, response.search_id);
      }
      if (isTrakeMode) {
        setResultType("trake");
        setPaths(response.paths || []);
        setTrakeEvents(response.events || events);
      } else {
        setResultType("retrieval");
        setFrames(response.results || []);
        setKisEvents(response.events || []);
        setSearchLatencyMs(response.latency_ms);
      }
      setWarnings(response.warnings || []);
    } catch (requestError) {
      if (requestError.name === "AbortError") return;
      if (requestError?.status === 410 && searchKey) {
        window.sessionStorage.removeItem(searchKey);
      }
      const resetInstruction = requestError?.status === 409
        ? " Reset this task with New Search before continuing."
        : "";
      setResultType(isTrakeMode ? "trake" : "retrieval");
      setError(`${requestError.message || "Failed to contact search API"}${resetInstruction}`);
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [eventDescription, isSearching, topK]);

  const handleNewSearch = useCallback(() => {
    PROGRESSIVE_TASKS.forEach((task) => {
      window.sessionStorage.removeItem(progressiveSearchIdKey(task));
    });
    setEventDescription("");
    setFrames([]);
    setKisEvents([]);
    setPaths([]);
    setTrakeEvents([]);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
  }, []);



  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
      }
      if (event.key.toLowerCase() === 'n') {
        event.preventDefault();
        handleNewSearch();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNewSearch]);

  return (
    <div className="adhoc-workspace search-workspace">
      <form className="search-query-form" onSubmit={submit}>
        <div className="search-query-row">
          <div className="query-input-wrapper">
            <textarea
              ref={setQueryTextareaRef}
              id="event-query"
              className="input-text query-input-field"
              rows={1}
              value={eventDescription}
              onChange={(event) => setEventDescription(event.target.value)}
              placeholder="Describe the event, or add E1, E2, ... for TRAKE"
              onFocus={onFocusQueryInput}
              onBlur={onBlurQueryInput}
              disabled={isSearching}
              onKeyDown={(event) => {
                const isTrakeInput = parseTrakeEvents(eventDescription) !== null;
                if (event.key === "Enter" && !event.shiftKey && !isTrakeInput) {
                  event.preventDefault();
                  submit(event);
                }
              }}
            />
          </div>
          <div className="search-query-actions">
            <button
              type="submit"
              className="btn-primary query-submit-btn"
              disabled={isSearching || !eventDescription.trim()}
            >
              {isSearching ? "Searching..." : "Search"}
            </button>
            <button type="button" className="btn-secondary search-action-btn" onClick={handleNewSearch} title="Shortcut: N">
              New Search
            </button>
            <button
              type="button"
              className="btn-secondary query-suggest-btn"
              onClick={handleSuggestQuery}
              disabled={isSuggesting}
              title="Suggest 5 Queries"
            >
              {isSuggesting ? "Suggesting..." : "Suggest Query"}
            </button>
          </div>
        </div>
      </form>

      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox
            topK={topK}
            setTopK={setTopK}
            onReset={() => setTopK(20)}
          />
        </aside>
        <div className="adhoc-results">
          <GifLoaderOverlay isVisible={isSearching} />
          {!isSearching && resultType === "trake" && (
            <TrakeResults
              events={trakeEvents}
              paths={paths}
              warnings={warnings}
              error={error}
              hasSearched
              onFrameClick={onFrameClick}
              onTrakeSubmit={handleTrakeSubmit}
            />
          )}
          {!isSearching && resultType === "retrieval" && (
            <FramesBox
              results={frames}
              isLoading={false}
              error={error}
              latencyMs={searchLatencyMs}
              warnings={warnings}
              events={kisEvents}
              onFrameClick={onFrameClick}
              onSubmit={handleFrameSubmit}
            />
          )}
          {!isSearching && !resultType && (
            <FramesBox
              results={frames}
              isLoading={false}
              error={error}
              latencyMs={searchLatencyMs}
              warnings={warnings}
              events={kisEvents}
              onFrameClick={onFrameClick}
              onSubmit={handleFrameSubmit}
            />
          )}
      </div>
      <QuerySuggestionsPanel
          suggestions={suggestions}
          isLoading={isSuggesting}
          error={suggestError}
          onSelectSuggestion={handleSelectSuggestion}
        onRefresh={handleSuggestQuery}
      />
      </div>
    </div>
  );
};

export default SearchWorkspace;
