import React, { useCallback, useEffect, useRef, useState } from "react";
import { frameAssetUrl, searchFrames, searchTrake, suggestQueries } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import FrameCard from "../../frames/components/FrameCard";
import ToolBox from "../../search-controls/components/ToolBox";
import QuerySuggestionsPanel from "../../search-controls/components/QuerySuggestionsPanel";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import { displayVideoId } from "../../frames/videoSource";
import FileSelectionModal from "../../submission/components/FileSelectionModal";
import { useSubmission } from "../../submission/contexts/SubmissionContext";

const RETRIEVAL_PREFIX = /^\/(kis)\b\s*/i;
const TRAKE_PREFIX = /^\/trake\b\s*/i;
const TRAKE_EVENT_LABEL = /^\s*E(\d+)\s*:\s*(.*)$/i;
const SESSION_FINGERPRINT_KEY = "hcmai.session.fingerprint";
const SEARCH_ID_PREFIX = "hcmai.progressive.search_id";
const PROGRESSIVE_TASKS = ["kis"];

export const parseRetrievalDescription = (description) => {
  const match = description.match(RETRIEVAL_PREFIX);
  if (!match) return null;
  const rawType = match[1].toLowerCase();
  const queryType = rawType;
  const query = description.slice(match[0].length).trim();
  return query ? { queryType, query } : null;
};

export const parseTrakeEvents = (description) => {
  if (!TRAKE_PREFIX.test(description)) return null;
  const body = description.replace(TRAKE_PREFIX, "").trim();
  if (!body) return [];

  const events = [];
  let currentEvent = null;
  let expectedEventNumber = 1;

  for (const line of body.split(/\r?\n/)) {
    const label = line.match(TRAKE_EVENT_LABEL);
    if (label) {
      const eventNumber = Number(label[1]);
      if (eventNumber !== expectedEventNumber) return [];
      expectedEventNumber += 1;
      currentEvent = [];
      events.push(currentEvent);
      if (label[2].trim()) currentEvent.push(label[2].trim());
      continue;
    }

    if (!currentEvent) {
      if (line.trim()) return [];
      continue;
    }
    if (line.trim()) currentEvent.push(line.trim());
  }

  const normalizedEvents = events.map((parts) => parts.join(" ").trim());
  return normalizedEvents.some((event) => !event) ? [] : normalizedEvents;
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

export const materializeTrakeFrames = (submission, events) => (
  submission.frame_ids
    .map((frameId, eventIndex) => ({
      frame_id: frameId,
      video_id: submission.video_id,
      frame_idx: submission.frame_idxs[eventIndex],
      timestamp_ms: submission.timestamps_ms?.[eventIndex],
      fps: submission.fps,
      caption: events[eventIndex] || `TRAKE event ${eventIndex + 1}`,
      thumbnail_url: frameAssetUrl(frameId, 'thumbnail'),
      frame_url: frameAssetUrl(frameId, 'image'),
      submission_rank: submission.rank,
      event_index: eventIndex,
    }))
);

export const groupTrakeFramesByVideo = (submissions, events) => {
  const groups = new Map();
  submissions.forEach((submission) => {
    const frames = groups.get(submission.video_id) || [];
    frames.push(...materializeTrakeFrames(submission, events));
    groups.set(submission.video_id, frames);
  });
  return Array.from(groups, ([videoId, frames]) => ({
    video_id: videoId,
    best_rank: Math.min(...frames.map((frame) => frame.submission_rank)),
    frames: frames.sort((left, right) => (
      (left.timestamp_ms ?? Number.MAX_SAFE_INTEGER)
        - (right.timestamp_ms ?? Number.MAX_SAFE_INTEGER)
      || left.frame_idx - right.frame_idx
      || left.submission_rank - right.submission_rank
      || left.event_index - right.event_index
    )),
  })).sort((left, right) => left.best_rank - right.best_rank || left.video_id.localeCompare(right.video_id));
};

const TrakeResults = ({ events, submissions, warnings, error, hasSearched, onFrameClick, onTrakeSubmit, onFrameSubmit }) => (
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
    {!error && submissions.length > 0 && (
      <div className="frames-scroll-region">
        {groupTrakeFramesByVideo(submissions, events).map((group) => (
          <article className="trake-video-group" key={group.video_id} aria-label={`TRAKE frames for ${displayVideoId(group.video_id)}`}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {displayVideoId(group.video_id)}
              <button 
                className="btn-secondary" 
                style={{ padding: '2px 6px', fontSize: '12px' }}
                onClick={() => onTrakeSubmit(group)}
                title="Submit this TRAKE path"
              >
                ⬆
              </button>
            </h3>
            <div className="frames-grid">
              {group.frames.map((frame) => (
                <FrameCard
                  key={`${frame.frame_id}-${frame.submission_rank}-${frame.event_index}`}
                  frame={frame}
                  onClick={() => onFrameClick?.(frame)}
                  onSubmit={onFrameSubmit}
                />
              ))}
            </div>
          </article>
        ))}
      </div>
    )}
    {!error && hasSearched && submissions.length === 0 && (
      <div className="frames-empty-state">
        <p className="body-md frames-empty-text">No ordered TRAKE paths found</p>
      </div>
    )}
  </section>
);

const VqaSearchWorkspace = ({
  topK,
  setTopK,
  onFrameClick,
  queryInputRef,
  onFocusQueryInput,
  onBlurQueryInput,
}) => {
  const [eventDescription, setEventDescription] = useState("");
  const [question, setQuestion] = useState("");
  const [resultType, setResultType] = useState(null);
  const [frames, setFrames] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [trakeEvents, setTrakeEvents] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [vqaLatencyMs, setVqaLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isFileModalOpen, setIsFileModalOpen] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestError, setSuggestError] = useState(null);
  const { appendLine } = useSubmission();

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
    const formatted = trimmed.startsWith("/") ? trimmed : `/kis ${trimmed}`;
    setEventDescription(formatted);
    if (queryInputRef?.current) {
      queryInputRef.current.focus();
    }
  }, [queryInputRef]);

  const handleFrameSubmit = (frame) => {
    const vid = displayVideoId(frame.video_id);
    if (resultType === 'vqa') {
      const answer = frame.answer ? frame.answer.replace(/"/g, '""') : '';
      appendLine(`${vid},${frame.frame_idx},"${answer}"`);
    } else {
      appendLine(`${vid},${frame.frame_idx}`);
    }
  };

  const handleTrakeSubmit = (group) => {
    const vid = displayVideoId(group.video_id);
    const framesStr = group.frames.map(f => f.frame_idx).join(',');
    appendLine(`${vid},${framesStr}`);
  };
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const rawEventText = eventDescription.trim();
    const questionText = question.trim();
    if (!rawEventText || isSearching) return;

    if (questionText) {
      setResultType("vqa");
      setError("VQA search is no longer available. Remove the question and use /kis or /trake.");
      return;
    }

    const events = parseTrakeEvents(rawEventText);
    const isTrakeMode = events !== null;
    let retrieval = null;

    if (isTrakeMode) {
      if (events.length < 2) {
        setResultType("trake");
        setError("TRAKE requires at least two ordered events labeled E1:, E2:, ... on separate lines.");
        return;
      }
    } else {
      retrieval = parseRetrievalDescription(rawEventText);
      if (!retrieval) {
        setResultType(null);
        setError("Without a question, Event description must start with /kis or /trake.");
        return;
      }
    }

    const task = isTrakeMode ? "trake" : retrieval.queryType;
    const searchKey = task === "trake" ? null : progressiveSearchIdKey(task);
    const searchId = searchKey ? window.sessionStorage.getItem(searchKey) : null;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setSubmissions([]);
    setTrakeEvents([]);
    setSearchLatencyMs(null);
    setVqaLatencyMs(null);

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
        setSubmissions(response.submissions || []);
        setTrakeEvents(response.events || events);
      } else {
        setResultType("retrieval");
        setFrames(response.results || []);
        setSearchLatencyMs(response.latency_ms);
      }
      setWarnings(response.warnings || []);
    } catch (requestError) {
      if (requestError.name === "AbortError") return;
      if (requestError?.status === 410 && searchKey) {
        window.sessionStorage.removeItem(searchKey);
      }
      const resetInstruction = requestError?.status === 409
        ? " Reset this task with New Question before continuing."
        : "";
      setResultType(isTrakeMode ? "trake" : "retrieval");
      setError(`${requestError.message || "Failed to contact search API"}${resetInstruction}`);
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [eventDescription, isSearching, question, topK]);

  const handleNewQuestion = useCallback(() => {
    PROGRESSIVE_TASKS.forEach((task) => {
      window.sessionStorage.removeItem(progressiveSearchIdKey(task));
    });
    setEventDescription("");
    setQuestion("");
    setFrames([]);
    setSubmissions([]);
    setTrakeEvents([]);
    setWarnings([]);
    setResultType(null);
    setError(null);
    setSearchLatencyMs(null);
    setVqaLatencyMs(null);
  }, []);



  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
      }
      if (event.key.toLowerCase() === 'n') {
        event.preventDefault();
        handleNewQuestion();
      } else if (event.key.toLowerCase() === 'f') {
        event.preventDefault();
        setIsFileModalOpen(true);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleNewQuestion]);

  const vqaFrames = submissions.map((submission) => ({
    ...submission,
    scores: { final: submission.joint_score },
    caption: submission.caption
      || submission.evidence_summary
      || `Answer: ${submission.answer}`,
  }));

  return (
    <div className="adhoc-workspace vqa-workspace">
      <form className="vqa-query-form" onSubmit={submit}>
        <div className="adhoc-query-bar">
          <div className="query-input-wrapper">
            <textarea
              ref={queryInputRef}
              id="vqa-event"
              className="input-text query-input-field"
              rows={1}
              value={eventDescription}
              onChange={(event) => setEventDescription(event.target.value)}
              placeholder="Event query (/kis, /trake E1: ... E2: ... on new lines)..."
              onFocus={onFocusQueryInput}
              onBlur={onBlurQueryInput}
              disabled={isSearching}
              onKeyDown={(event) => {
                const isTrakeInput = !question.trim() && TRAKE_PREFIX.test(eventDescription);
                if (event.key === "Enter" && !event.shiftKey && !isTrakeInput) {
                  event.preventDefault();
                  submit(event);
                }
              }}
            />
          </div>
        </div>

        <div className="adhoc-query-bar">
          <input
            id="vqa-question"
            className="input-text query-input-field"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Question (optional for VQA)..."
            disabled={isSearching}
          />
          <button
            type="submit"
            className="btn-primary query-submit-btn"
            disabled={isSearching || !eventDescription.trim()}
          >
            {isSearching ? "Searching..." : "Search"}
          </button>
          <button type="button" className="btn-secondary" onClick={() => setIsFileModalOpen(true)} title="Shortcut: F">
            Choose CSV
          </button>
          <button type="button" className="btn-secondary" onClick={handleNewQuestion} title="Shortcut: N">
            New Question
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
      </form>

      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox topK={topK} setTopK={setTopK} onReset={() => setTopK(20)} />
        </aside>
        <div className="adhoc-results">
          <GifLoaderOverlay isVisible={isSearching} />
          {!isSearching && resultType === "trake" && (
            <TrakeResults
              events={trakeEvents}
              submissions={submissions}
              warnings={warnings}
              error={error}
              hasSearched
              onFrameClick={onFrameClick}
              onTrakeSubmit={handleTrakeSubmit}
              onFrameSubmit={handleFrameSubmit}
            />
          )}
          {!isSearching && resultType === "retrieval" && (
            <FramesBox
              results={frames}
              isLoading={false}
              error={error}
              latencyMs={searchLatencyMs}
              warnings={warnings}
              onFrameClick={onFrameClick}
              onSubmit={handleFrameSubmit}
            />
          )}
          {!isSearching && resultType !== "trake" && resultType !== "retrieval" && (
            <FramesBox
              results={vqaFrames}
              isLoading={false}
              error={error}
              latencyMs={vqaLatencyMs}
              warnings={warnings}
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
      <FileSelectionModal isOpen={isFileModalOpen} onClose={() => setIsFileModalOpen(false)} />
    </div>
  );
};

export default VqaSearchWorkspace;
