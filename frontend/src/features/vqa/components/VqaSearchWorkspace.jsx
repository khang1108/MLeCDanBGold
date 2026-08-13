import React, { useCallback, useEffect, useRef, useState } from "react";
import { searchFrames, searchTrake, searchVqa } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";

const RETRIEVAL_PREFIX = /^\/(kis|tkis|vkis)\b\s*/i;
const ANY_PREFIX = /^\/(vqa|kis|tkis|vkis|trake)\b\s*/i;
const TRAKE_PREFIX = /^\/trake\b\s*/i;
const SESSION_FINGERPRINT_KEY = "hcmai.session.fingerprint";
const SEARCH_ID_PREFIX = "hcmai.progressive.search_id";
const PROGRESSIVE_TASKS = ["kis", "vkis", "vqa"];

export const parseRetrievalDescription = (description) => {
  const match = description.match(RETRIEVAL_PREFIX);
  if (!match) return null;
  const rawType = match[1].toLowerCase();
  const queryType = rawType === "tkis" ? "kis" : rawType;
  const query = description.slice(match[0].length).trim();
  return query ? { queryType, query } : null;
};

export const parseTrakeEvents = (description) => {
  if (!TRAKE_PREFIX.test(description)) return null;
  return description
    .replace(TRAKE_PREFIX, "")
    .split(/\s*(?:->|\r?\n)\s*/)
    .map((event) => event.trim())
    .filter(Boolean);
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

const TrakeResults = ({ events, submissions, warnings, error, hasSearched }) => (
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
        <h3>Ordered TRAKE paths</h3>
        {submissions.map((submission) => (
          <article key={`${submission.rank}-${submission.video_id}`}>
            <h4>Rank {submission.rank}: {submission.video_id}</h4>
            <ol>
              {events.map((event, index) => (
                <li key={`${submission.frame_ids[index]}-${index}`}>
                  {event}: frame {submission.frame_idxs[index]} ({submission.frame_ids[index]})
                </li>
              ))}
            </ol>
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
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const rawEventText = eventDescription.trim();
    const questionText = question.trim();
    if (!rawEventText || isSearching) return;

    const isVqaMode = Boolean(questionText);
    const events = isVqaMode ? null : parseTrakeEvents(rawEventText);
    const isTrakeMode = events !== null;
    let eventTextForSubmit = rawEventText;
    let retrieval = null;

    if (isVqaMode) {
      eventTextForSubmit = rawEventText.replace(ANY_PREFIX, "").trim();
    } else if (isTrakeMode) {
      if (events.length < 2) {
        setResultType("trake");
        setError("TRAKE requires at least two non-empty ordered events separated by -> or new lines.");
        return;
      }
    } else {
      retrieval = parseRetrievalDescription(rawEventText);
      if (!retrieval) {
        setResultType(null);
        setError("Without a question, Event description must start with /tkis, /vkis, or /trake.");
        return;
      }
    }

    const task = isVqaMode ? "vqa" : (isTrakeMode ? "trake" : retrieval.queryType);
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
      if (isVqaMode) {
        response = await searchVqa({
          eventDescription: eventTextForSubmit,
          question: questionText,
          topK,
          searchId,
          signal: controller.signal,
        });
      } else if (isTrakeMode) {
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
      if (isVqaMode) {
        setResultType("vqa");
        setSubmissions(response.submissions || []);
        setVqaLatencyMs(response.latency_ms);
      } else if (isTrakeMode) {
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
      setResultType(isVqaMode ? "vqa" : (isTrakeMode ? "trake" : "retrieval"));
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
              placeholder="Event query (/tkis, /vkis, /trake)..."
              onFocus={onFocusQueryInput}
              onBlur={onBlurQueryInput}
              disabled={isSearching}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
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
          <button type="button" className="btn-secondary" onClick={handleNewQuestion}>
            New Question
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
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default VqaSearchWorkspace;
