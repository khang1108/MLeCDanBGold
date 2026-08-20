import React, { useCallback, useEffect, useRef, useState } from "react";
import { frameAssetUrl, searchFrames, searchTrake, searchVqa } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import FrameCard from "../../frames/components/FrameCard";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import { displayVideoId } from "../../frames/videoSource";
import FileSelectionModal from "../../submission/components/FileSelectionModal";
import { useSubmission } from "../../submission/contexts/SubmissionContext";

const RETRIEVAL_PREFIX = /^\/(kis)\b\s*/i;
const ANY_PREFIX = /^\/(vqa|kis|trake)\b\s*/i;
const TRAKE_PREFIX = /^\/trake\b\s*/i;
const SESSION_FINGERPRINT_KEY = "hcmai.session.fingerprint";
const SEARCH_ID_PREFIX = "hcmai.progressive.search_id";
const PROGRESSIVE_TASKS = ["kis", "vqa"];

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
  return description
    .replace(TRAKE_PREFIX, "")
    .split(/\s*\|\s*/)
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

export const materializeTrakeFrames = (submission, events) => (
  submission.frame_ids
    .map((frameId, eventIndex) => ({
      frame_id: frameId,
      video_id: submission.video_id,
      frame_idx: submission.frame_idxs[eventIndex],
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
      left.frame_idx - right.frame_idx
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
  const { appendLine } = useSubmission();

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
        setError("Without a question, Event description must start with /kis or /trake.");
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
              placeholder="Event query (/kis, /trake)..."
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
          <button type="button" className="btn-secondary" onClick={() => setIsFileModalOpen(true)} title="Shortcut: F">
            Choose CSV
          </button>
          <button type="button" className="btn-secondary" onClick={handleNewQuestion} title="Shortcut: N">
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
      </div>
      <FileSelectionModal isOpen={isFileModalOpen} onClose={() => setIsFileModalOpen(false)} />
    </div>
  );
};

export default VqaSearchWorkspace;
