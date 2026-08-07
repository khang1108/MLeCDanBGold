import React, { useCallback, useEffect, useRef, useState } from "react";
import { searchFrames, searchVqa } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import VqaResults from "./VqaResults";

const RETRIEVAL_PREFIX = /^\/(kis|trake)\b\s*/i;

export const parseRetrievalDescription = (description) => {
  const match = description.match(RETRIEVAL_PREFIX);
  if (!match) return null;
  const query = description.slice(match[0].length).trim();
  return query ? { queryType: match[1].toLowerCase(), query } : null;
};

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
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [vqaLatencyMs, setVqaLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const eventText = eventDescription.trim();
    const questionText = question.trim();
    if (!eventText || isSearching) return;

    const retrieval = questionText ? null : parseRetrievalDescription(eventText);
    if (!questionText && !retrieval) {
      setResultType(null);
      setError("Without a question, Event Description must start with /kis or /trake.");
      return;
    }

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsSearching(true);
    setError(null);
    setWarnings([]);
    setFrames([]);
    setSubmissions([]);
    setSearchLatencyMs(null);
    setVqaLatencyMs(null);
    try {
      const response = questionText
        ? await searchVqa({
          eventDescription: eventText,
          question: questionText,
          topK,
          signal: controller.signal,
        })
        : await searchFrames({
          query: retrieval.query,
          queryType: retrieval.queryType,
          topK,
          signal: controller.signal,
        });
      if (questionText) {
        setResultType("vqa");
        setSubmissions(response.submissions || []);
        setVqaLatencyMs(response.latency_ms);
      } else {
        setResultType("retrieval");
        setFrames(response.results || []);
        setSearchLatencyMs(response.latency_ms);
      }
      setWarnings(response.warnings || []);
    } catch (requestError) {
      if (requestError.name === "AbortError") return;
      setResultType(questionText ? "vqa" : "retrieval");
      setError(requestError.message || "Failed to contact search API");
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setIsSearching(false);
      }
    }
  }, [eventDescription, isSearching, question, topK]);

  return (
    <div className="adhoc-workspace vqa-workspace">
      <form className="vqa-query-form" onSubmit={submit}>
        <label htmlFor="vqa-event">Event description</label>
        <textarea
          ref={queryInputRef}
          id="vqa-event"
          className="input-text"
          value={eventDescription}
          onChange={(event) => setEventDescription(event.target.value)}
          placeholder="Use /kis or /trake for retrieval, or describe an event and add a question..."
          onFocus={onFocusQueryInput}
          onBlur={onBlurQueryInput}
          disabled={isSearching}
        />
        <label htmlFor="vqa-question">Question (optional)</label>
        <div className="adhoc-query-bar">
          <input
            id="vqa-question"
            className="input-text query-input-field"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Leave empty for /kis or /trake; fill to run QA"
            disabled={isSearching}
          />
          <button
            type="submit"
            className="btn-primary query-submit-btn"
            disabled={isSearching || !eventDescription.trim()}
          >
            {isSearching
              ? "Searching..."
              : question.trim() ? "Search QA" : "Search KIS / TRAKE"}
          </button>
        </div>
        <p className="vqa-query-hint">
          Question takes priority. Without a question, begin the description
          with <code>/kis</code> or <code>/trake</code>.
        </p>
      </form>
      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox topK={topK} setTopK={setTopK} onReset={() => setTopK(20)} />
        </aside>
        <div className="adhoc-results">
          <GifLoaderOverlay isVisible={isSearching} />
          {!isSearching && (
            resultType === "retrieval" ? (
              <FramesBox
                results={frames}
                isLoading={false}
                error={error}
                latencyMs={searchLatencyMs}
                warnings={warnings}
                onFrameClick={onFrameClick}
              />
            ) : (
              <VqaResults
                submissions={submissions}
                warnings={warnings}
                latencyMs={vqaLatencyMs}
                error={error}
                hasSearched={vqaLatencyMs !== null || error !== null}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default VqaSearchWorkspace;
