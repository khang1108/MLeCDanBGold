import React, { useCallback, useEffect, useRef, useState } from "react";
import { searchVqa } from "../../../api/search";
import ToolBox from "../../search-controls/components/ToolBox";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import VqaResults from "./VqaResults";

const VqaSearchWorkspace = ({ topK, setTopK }) => {
  const [eventDescription, setEventDescription] = useState("");
  const [question, setQuestion] = useState("");
  const [submissions, setSubmissions] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [latencyMs, setLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => () => requestRef.current?.abort(), []);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const eventText = eventDescription.trim();
    const questionText = question.trim();
    if (!eventText || !questionText || isSearching) return;

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setIsSearching(true);
    setError(null);
    try {
      const response = await searchVqa({
        eventDescription: eventText,
        question: questionText,
        topK,
        signal: controller.signal,
      });
      setSubmissions(response.submissions);
      setWarnings(response.warnings || []);
      setLatencyMs(response.latency_ms);
    } catch (requestError) {
      if (requestError.name === "AbortError") return;
      setError(requestError.message || "Failed to contact VQA API");
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
          id="vqa-event"
          className="input-text"
          value={eventDescription}
          onChange={(event) => setEventDescription(event.target.value)}
          placeholder="Describe the event to retrieve..."
          disabled={isSearching}
        />
        <label htmlFor="vqa-question">Question</label>
        <div className="adhoc-query-bar">
          <input
            id="vqa-question"
            className="input-text query-input-field"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What information should the evidence answer?"
            disabled={isSearching}
          />
          <button
            type="submit"
            className="btn-primary query-submit-btn"
            disabled={isSearching || !eventDescription.trim() || !question.trim()}
          >
            {isSearching ? "Searching..." : "Search VQA"}
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
          {!isSearching && (
            <VqaResults
              submissions={submissions}
              warnings={warnings}
              latencyMs={latencyMs}
              error={error}
              hasSearched={latencyMs !== null || error !== null}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default VqaSearchWorkspace;
