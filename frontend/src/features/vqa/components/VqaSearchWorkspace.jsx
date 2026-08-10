import React, { useCallback, useEffect, useRef, useState } from "react";
import { searchFrames, searchVqa, suggestQueries } from "../../../api/search";
import FramesBox from "../../frames/components/FramesBox";
import ToolBox from "../../search-controls/components/ToolBox";
import QuerySuggestionsBox from "../../search-controls/components/QuerySuggestionsBox";
import GifLoaderOverlay from "../../search/components/GifLoaderOverlay";
import VqaResults from "./VqaResults";
import MiniChallengePanel from "../../minichallenge/components/MiniChallengePanel";
import { useMiniChallenge } from "../../minichallenge/hooks/useMiniChallenge";

const RETRIEVAL_PREFIX = /^\/(kis|tkis|vkis|trake)\b\s*/i;
const ANY_PREFIX = /^\/(vqa|kis|tkis|vkis|trake)\b\s*/i;

export const parseRetrievalDescription = (description) => {
  const match = description.match(RETRIEVAL_PREFIX);
  if (!match) return null;
  const rawType = match[1].toLowerCase();
  const queryType = rawType === "tkis" ? "kis" : rawType;
  const query = description.slice(match[0].length).trim();
  return query ? { queryType, query } : null;
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
  
  const [suggestCount, setSuggestCount] = useState(5);
  const [suggestions, setSuggestions] = useState([]);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [suggestError, setSuggestError] = useState(null);
  const suggestControllerRef = useRef(null);

  const [resultType, setResultType] = useState(null);
  const [frames, setFrames] = useState([]);
  const [submissions, setSubmissions] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [searchLatencyMs, setSearchLatencyMs] = useState(null);
  const [vqaLatencyMs, setVqaLatencyMs] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const requestRef = useRef(null);
  const challenge = useMiniChallenge();

  useEffect(() => () => {
    requestRef.current?.abort();
    suggestControllerRef.current?.abort();
  }, []);

  const handleSuggest = useCallback(async () => {
    const rawText = eventDescription.trim();
    if (!rawText) return;
    
    // Extract query text by stripping prefix if present
    const cleanQuery = rawText.replace(ANY_PREFIX, "").trim();
    if (!cleanQuery) return;

    suggestControllerRef.current?.abort();
    const controller = new AbortController();
    suggestControllerRef.current = controller;
    setIsSuggesting(true);
    setSuggestError(null);
    setSuggestions([]);

    try {
      const response = await suggestQueries({
        query: cleanQuery,
        count: suggestCount,
        signal: controller.signal,
      });
      setSuggestions(response.suggestions || []);
    } catch (err) {
      if (err.name === "AbortError") return;
      setSuggestError(err.message || "Failed to load suggestions.");
    } finally {
      if (suggestControllerRef.current === controller) {
        suggestControllerRef.current = null;
        setIsSuggesting(false);
      }
    }
  }, [eventDescription, suggestCount]);

  const handleSelectSuggestion = useCallback((suggestionQuery) => {
    const rawText = eventDescription.trimStart();
    const match = rawText.match(ANY_PREFIX);
    if (match) {
      setEventDescription(`${match[0]}${suggestionQuery}`);
    } else {
      setEventDescription(suggestionQuery);
    }
  }, [eventDescription]);

  const handleChallengeSubmit = useCallback((frame, overrideAnswer = null) => {
    const taskName = challenge.submissionTaskName;
    if (!taskName) return;
    const confirmed = window.confirm(
      `Submit ${frame.video_id} (frame ${frame.frame_idx}) to “${taskName}”?`,
    );
    if (confirmed) challenge.submitFrame(frame, overrideAnswer);
  }, [challenge]);

  const submit = useCallback(async (event) => {
    event.preventDefault();
    const rawEventText = eventDescription.trim();
    const questionText = question.trim();
    if (!rawEventText || isSearching) return;

    const isVqaMode = Boolean(questionText);
    let eventTextForSubmit = rawEventText;
    let retrieval = null;

    if (isVqaMode) {
      // Auto VQA: strip prefix if user left it there
      eventTextForSubmit = rawEventText.replace(ANY_PREFIX, "").trim();
    } else {
      retrieval = parseRetrievalDescription(rawEventText);
      if (!retrieval) {
        setResultType(null);
        setError("Without a question, Event description must start with /tkis, /vkis, or /trake.");
        return;
      }
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
      const response = isVqaMode
        ? await searchVqa({
          eventDescription: eventTextForSubmit,
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

      if (isVqaMode) {
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
      setResultType(isVqaMode ? "vqa" : "retrieval");
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
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(e);
                }
              }}
            />
          </div>
          <button
            type="button"
            className="query-suggest-btn"
            onClick={handleSuggest}
            disabled={isSuggesting || !eventDescription.trim()}
          >
            {isSuggesting ? "..." : "Suggest"}
          </button>
        </div>

        <QuerySuggestionsBox
          suggestions={suggestions}
          isLoading={isSuggesting}
          error={suggestError}
          onSelectSuggestion={handleSelectSuggestion}
        />

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
        </div>
      </form>

      <div className="adhoc-workspace-body">
        <aside className="adhoc-sidebar">
          <h3 className="adhoc-sidebar-title">Options</h3>
          <ToolBox 
            topK={topK} 
            setTopK={setTopK} 
            suggestCount={suggestCount}
            setSuggestCount={setSuggestCount}
            onReset={() => {
              setTopK(20);
              setSuggestCount(5);
            }} 
          />
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
                onChallengeSubmit={challenge.currentTask ? handleChallengeSubmit : null}
                submittingFrameId={challenge.submittingFrameId}
              />
            ) : (
              <FramesBox
                results={submissions.map(sub => ({
                  ...sub,
                  scores: { final: sub.joint_score },
                  caption: sub.evidence_summary || `Answer: ${sub.answer}`
                }))}
                isLoading={false}
                error={error}
                hasSearched={vqaLatencyMs !== null || error !== null}
                onChallengeSubmit={challenge.currentTask ? handleChallengeSubmit : null}
                submittingFrameId={challenge.submittingFrameId}
                latencyMs={vqaLatencyMs}
                warnings={warnings}
                onFrameClick={onFrameClick}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
};

export default VqaSearchWorkspace;
