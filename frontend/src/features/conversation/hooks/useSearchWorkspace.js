import { useCallback, useState } from 'react';
import { searchFrames } from '../../../api/search';
import { getSession, updateFeedback } from '../../../api/sessions';

// Manages transient result data; the active session remains server-owned elsewhere.
export const useSearchWorkspace = ({ session, topK, searchMode, draftFeedback, feedbackDirty, runRequest, setSession }) => {
  const [results, setResults] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [latencyMs, setLatencyMs] = useState(null);
  const [lastRequestId, setLastRequestId] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  const clear = useCallback(() => {
    setResults([]); setWarnings([]); setLatencyMs(null); setLastRequestId(null); setError(null);
  }, []);

  const submit = useCallback(async (value) => {
    const query = value.trim();
    if (!session || (!query && !feedbackDirty)) return false;
    setError(null);
    setIsSearching(Boolean(query));

    const completed = await runRequest(async () => {
      try {
        if (query) {
          const response = await searchFrames({
            query, topK, searchMode,
            sessionId: session.session_id, feedback: feedbackDirty ? draftFeedback : undefined,
          });
          setResults(response.results);
          setWarnings(response.warnings || []);
          setLatencyMs(response.latency_ms);
          setLastRequestId(response.request_id || null);
          setSession(await getSession(session.session_id));
          return true;
        }
        setSession(await updateFeedback(session.session_id, draftFeedback));
        return true;
      } catch (requestError) {
        setError(requestError.message || 'Could not contact the search API.');
        return false;
      } finally {
        setIsSearching(false);
      }
    });
    return completed || false;
  }, [draftFeedback, feedbackDirty, runRequest, searchMode, session, setSession, topK]);

  return { results, warnings, latencyMs, lastRequestId, error, isSearching, clear, submit };
};