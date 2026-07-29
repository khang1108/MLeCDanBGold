import { useCallback, useState } from 'react';
import { searchKisc } from '../../../api/search';

const nextTurn = (sender, message, turns, createdAt = Date.now()) => ({
  turn_id: `turn_${String(turns.length + 1).padStart(4, '0')}`,
  sender,
  message,
  created_at: createdAt,
  reply_to_turn_id: sender === 'ai' ? turns.at(-1)?.turn_id || null : null,
});

// Sends browser-owned conversation memory through the stateless KISC agent.
export const useSearchWorkspace = ({ session, topK, draftFeedback, feedbackDirty, runRequest, setSession }) => {
  const [results, setResults] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [latencyMs, setLatencyMs] = useState(null);
  const [lastRequestId, setLastRequestId] = useState(null);
  const [error, setError] = useState(null);
  const [isSearching, setIsSearching] = useState(false);

  const clear = useCallback(() => {
    setResults([]); setWarnings([]); setLatencyMs(null); setLastRequestId(null); setError(null);
  }, []);

  const submit = useCallback(async (value, overrideSession) => {
    const activeSession = overrideSession || session;
    const query = typeof value === 'string' ? value.trim() : '';
    if (!activeSession || (!query && !feedbackDirty)) return false;
    setError(null);
    setIsSearching(true);

    const completed = await runRequest(async () => {
      try {
        const currentMessage = query || session.interpreted_state?.standalone_query;
        if (!currentMessage) return false;
        const response = await searchKisc({
          history: session.turns,
          currentMessage,
          previousState: session.interpreted_state,
          feedback: draftFeedback,
          topK,
        });
        const searchResponse = response.search;
        setResults(searchResponse.results);
        setWarnings(response.warnings || searchResponse.warnings || []);
        setLatencyMs(searchResponse.latency_ms);
        setLastRequestId(searchResponse.request_id || null);

        const turns = [...session.turns];
        if (query) turns.push(nextTurn('user', query, turns));
        const message = `Retrieved ${searchResponse.results.length} frame candidates for “${response.interpreted_state.standalone_query}”.`;
        const latestTime = turns.at(-1)?.created_at || 0;
        turns.push(nextTurn('ai', message, turns, Math.max(Date.now(), latestTime + 1)));
        setSession({
          ...session,
          turns,
          feedback: {
            accepted_frame_ids: response.interpreted_state.accepted_frame_ids,
            rejected_frame_ids: response.interpreted_state.rejected_frame_ids,
          },
          interpreted_state: response.interpreted_state,
        });
      } catch (requestError) {
        setError(requestError.message || 'Could not contact the search API.');
        return false;
      } finally {
        setIsSearching(false);
      }
    });
    return completed || false;
  }, [draftFeedback, feedbackDirty, runRequest, session, setSession, topK]);

  return { results, warnings, latencyMs, lastRequestId, error, isSearching, clear, submit };
};
