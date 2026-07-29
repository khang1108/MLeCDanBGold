import { useCallback, useState } from 'react';
import { listConversationIds } from '../utils/conversationStorage';

// Lists conversations persisted by this browser for the stateless KISC API.
export const useSessionHistory = () => {
  const [sessionIds, setSessionIds] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadHistory = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const ids = listConversationIds();
      setSessionIds(Array.isArray(ids) ? ids : []);
    } catch (requestError) {
      setError(requestError.message || 'Could not load conversation history.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const removeSessionId = useCallback((targetId) => {
    setSessionIds((prev) => prev.filter((id) => id !== targetId));
  }, []);

  return { sessionIds, isLoading, error, loadHistory, removeSessionId, setError };
};
