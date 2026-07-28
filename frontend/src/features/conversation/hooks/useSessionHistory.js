import { useCallback, useState } from 'react';
import { listSessions } from '../../../api/sessions';

// Fetches the ID-only history endpoint when the popover is opened.
export const useSessionHistory = () => {
  const [sessionIds, setSessionIds] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadHistory = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const ids = await listSessions();
      setSessionIds(Array.isArray(ids) ? ids : []);
    } catch (requestError) {
      setError(requestError.message || 'Could not load conversation history.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  return { sessionIds, isLoading, error, loadHistory, setError };
};