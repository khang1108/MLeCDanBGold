import { useCallback, useState } from 'react';
import { createSession, getSession } from '../../../api/sessions';

// Owns only the active server session and its create/load errors.
export const useConversationSession = (runRequest) => {
  const [session, setSession] = useState(null);
  const [sessionError, setSessionError] = useState(null);

  const create = useCallback(async () => runRequest(async () => {
    setSessionError(null);
    try {
      const nextSession = await createSession();
      setSession(nextSession);
      return nextSession;
    } catch (error) {
      setSessionError(error.message || 'Could not create a conversation.');
      return null;
    }
  }), [runRequest]);

  const load = useCallback(async (sessionId) => runRequest(async () => {
    setSessionError(null);
    try {
      const nextSession = await getSession(sessionId);
      setSession(nextSession);
      return nextSession;
    } catch (error) {
      setSessionError(error.message || 'Could not load this conversation.');
      return null;
    }
  }), [runRequest]);

  return { session, sessionError, create, load, setSession };
};