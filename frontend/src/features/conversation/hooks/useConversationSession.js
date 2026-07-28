import { useCallback, useState } from 'react';
import {
  createConversation,
  loadConversation,
  saveConversation,
} from '../utils/conversationStorage';

// Owns the browser-side KISC memory sent in full on every stateless agent turn.
export const useConversationSession = (runRequest) => {
  const [session, setSession] = useState(null);
  const [sessionError, setSessionError] = useState(null);

  const create = useCallback(async () => runRequest(async () => {
    setSessionError(null);
    try {
      const nextSession = createConversation();
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
      const nextSession = loadConversation(sessionId);
      if (!nextSession) throw new Error('Conversation was not found in this browser.');
      setSession(nextSession);
      return nextSession;
    } catch (error) {
      setSessionError(error.message || 'Could not load this conversation.');
      return null;
    }
  }), [runRequest]);

  const update = useCallback((next) => {
    setSession((current) => {
      const value = typeof next === 'function' ? next(current) : next;
      if (value) saveConversation(value);
      return value;
    });
  }, []);

  return { session, sessionError, create, load, setSession: update };
};
