const STORAGE_KEY = 'hcmai.kisc.conversations.v1';

const readAll = () => {
  try {
    const value = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}');
    return value && typeof value === 'object' ? value : {};
  } catch {
    return {};
  }
};

const writeAll = (sessions) => {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
};

export const saveConversation = (session) => {
  if (!session?.session_id) return;
  writeAll({ ...readAll(), [session.session_id]: session });
};

export const loadConversation = (sessionId) => readAll()[sessionId] || null;

export const listConversationIds = () => Object.values(readAll())
  .sort((left, right) => right.created_at - left.created_at)
  .map((session) => session.session_id);

export const createConversation = () => {
  const now = Date.now();
  const random = Math.random().toString(16).slice(2, 10);
  const session = {
    session_id: `kisc_local_${random}`,
    created_at: now,
    problem_id: null,
    turns: [],
    feedback: { accepted_frame_ids: [], rejected_frame_ids: [] },
    interpreted_state: null,
  };
  saveConversation(session);
  return session;
};
