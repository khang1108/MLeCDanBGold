import {
  createConversation,
  listConversationIds,
  loadConversation,
  saveConversation,
} from './conversationStorage';

beforeEach(() => window.localStorage.clear());

test('persists and restores browser-owned KISC memory', () => {
  const session = createConversation();
  const updated = {
    ...session,
    turns: [{
      turn_id: 'turn_0001',
      sender: 'user',
      message: 'find a red car',
      created_at: session.created_at + 1,
      reply_to_turn_id: null,
    }],
    interpreted_state: {
      standalone_query: 'a red car',
      positive_constraints: ['red car'],
      negative_constraints: [],
      uncertain_constraints: [],
      accepted_frame_ids: [],
      rejected_frame_ids: [],
    },
  };

  saveConversation(updated);

  expect(loadConversation(session.session_id)).toEqual(updated);
  expect(listConversationIds()).toContain(session.session_id);
});
