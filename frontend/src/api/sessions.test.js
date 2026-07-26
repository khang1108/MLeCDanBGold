import { createSession, getSession, listSessions, updateFeedback } from './sessions';

const response = (payload, status = 200) => ({ ok: status >= 200 && status < 300, status, json: jest.fn().mockResolvedValue(payload) });
afterEach(() => jest.restoreAllMocks());

test('uses the published session, history, and feedback routes', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({ session_id: 'kisc_sess_1234' }));
  await createSession('problem A'); await listSessions(); await getSession('kisc_sess_1234'); await updateFeedback('kisc_sess_1234', { accepted_frame_ids: ['frame_A'], rejected_frame_ids: [] });
  expect(global.fetch).toHaveBeenNthCalledWith(1, 'http://127.0.0.1:8000/api/v1/session?problem_id=problem%20A', expect.objectContaining({ method: 'POST' }));
  expect(global.fetch).toHaveBeenNthCalledWith(2, 'http://127.0.0.1:8000/api/v1/sessions', expect.objectContaining({ method: 'GET' }));
  expect(global.fetch).toHaveBeenNthCalledWith(3, 'http://127.0.0.1:8000/api/v1/session/kisc_sess_1234', expect.objectContaining({ method: 'GET' }));
  expect(global.fetch).toHaveBeenNthCalledWith(4, 'http://127.0.0.1:8000/api/v1/feedback?session_id=kisc_sess_1234', expect.objectContaining({ method: 'POST', body: JSON.stringify({ accepted_frame_ids: ['frame_A'], rejected_frame_ids: [] }) }));
});

test('normalizes FastAPI, malformed JSON, and network errors', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValueOnce(response({ detail: [{ msg: 'feedback requires session_id' }] }, 422)).mockResolvedValueOnce({ ok: true, status: 200, json: jest.fn().mockRejectedValue(new SyntaxError('bad')) }).mockRejectedValueOnce(new TypeError('Failed to fetch'));
  await expect(updateFeedback('s', { accepted_frame_ids: [], rejected_frame_ids: [] })).rejects.toMatchObject({ message: 'feedback requires session_id', status: 422 });
  await expect(listSessions()).rejects.toMatchObject({ message: 'Backend returned invalid JSON', status: 200 });
  await expect(listSessions()).rejects.toThrow('Could not reach the backend: Failed to fetch');
});