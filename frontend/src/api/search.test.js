import { answerFrameQuestion, searchFrames, searchKisc } from './search';

const response = (payload, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  json: jest.fn().mockResolvedValue(payload),
});

afterEach(() => jest.restoreAllMocks());

test('posts the canonical stateful search request', async () => {
  const payload = { results: [], latency_ms: { total: 1 } };
  jest.spyOn(global, 'fetch').mockResolvedValue(response(payload));

  await expect(searchFrames({
    query: ' red boat ',
    topK: 20,
    searchMode: 'accurate',
    sessionId: 'kisc_sess_1234',
    feedback: {
      accepted_frame_ids: ['frame_A'],
      rejected_frame_ids: [],
    },
  })).resolves.toEqual(payload);

  expect(global.fetch).toHaveBeenCalledWith(
    'http://127.0.0.1:8000/api/v1/search',
    expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({
        query: 'red boat',
        top_k: 20,
        search_mode: 'accurate',
        session_id: 'kisc_sess_1234',
        feedback: {
          accepted_frame_ids: ['frame_A'],
          rejected_frame_ids: [],
        },
      }),
    }),
  );
});

test('resolves API-relative frame asset URLs', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(response({
    results: [{
      frame_id: 'f1',
      thumbnail_url: '/api/v1/frames/f1/thumbnail',
      frame_url: '/api/v1/frames/f1/image',
    }],
    latency_ms: { total: 1 },
  }));

  const payload = await searchFrames({
    query: 'red car',
    topK: 1,
    searchMode: 'fast',
  });

  expect(payload.results[0].thumbnail_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/f1/thumbnail',
  );
  expect(payload.results[0].frame_url).toBe(
    'http://127.0.0.1:8000/api/v1/frames/f1/image',
  );
});

test('surfaces the backend error message', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ detail: 'Search engine is not initialized' }, 503),
  );

  await expect(searchFrames({
    query: 'red car',
    topK: 20,
    searchMode: 'fast',
  })).rejects.toThrow('Search engine is not initialized');
});

test('rejects a malformed successful search response', async () => {
  jest.spyOn(global, 'fetch').mockResolvedValue(
    response({ latency_ms: { total: 1 } }),
  );

  await expect(searchFrames({
    query: 'boat',
    topK: 20,
    searchMode: 'fast',
  })).rejects.toThrow('invalid response contract');
});

test('posts canonical KISC and VQA requests', async () => {
  const responses = [
    {
      interpreted_state: { standalone_query: 'red car' },
      search: { results: [], latency_ms: { total: 1 } },
    },
    { frame_id: 'f1', answer: 'A red car.' },
  ];
  jest.spyOn(global, 'fetch').mockImplementation(async () => (
    response(responses.shift())
  ));

  await searchKisc({
    history: [{
      turn_id: 'turn_0001',
      sender: 'user',
      message: 'find a vehicle',
      created_at: 1,
      reply_to_turn_id: null,
    }],
    currentMessage: ' red car ',
    previousState: {
      standalone_query: 'find a vehicle',
      positive_constraints: ['vehicle'],
      negative_constraints: [],
      uncertain_constraints: [],
      accepted_frame_ids: [],
      rejected_frame_ids: [],
    },
    feedback: {
      accepted_frame_ids: ['f1'],
      rejected_frame_ids: [],
    },
  });
  await answerFrameQuestion({
    frameId: 'f1',
    question: ' What is visible? ',
  });

  expect(global.fetch.mock.calls[0][0]).toContain('/api/v1/kisc/search');
  const kiscBody = JSON.parse(global.fetch.mock.calls[0][1].body);
  expect(kiscBody.current_message).toBe('red car');
  expect(kiscBody.history).toHaveLength(1);
  expect(kiscBody.previous_state.standalone_query).toBe('find a vehicle');
  expect(kiscBody.feedback.accepted_frame_ids).toEqual(['f1']);
  expect(global.fetch.mock.calls[1][0]).toContain('/api/v1/vqa');
  expect(JSON.parse(global.fetch.mock.calls[1][1].body)).toEqual({
    frame_id: 'f1',
    question: 'What is visible?',
  });
});
