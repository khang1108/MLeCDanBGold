import { submitDresAnswer, submitDresAnswers } from './submissions';

const recordedResponse = () => ({
  ok: true,
  status: 200,
  headers: { get: () => null },
  json: async () => ({
    state: 'RECORDED',
    recorded: true,
    verdict: 'CORRECT',
    message: 'DRES recorded the answer batch',
  }),
});

describe('submission transport', () => {
  beforeEach(() => {
    global.fetch = jest.fn().mockResolvedValue(recordedResponse());
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  test('posts AVS answers as one answers array', async () => {
    await submitDresAnswers({
      userId: 'team-a',
      expectedTaskScopeKey: 'scope-1',
      evaluationId: 'eval-1',
      taskName: 'AVS task',
      answers: [
        { kind: 'TEMPORAL', video_id: 'V1', start_ms: 1000, end_ms: 1000 },
        { kind: 'TEMPORAL', video_id: 'V2', start_ms: 2000, end_ms: 2000 },
      ],
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.expected_task_scope_key).toBe('scope-1');
    expect(body.answers).toHaveLength(2);
    expect(body).not.toHaveProperty('answer');
  });

  test('singular wrapper still sends a one-item answers array', async () => {
    await submitDresAnswer({
      userId: 'team-a',
      expectedTaskScopeKey: 'scope-1',
      answer: { kind: 'TEXT', text: 'answer' },
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);
    const body = JSON.parse(global.fetch.mock.calls[0][1].body);
    expect(body.answers).toEqual([{ kind: 'TEXT', text: 'answer' }]);
    expect(body).not.toHaveProperty('answer');
  });
});
