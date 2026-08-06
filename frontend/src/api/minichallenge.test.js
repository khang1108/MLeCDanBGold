import {
  getMiniChallengeCurrentTask,
  listMiniChallengeEvaluations,
  submitMiniChallengeFrame,
} from './minichallenge';
import * as client from './client';

jest.mock('./client');

describe('mini-challenge API', () => {
  afterEach(() => jest.clearAllMocks());

  test('uses the session header for evaluation and current-task requests', async () => {
    client.requestJson
      .mockResolvedValueOnce([{ id: 'evaluation 1', status: 'ACTIVE' }])
      .mockResolvedValueOnce({ name: 'QA', taskGroup: 'qa', taskType: 'QA' });

    await listMiniChallengeEvaluations(' private-token ');
    await getMiniChallengeCurrentTask('private-token', 'evaluation 1');

    expect(client.requestJson).toHaveBeenNthCalledWith(
      1,
      '/api/v1/minichallenge/evaluations',
      { signal: undefined, headers: { 'X-DRES-Session': 'private-token' } },
    );
    expect(client.requestJson).toHaveBeenNthCalledWith(
      2,
      '/api/v1/minichallenge/evaluations/evaluation%201/current-task',
      { signal: undefined, headers: { 'X-DRES-Session': 'private-token' } },
    );
  });

  test('submits canonical frame identity and task answer', async () => {
    client.requestJson.mockResolvedValueOnce({
      status: true,
      submission: 'CORRECT',
      description: 'Accepted',
    });

    const result = await submitMiniChallengeFrame({
      session: 'token',
      evaluationId: 'evaluation-1',
      frameId: 'frame-90',
      taskName: 'QA task',
      text: ' Bơ ',
    });

    expect(client.requestJson).toHaveBeenCalledWith(
      '/api/v1/minichallenge/evaluations/evaluation-1/submit',
      {
        method: 'POST',
        body: { frame_id: 'frame-90', task_name: 'QA task', text: 'Bơ' },
        signal: undefined,
        headers: { 'X-DRES-Session': 'token' },
      },
    );
    expect(result.submission).toBe('CORRECT');
  });

  test('rejects malformed backend contracts', async () => {
    client.requestJson.mockResolvedValueOnce({});
    await expect(listMiniChallengeEvaluations('token')).rejects.toThrow(
      'invalid evaluation list',
    );
  });
});
