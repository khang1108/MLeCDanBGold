import { act, renderHook } from '@testing-library/react';
import { submitDresAnswers } from '../../../api/submissions';
import { useAvsSubmission } from './useAvsSubmission';

jest.mock('../../../api/submissions', () => ({
  submitDresAnswers: jest.fn(),
}));

const batch = {
  candidateIds: ['f1', 'f2'],
  answers: [
    { kind: 'TEMPORAL', video_id: 'V1', start_ms: 1000, end_ms: 1000 },
    { kind: 'TEMPORAL', video_id: 'V2', start_ms: 2000, end_ms: 2000 },
  ],
};

const selectedTask = { evaluationId: 'eval-1', taskName: 'AVS task' };

const renderSubmission = (onSessionRejected = jest.fn()) => renderHook(() => useAvsSubmission({
  userId: 'team-a',
  selectedTask,
  taskScopeKey: 'scope-1',
  onSessionRejected,
}));

describe('useAvsSubmission', () => {
  beforeEach(() => {
    submitDresAnswers.mockReset();
  });

  test('RECORDED sends the entire batch exactly once', async () => {
    submitDresAnswers.mockResolvedValue({
      state: 'RECORDED',
      recorded: true,
      verdict: 'CORRECT',
      message: 'ok',
    });
    const { result } = renderSubmission();

    await act(async () => {
      await result.current.submitBatch(batch);
    });

    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
    expect(submitDresAnswers).toHaveBeenCalledWith(expect.objectContaining({
      userId: 'team-a',
      expectedTaskScopeKey: 'scope-1',
      evaluationId: 'eval-1',
      taskName: 'AVS task',
      answers: batch.answers,
    }));
    expect(result.current.status).toBe('RECORDED');
  });

  test('NOT_RECORDED is definitive and does not retry', async () => {
    submitDresAnswers.mockResolvedValue({
      state: 'NOT_RECORDED',
      recorded: false,
      reason: 'DRES_REJECTED',
      message: 'rejected',
    });
    const { result } = renderSubmission();

    await act(async () => {
      await result.current.submitBatch(batch);
    });

    expect(result.current.status).toBe('NOT_RECORDED');
    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  });

  test('UNKNOWN does not trigger an automatic second call', async () => {
    submitDresAnswers.mockResolvedValue({
      state: 'UNKNOWN',
      recorded: null,
      message: 'check DRES',
    });
    const { result } = renderSubmission();

    await act(async () => {
      await result.current.submitBatch(batch);
    });

    expect(result.current.status).toBe('UNKNOWN');
    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  });

  test('TASK_SCOPE_MISMATCH remains a definite local error', async () => {
    const error = Object.assign(new Error('scope changed'), {
      status: 409,
      code: 'TASK_SCOPE_MISMATCH',
    });
    submitDresAnswers.mockRejectedValue(error);
    const { result } = renderSubmission();

    await act(async () => {
      await result.current.submitBatch(batch);
    });

    expect(result.current.status).toBe('IDLE');
    expect(result.current.error).toMatchObject({ code: 'TASK_SCOPE_MISMATCH' });
    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  });

  test('DRES_AUTH_REJECTED notifies the session owner once', async () => {
    const onSessionRejected = jest.fn();
    submitDresAnswers.mockResolvedValue({
      state: 'NOT_RECORDED',
      recorded: false,
      reason: 'DRES_AUTH_REJECTED',
      message: 'reconnect',
    });
    const { result } = renderSubmission(onSessionRejected);

    await act(async () => {
      await result.current.submitBatch(batch);
    });

    expect(result.current.status).toBe('NOT_RECORDED');
    expect(onSessionRejected).toHaveBeenCalledTimes(1);
  });
});
