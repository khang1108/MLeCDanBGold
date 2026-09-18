import { act, renderHook } from '@testing-library/react';
import { getCurrentDresTask, submitDresAnswer } from '../../../api/submissions';
import { useQaSubmission } from './useQaSubmission';

jest.mock('../../../api/submissions', () => ({
  getCurrentDresTask: jest.fn(),
  submitDresAnswer: jest.fn(),
}));

describe('useQaSubmission', () => {
  const selectedTask = {
    evaluationId: 'eval-qa',
    taskName: 'QA_01',
    taskGroup: 'QA',
    taskType: 'VQA',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('validates userId and text before calling API', async () => {
    const { result } = renderHook(() => useQaSubmission({
      userId: '',
      selectedTask,
    }));

    await act(async () => {
      await result.current.submit('apple');
    });

    expect(result.current.error).toMatch(/Connect a VBS participant/i);
    expect(submitDresAnswer).not.toHaveBeenCalled();

    // Now with userId but empty text
    const { result: res2 } = renderHook(() => useQaSubmission({
      userId: 'team-1',
      selectedTask,
    }));

    await act(async () => {
      await res2.current.submit('   ');
    });

    expect(res2.current.error).toMatch(/non-empty/i);
    expect(submitDresAnswer).not.toHaveBeenCalled();
  });

  test('submits text answer and records CORRECT verdict in lastOutcome and history', async () => {
    getCurrentDresTask.mockResolvedValue({
      task_scope_key: 'scope_qa_123',
    });
    submitDresAnswer.mockResolvedValue({
      state: 'RECORDED',
      recorded: true,
      verdict: 'CORRECT',
      message: 'Submission correct, well done!',
    });

    const { result } = renderHook(() => useQaSubmission({
      userId: 'team-1',
      selectedTask,
    }));

    act(() => {
      result.current.setDraft('blue car');
    });

    await act(async () => {
      await result.current.submit();
    });

    expect(getCurrentDresTask).toHaveBeenCalledWith('team-1', {
      evaluationId: 'eval-qa',
      taskName: 'QA_01',
    });

    expect(submitDresAnswer).toHaveBeenCalledWith({
      userId: 'team-1',
      expectedTaskScopeKey: 'scope_qa_123',
      evaluationId: 'eval-qa',
      taskName: 'QA_01',
      answer: { kind: 'TEXT', text: 'blue car' },
    });

    expect(result.current.lastOutcome).toEqual(expect.objectContaining({
      state: 'RECORDED',
      verdict: 'CORRECT',
      text: 'blue car',
      message: 'Submission correct, well done!',
    }));

    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].verdict).toBe('CORRECT');
    expect(result.current.draft).toBe(''); // Draft cleared on correct
  });

  test('records WRONG verdict in history and keeps draft for editing', async () => {
    getCurrentDresTask.mockResolvedValue({
      task_scope_key: 'scope_qa_123',
    });
    submitDresAnswer.mockResolvedValue({
      state: 'RECORDED',
      recorded: true,
      verdict: 'WRONG',
      message: 'Submission wrong, try again!',
    });

    const { result } = renderHook(() => useQaSubmission({
      userId: 'team-1',
      selectedTask,
    }));

    act(() => {
      result.current.setDraft('red car');
    });

    await act(async () => {
      await result.current.submit();
    });

    expect(result.current.lastOutcome.verdict).toBe('WRONG');
    expect(result.current.history).toHaveLength(1);
    expect(result.current.history[0].verdict).toBe('WRONG');
    expect(result.current.draft).toBe('red car'); // Kept for editing
  });

  test('handles DRES_AUTH_REJECTED by calling onSessionRejected', async () => {
    getCurrentDresTask.mockResolvedValue({
      task_scope_key: 'scope_qa_123',
    });
    submitDresAnswer.mockResolvedValue({
      state: 'NOT_RECORDED',
      recorded: false,
      reason: 'DRES_AUTH_REJECTED',
      message: 'Auth rejected',
    });
    const onSessionRejected = jest.fn();

    const { result } = renderHook(() => useQaSubmission({
      userId: 'team-1',
      selectedTask,
      onSessionRejected,
    }));

    await act(async () => {
      await result.current.submit('test');
    });

    expect(onSessionRejected).toHaveBeenCalledWith('team-1');
  });
});
