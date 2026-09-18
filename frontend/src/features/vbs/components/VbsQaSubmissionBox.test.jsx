import React from 'react';
import { act, fireEvent, render, screen } from '@testing-library/react';
import VbsQaSubmissionBox from './VbsQaSubmissionBox';
import * as SubmissionsApi from '../../../api/submissions';

jest.mock('../../../api/submissions', () => ({
  getCurrentDresTask: jest.fn(),
  submitDresAnswer: jest.fn(),
}));

describe('VbsQaSubmissionBox', () => {
  const selectedTask = {
    evaluationId: 'eval-1',
    taskName: 'QA_01',
    taskGroup: 'QA',
    taskType: 'VQA',
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders input and title', () => {
    render(
      <VbsQaSubmissionBox
        selectedTask={selectedTask}
        connectedUserId="team-a"
      />
    );

    expect(screen.getByText('QA Text Answer')).toBeInTheDocument();
    expect(screen.getByText('QA_01')).toBeInTheDocument();
    expect(screen.getByLabelText('QA text answer')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /submit/i })).toBeDisabled();
  });

  test('submits text and displays CORRECT verdict banner and history entry', async () => {
    SubmissionsApi.getCurrentDresTask.mockResolvedValue({
      task_scope_key: 'scope-qa-1',
    });
    SubmissionsApi.submitDresAnswer.mockResolvedValue({
      state: 'RECORDED',
      recorded: true,
      verdict: 'CORRECT',
      message: 'Well done!',
    });

    render(
      <VbsQaSubmissionBox
        selectedTask={selectedTask}
        connectedUserId="team-a"
      />
    );

    const input = screen.getByLabelText('QA text answer');
    fireEvent.change(input, { target: { value: 'blue' } });

    const submitBtn = screen.getByRole('button', { name: /submit/i });
    expect(submitBtn).not.toBeDisabled();

    await act(async () => {
      fireEvent.click(submitBtn);
    });

    expect(SubmissionsApi.submitDresAnswer).toHaveBeenCalledWith({
      userId: 'team-a',
      expectedTaskScopeKey: 'scope-qa-1',
      evaluationId: 'eval-1',
      taskName: 'QA_01',
      answer: { kind: 'TEXT', text: 'blue' },
    });

    // Check verdict banner
    expect(screen.getByRole('status')).toHaveTextContent(/correct/i);
    expect(screen.getByText('Well done!')).toBeInTheDocument();

    // Check history item
    expect(screen.getByText('"blue"')).toBeInTheDocument();
    expect(screen.getByText('Attempts (1)')).toBeInTheDocument();
  });

  test('supports pressing Enter to submit', async () => {
    SubmissionsApi.getCurrentDresTask.mockResolvedValue({
      task_scope_key: 'scope-qa-1',
    });
    SubmissionsApi.submitDresAnswer.mockResolvedValue({
      state: 'RECORDED',
      recorded: true,
      verdict: 'WRONG',
      message: 'Try again',
    });

    render(
      <VbsQaSubmissionBox
        selectedTask={selectedTask}
        connectedUserId="team-a"
      />
    );

    const input = screen.getByLabelText('QA text answer');
    fireEvent.change(input, { target: { value: 'green' } });

    await act(async () => {
      fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
    });

    expect(SubmissionsApi.submitDresAnswer).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('status')).toHaveTextContent(/wrong/i);
  });
});
