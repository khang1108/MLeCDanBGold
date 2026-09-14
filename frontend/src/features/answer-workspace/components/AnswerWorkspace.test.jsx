import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import AnswerWorkspace from './AnswerWorkspace';
import { useAnswerWorkspace } from '../contexts/AnswerWorkspaceContext';
import {
  resolveSubmissionAttempt,
  submitAnswerCandidate,
  submitAvsAnswers,
} from '../../../api/answerWorkspace';
import { resolveFrameAtTimestamp } from '../../../api/frames';

jest.mock('../contexts/AnswerWorkspaceContext', () => ({
  useAnswerWorkspace: jest.fn(),
}));
jest.mock('../../../api/answerWorkspace', () => ({
  resolveSubmissionAttempt: jest.fn(),
  submitAnswerCandidate: jest.fn(),
  submitAvsAnswers: jest.fn(),
}));
jest.mock('../../../api/frames', () => ({ resolveFrameAtTimestamp: jest.fn() }));
jest.mock('../../frames/components/ImageModal', () => (props) => (
  <div data-testid="answer-frame-viewer" data-initial-timestamp={props.initialTimestampMs} />
));

const frameCandidate = (overrides = {}) => ({
  candidate_id: 'frame-1',
  kind: 'FRAME',
  source_frame_id: 'internal-frame',
  video_id: 'V01',
  timestamp_ms: 12_000,
  text: null,
  contributed_by_user_id: 'team-a',
  created_at_ms: 1,
  revision: 3,
  submitted_at_ms: null,
  submitted_by_user_id: null,
  dres_status: null,
  evaluation_id: 'eval-1',
  task_scope_key: 'dres-task-v1:key-1',
  ...overrides,
});

const textCandidate = (overrides = {}) => ({
  candidate_id: 'text-1',
  kind: 'TEXT',
  source_frame_id: null,
  video_id: null,
  timestamp_ms: null,
  text: 'A person opens the door',
  contributed_by_user_id: 'team-a',
  created_at_ms: 2,
  revision: 1,
  submitted_at_ms: null,
  submitted_by_user_id: null,
  dres_status: null,
  evaluation_id: 'eval-1',
  task_scope_key: 'dres-task-v1:key-1',
  ...overrides,
});

const baseWorkspace = (overrides = {}) => ({
  avs_enabled: false,
  evaluation_id: 'eval-1',
  task_scope_key: 'dres-task-v1:key-1',
  task_name: 'KIS task',
  revision: 9,
  updated_by_user_id: 'team-a',
  updated_at_ms: 9,
  candidates: [frameCandidate(), textCandidate()],
  pending_submission: null,
  active_evaluation_id: 'eval-1',
  active_task_scope_key: 'dres-task-v1:key-1',
  active_task_name: 'KIS task',
  task_scope_mismatch: false,
  ...overrides,
});

const makeContext = (overrides = {}) => ({
  connectedUserId: 'team-a',
  workspace: baseWorkspace(),
  candidates: baseWorkspace().candidates,
  isConnected: true,
  connectionError: null,
  pendingAction: null,
  addFrame: jest.fn(),
  addText: jest.fn(),
  updateFrame: jest.fn(),
  updateText: jest.fn(),
  remove: jest.fn(),
  clear: jest.fn(),
  clearAndSwitchTask: jest.fn(),
  setAvsEnabled: jest.fn(),
  refreshWorkspace: jest.fn().mockResolvedValue(baseWorkspace()),
  ...overrides,
});

beforeEach(() => {
  jest.clearAllMocks();
  useAnswerWorkspace.mockReturnValue(makeContext());
  submitAnswerCandidate.mockResolvedValue({ state: 'ACCEPTED', accepted: true, attempt_id: 'attempt-1' });
  submitAvsAnswers.mockResolvedValue({ state: 'ACCEPTED', accepted: true, attempt_id: 'attempt-2' });
  resolveSubmissionAttempt.mockResolvedValue({ state: 'ACCEPTED', accepted: true });
  resolveFrameAtTimestamp.mockResolvedValue({
    video_id: 'V01', frame_id: 'nearby-frame', frame_idx: 99, timestamp_ms: 12_040,
  });
});

test('shows exact typed answer rows and per-row actions while AVS is off', () => {
  render(<AnswerWorkspace />);

  expect(screen.getByRole('region', { name: 'Answer workspace' })).toBeTruthy();
  expect(screen.getByRole('switch', { name: 'AVS mode' }).checked).toBe(false);
  expect(screen.getByText('2 answers')).toBeTruthy();
  expect(screen.getByText('V01,12000,12000')).toBeTruthy();
  expect(screen.getByText('A person opens the door')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Submit KIS answer' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Submit VQA answer' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Open answer in viewer' })).toBeTruthy();
  expect(screen.getAllByRole('button', { name: 'Delete answer' })).toHaveLength(2);
  expect(screen.queryByText(/frame-1|team-a|revision|created/i)).toBeNull();
});

test('freezes single-candidate and workspace revisions before KIS confirmation', async () => {
  const workspace = baseWorkspace();
  render(<AnswerWorkspace />);

  fireEvent.click(screen.getByRole('button', { name: 'Submit KIS answer' }));
  expect(screen.getByRole('dialog', { name: 'Confirm KIS submission' })).toBeTruthy();
  expect(screen.getByText('V01,12000,12000')).toBeTruthy();
  expect(screen.queryByLabelText('timestamp_ms')).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and submit' }));

  await waitFor(() => expect(submitAnswerCandidate).toHaveBeenCalledWith({
    kind: 'FRAME',
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:key-1',
    expectedWorkspaceRevision: workspace.revision,
    candidateId: 'frame-1',
    expectedCandidateRevision: 3,
  }));
});

test('keeps the reviewed KIS snapshot when the live workspace changes before confirmation', async () => {
  const { rerender } = render(<AnswerWorkspace />);
  fireEvent.click(screen.getByRole('button', { name: 'Submit KIS answer' }));

  const updatedWorkspace = baseWorkspace({
    revision: 10,
    candidates: [frameCandidate({ timestamp_ms: 15_000, revision: 4 }), textCandidate()],
  });
  useAnswerWorkspace.mockReturnValue(makeContext({
    workspace: updatedWorkspace,
    candidates: updatedWorkspace.candidates,
  }));
  rerender(<AnswerWorkspace />);
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and submit' }));

  await waitFor(() => expect(submitAnswerCandidate).toHaveBeenCalledWith({
    kind: 'FRAME',
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:key-1',
    expectedWorkspaceRevision: 9,
    candidateId: 'frame-1',
    expectedCandidateRevision: 3,
  }));
});

test('confirms and submits one VQA TEXT answer with its frozen candidate revision', async () => {
  render(<AnswerWorkspace />);
  fireEvent.click(screen.getByRole('button', { name: 'Submit VQA answer' }));
  expect(screen.getByRole('dialog', { name: 'Confirm VQA submission' })).toBeTruthy();
  expect(screen.getByText('A person opens the door')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and submit' }));

  await waitFor(() => expect(submitAnswerCandidate).toHaveBeenCalledWith({
    kind: 'TEXT',
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:key-1',
    expectedWorkspaceRevision: 9,
    candidateId: 'text-1',
    expectedCandidateRevision: 1,
  }));
});

test('requires confirmation before clearing the shared candidate list', async () => {
  const context = makeContext();
  useAnswerWorkspace.mockReturnValue(context);
  render(<AnswerWorkspace />);

  fireEvent.click(screen.getByRole('button', { name: 'Clear workspace' }));
  expect(screen.getByRole('dialog', { name: 'Confirm clear answers' })).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(context.clear).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Clear workspace' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and clear answers' }));
  await waitFor(() => expect(context.clear).toHaveBeenCalledTimes(1));
});

test('adds a typed VQA answer only after Enter validates the editable candidate', async () => {
  const context = makeContext();
  useAnswerWorkspace.mockReturnValue(context);
  render(<AnswerWorkspace />);

  fireEvent.change(screen.getByRole('textbox', { name: 'VQA answer' }), {
    target: { value: 'A person enters through the side door' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Add VQA answer' }));
  const dialog = screen.getByRole('dialog', { name: 'Add TEXT answer' });
  fireEvent.keyDown(screen.getByLabelText('Text answer'), { key: 'Enter', code: 'Enter' });

  await waitFor(() => expect(context.addText).toHaveBeenCalledWith({
    text: 'A person enters through the side door',
  }));
  expect(dialog).toBeTruthy();
});

test('AVS hides retained text, keeps frame collection, and submits all unsent FRAME rows once', async () => {
  const frameTwo = frameCandidate({ candidate_id: 'frame-2', video_id: 'V02', timestamp_ms: 30_000, revision: 5 });
  const snapshot = baseWorkspace({ avs_enabled: true, candidates: [frameCandidate(), textCandidate(), frameTwo] });
  useAnswerWorkspace.mockReturnValue(makeContext({ workspace: snapshot, candidates: snapshot.candidates }));

  render(<AnswerWorkspace />);

  expect(screen.queryByText('A person opens the door')).toBeNull();
  expect(screen.queryByRole('textbox', { name: 'VQA answer' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Add frame to answer workspace' })).toBeTruthy();
  expect(screen.queryByRole('button', { name: 'Submit KIS answer' })).toBeNull();
  expect(screen.queryByRole('button', { name: 'Submit VQA answer' })).toBeNull();
  expect(screen.getByRole('button', { name: 'Submit all 2 AVS answers' })).toBeTruthy();

  fireEvent.click(screen.getByRole('button', { name: 'Submit all 2 AVS answers' }));
  expect(screen.getByRole('dialog', { name: 'Confirm AVS submission' })).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and submit' }));

  await waitFor(() => expect(submitAvsAnswers).toHaveBeenCalledWith({
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:key-1',
    expectedWorkspaceRevision: 9,
    candidates: [
      { candidateId: 'frame-1', expectedRevision: 3 },
      { candidateId: 'frame-2', expectedRevision: 5 },
    ],
  }));
  expect(submitAvsAnswers).toHaveBeenCalledTimes(1);
});

test('AVS sends the complete eligible frame order once and excludes a KIS-submitted row', async () => {
  const snapshot = baseWorkspace({
    avs_enabled: true,
    candidates: [
      frameCandidate({ candidate_id: 'kis-submitted', submitted_at_ms: 5, submitted_by_user_id: 'team-a' }),
      frameCandidate({ candidate_id: 'frame-c', video_id: 'V03', timestamp_ms: 30_000, revision: 8 }),
      frameCandidate({ candidate_id: 'frame-b', video_id: 'V02', timestamp_ms: 20_000, revision: 6 }),
      textCandidate(),
    ],
  });
  useAnswerWorkspace.mockReturnValue(makeContext({ workspace: snapshot, candidates: snapshot.candidates }));

  render(<AnswerWorkspace />);
  fireEvent.click(screen.getByRole('button', { name: 'Submit all 2 AVS answers' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and submit' }));

  await waitFor(() => expect(submitAvsAnswers).toHaveBeenCalledWith({
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:key-1',
    expectedWorkspaceRevision: 9,
    candidates: [
      { candidateId: 'frame-c', expectedRevision: 8 },
      { candidateId: 'frame-b', expectedRevision: 6 },
    ],
  }));
  expect(submitAvsAnswers).toHaveBeenCalledTimes(1);
});

test('AVS sends every eligible unsent FRAME in workspace order and skips a KIS-submitted row', async () => {
  const snapshot = baseWorkspace({
    avs_enabled: true,
    candidates: [
      frameCandidate({ candidate_id: 'kis-submitted', submitted_at_ms: 5, submitted_by_user_id: 'team-a' }),
      frameCandidate({ candidate_id: 'frame-c', video_id: 'V03', timestamp_ms: 30_000, revision: 8 }),
      frameCandidate({ candidate_id: 'frame-b', video_id: 'V02', timestamp_ms: 20_000, revision: 6 }),
      textCandidate(),
    ],
  });
  useAnswerWorkspace.mockReturnValue(makeContext({ workspace: snapshot, candidates: snapshot.candidates }));

  render(<AnswerWorkspace />);
  fireEvent.click(screen.getByRole('button', { name: 'Submit all 2 AVS answers' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and submit' }));

  await waitFor(() => expect(submitAvsAnswers).toHaveBeenCalledWith({
    userId: 'team-a',
    taskScopeKey: 'dres-task-v1:key-1',
    expectedWorkspaceRevision: 9,
    candidates: [
      { candidateId: 'frame-c', expectedRevision: 8 },
      { candidateId: 'frame-b', expectedRevision: 6 },
    ],
  }));
  expect(submitAvsAnswers).toHaveBeenCalledTimes(1);
});

test('locks workspace actions during UNKNOWN and requires an explicit outcome resolution', async () => {
  const context = makeContext({
    workspace: baseWorkspace({
      pending_submission: {
        attempt_id: 'attempt-unknown', kind: 'AVS', state: 'UNKNOWN',
        submitted_by_user_id: 'team-a', created_at_ms: 10,
      },
    }),
  });
  useAnswerWorkspace.mockReturnValue(context);

  render(<AnswerWorkspace />);

  expect(screen.getByRole('status').textContent).toMatch(/outcome is unknown/i);
  expect(screen.getByRole('button', { name: 'Resolve as accepted' }).disabled).toBe(false);
  expect(screen.getByRole('button', { name: 'Resolve as not accepted' }).disabled).toBe(false);
  expect(screen.getByRole('button', { name: 'Submit KIS answer' }).disabled).toBe(true);
  expect(screen.getByRole('switch', { name: 'AVS mode' }).disabled).toBe(true);

  fireEvent.click(screen.getByRole('button', { name: 'Resolve as accepted' }));
  await waitFor(() => expect(resolveSubmissionAttempt).toHaveBeenCalledWith({
    userId: 'team-a', attemptId: 'attempt-unknown', outcome: 'accepted',
  }));
  expect(context.refreshWorkspace).toHaveBeenCalled();
});

test('keeps old-task answers visible until clear-and-switch is explicitly confirmed', async () => {
  const context = makeContext({
    workspace: baseWorkspace({
      evaluation_id: 'eval-old',
      task_scope_key: 'dres-task-v1:old',
      task_name: 'Old KIS task',
      active_evaluation_id: 'eval-new',
      active_task_scope_key: 'dres-task-v1:new',
      active_task_name: 'New AVS task',
      task_scope_mismatch: true,
    }),
  });
  useAnswerWorkspace.mockReturnValue(context);

  render(<AnswerWorkspace />);

  expect(screen.getByRole('alert').textContent).toContain('Old KIS task');
  expect(screen.getByRole('alert').textContent).toContain('New AVS task');
  expect(screen.getByRole('alert').textContent).not.toContain('dres-task-v1:');
  expect(screen.getByText('V01,12000,12000')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: 'Clear and switch task' }));
  expect(screen.getByRole('dialog', { name: 'Confirm task switch' }).textContent)
    .toContain('Clear Old KIS task answers and switch to New AVS task?');
  expect(screen.getByRole('dialog', { name: 'Confirm task switch' }).textContent)
    .not.toContain('dres-task-v1:');
  expect(context.clearAndSwitchTask).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(context.clearAndSwitchTask).not.toHaveBeenCalled();

  fireEvent.click(screen.getByRole('button', { name: 'Clear and switch task' }));
  fireEvent.click(screen.getByRole('button', { name: 'Confirm and clear old-task answers' }));
  await waitFor(() => expect(context.clearAndSwitchTask).toHaveBeenCalledWith({
    expectedWorkspaceRevision: 9,
    oldEvaluationId: 'eval-old',
    oldTaskScopeKey: 'dres-task-v1:old',
    oldTaskName: 'Old KIS task',
    targetEvaluationId: 'eval-new',
    targetTaskScopeKey: 'dres-task-v1:new',
    targetTaskName: 'New AVS task',
  }));
});

test('edits from row body with revision checks and keeps delete action separate', async () => {
  const context = makeContext();
  useAnswerWorkspace.mockReturnValue(context);
  render(<AnswerWorkspace />);

  fireEvent.click(screen.getByRole('button', { name: 'Edit FRAME answer V01,12000,12000' }));
  expect(screen.getByRole('dialog', { name: 'Edit FRAME answer' })).toBeTruthy();
  fireEvent.change(screen.getByLabelText('timestamp_ms'), { target: { value: '15000' } });
  fireEvent.keyDown(screen.getByLabelText('timestamp_ms'), { key: 'Enter', code: 'Enter' });
  await waitFor(() => expect(context.updateFrame).toHaveBeenCalledWith({
    candidateId: 'frame-1', expectedCandidateRevision: 3, videoId: 'V01', timestampMs: 15_000,
  }));

  fireEvent.click(screen.getAllByRole('button', { name: 'Delete answer' })[0]);
  expect(screen.queryByRole('dialog', { name: 'Edit FRAME answer' })).toBeNull();
  expect(context.remove).toHaveBeenCalledWith({ candidateId: 'frame-1', expectedCandidateRevision: 3 });
});

test('opens the existing viewer at the exact candidate timestamp', async () => {
  render(<AnswerWorkspace />);

  fireEvent.click(screen.getByRole('button', { name: 'Open answer in viewer' }));
  await waitFor(() => expect(resolveFrameAtTimestamp).toHaveBeenCalledWith({
    videoId: 'V01', timestampMs: 12_000, signal: expect.any(AbortSignal),
  }));
  expect((await screen.findByTestId('answer-frame-viewer')).getAttribute('data-initial-timestamp')).toBe('12000');
});
