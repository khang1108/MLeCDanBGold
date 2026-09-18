import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { searchAvs } from '../../../api/avs';
import { getCurrentDresTask, submitDresAnswers } from '../../../api/submissions';
import AvsWorkspace from './AvsWorkspace';

jest.mock('../../../api/avs', () => ({ searchAvs: jest.fn() }));
jest.mock('../../../api/submissions', () => ({
  getCurrentDresTask: jest.fn(),
  submitDresAnswers: jest.fn(),
}));

const task = {
  evaluationId: 'eval-1',
  evaluationName: 'VBS',
  taskName: 'AVS task',
  taskGroup: 'AVS',
  taskType: 'AVS',
  duration: 300,
};

const candidate = (id, videoId, timestampMs) => ({
  candidate_id: id,
  frame_id: id,
  video_id: videoId,
  frame_idx: timestampMs / 1000,
  timestamp_ms: timestampMs,
  fps: 25,
  retrieval_rank: 1,
  retrieval_score: 0.9,
  metadata: { title: null, caption: null, ocr: null, objects: [], asr: null },
});

const responseWith = (results) => ({
  results,
  latency: { retrieval_ms: 1, coverage_ms: 0, materialization_ms: 0, total_ms: 1 },
  candidate_pool_size: results.length,
  deduplicated_candidate_count: results.length,
  unique_videos: new Set(results.map((item) => item.video_id)).size,
  warnings: [],
});

const renderWorkspace = ({ onInspect = jest.fn() } = {}) => {
  const setSelectedTask = jest.fn();
  render(
    <AvsWorkspace
      connectedUserId="team-a"
      evaluations={[]}
      selectedTask={task}
      setSelectedTask={setSelectedTask}
      onInspect={onInspect}
      onSessionRejected={jest.fn()}
    />,
  );
  return { onInspect, setSelectedTask };
};

const runSearch = async (text) => {
  fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
    target: { value: text },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
  await waitFor(() => expect(searchAvs).toHaveBeenCalled());
  await waitFor(() => expect(screen.queryByRole('status')).toBeNull());
};

describe('AvsWorkspace', () => {
  beforeEach(() => {
    searchAvs.mockReset().mockResolvedValue(responseWith([
      candidate('f1', 'V1', 1000),
      candidate('f2', 'V2', 2000),
    ]));
    getCurrentDresTask.mockReset().mockResolvedValue({
      user_id: 'team-a',
      evaluation_id: 'eval-1',
      task_scope_key: 'scope-1',
      task_name: 'AVS task',
      task_group: 'AVS',
      task_type: 'AVS',
      duration: 300,
    });
  });

  test('checkbox selection changes only local basket state', async () => {
    renderWorkspace();
    fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
      target: { value: 'seafood' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
    await screen.findByRole('button', { name: 'Inspect V1 at 1000 ms' });

    const checkbox = screen.getAllByRole('checkbox')[0];
    fireEvent.click(checkbox);

    expect(checkbox.checked).toBe(true);
  });

  test('thumbnail inspection does not select the candidate', async () => {
    const onInspect = jest.fn();
    renderWorkspace({ onInspect });
    fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
      target: { value: 'seafood' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
    const inspect = await screen.findByRole('button', { name: 'Inspect V1 at 1000 ms' });

    fireEvent.click(inspect);

    expect(onInspect).toHaveBeenCalledWith(expect.objectContaining({ candidate_id: 'f1' }));
    expect(screen.getAllByRole('checkbox')[0].checked).toBe(false);
  });

  test('Space toggles the focused card while Enter inspects without selecting', async () => {
    const onInspect = jest.fn();
    renderWorkspace({ onInspect });
    fireEvent.change(screen.getByRole('textbox', { name: 'AVS query' }), {
      target: { value: 'seafood' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Search AVS' }));
    const card = await screen.findByTestId('avs-card-f1');

    card.focus();
    fireEvent.keyDown(card, { key: ' ' });
    expect(screen.getAllByRole('checkbox')[0].checked).toBe(true);

    fireEvent.keyDown(card, { key: 'Enter' });
    expect(onInspect).toHaveBeenCalledTimes(1);
    expect(screen.getAllByRole('checkbox')[0].checked).toBe(true);
  });

  test('pending selections survive a later AVS search and remain reviewable', async () => {
    searchAvs
      .mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]))
      .mockResolvedValueOnce(responseWith([candidate('f2', 'V2', 2000)]));
    renderWorkspace();

    await runSearch('first query');
    fireEvent.click(await screen.findByRole('checkbox'));
    await runSearch('second query');

    fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
    expect(screen.getByRole('dialog', { name: 'Selected AVS answers' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Remove f1 from selection' })).toBeTruthy();
  });

  test('review remove deletes only the chosen candidate', async () => {
    searchAvs.mockResolvedValueOnce(responseWith([
      candidate('f1', 'V1', 1000), candidate('f2', 'V2', 2000),
    ]));
    renderWorkspace();
    await runSearch('seafood');
    screen.getAllByRole('checkbox').forEach((checkbox) => fireEvent.click(checkbox));

    fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
    fireEvent.click(screen.getByRole('button', { name: 'Remove f1 from selection' }));

    expect(screen.queryByRole('button', { name: 'Remove f1 from selection' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Remove f2 from selection' })).toBeTruthy();
  });

  test('Clear asks for confirmation and preserves pending items when cancelled', async () => {
    window.confirm = jest.fn().mockReturnValue(false);
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
    renderWorkspace();
    await runSearch('seafood');
    fireEvent.click(screen.getByRole('checkbox'));

    fireEvent.click(screen.getByRole('button', { name: 'Clear selections' }));

    expect(window.confirm).toHaveBeenCalledWith('Clear 1 pending AVS selection?');
    expect(screen.getByRole('checkbox').checked).toBe(true);
  });

  test('Clear removes pending items only after explicit confirmation', async () => {
    window.confirm = jest.fn().mockReturnValue(true);
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
    renderWorkspace();
    await runSearch('seafood');
    fireEvent.click(screen.getByRole('checkbox'));

    fireEvent.click(screen.getByRole('button', { name: 'Clear selections' }));

    expect(screen.getByRole('checkbox').checked).toBe(false);
  });

  const avsEvaluations = [{
    id: 'eval-1',
    name: 'VBS',
    taskTemplates: [
      { name: 'AVS task', taskGroup: 'AVS', taskType: 'AVS', duration: 300 },
      { name: 'AVS task 2', taskGroup: 'AVS', taskType: 'AVS', duration: 300 },
    ],
  }];

  const TaskHarness = () => {
    const [selectedTask, setSelectedTask] = React.useState(task);
    return (
      <AvsWorkspace
        connectedUserId="team-a"
        evaluations={avsEvaluations}
        selectedTask={selectedTask}
        setSelectedTask={setSelectedTask}
        onInspect={jest.fn()}
        onSessionRejected={jest.fn()}
      />
    );
  };

  test('task switch cancellation keeps the current task and basket', async () => {
    window.confirm = jest.fn().mockReturnValue(false);
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
    render(<TaskHarness />);
    await runSearch('seafood');
    fireEvent.click(screen.getByRole('checkbox'));

    fireEvent.change(screen.getByLabelText('Select evaluation task'), {
      target: { value: 'eval-1:AVS task 2' },
    });

    expect(window.confirm).toHaveBeenCalled();
    expect(screen.getByLabelText('Select evaluation task').value).toBe('eval-1:AVS task');
    expect(screen.getByRole('checkbox').checked).toBe(true);
  });

  test('confirmed task switch waits for the new scope before clearing pending state', async () => {
    window.confirm = jest.fn().mockReturnValue(true);
    getCurrentDresTask.mockImplementation(async (_userId, { taskName }) => ({
      user_id: 'team-a',
      evaluation_id: 'eval-1',
      task_scope_key: taskName === 'AVS task 2' ? 'scope-2' : 'scope-1',
      task_name: taskName,
      task_group: 'AVS',
      task_type: 'AVS',
      duration: 300,
    }));
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
    render(<TaskHarness />);
    await runSearch('seafood');
    fireEvent.click(screen.getByRole('checkbox'));

    fireEvent.change(screen.getByLabelText('Select evaluation task'), {
      target: { value: 'eval-1:AVS task 2' },
    });

    await waitFor(() => {
      expect(screen.getByLabelText('Select evaluation task').value).toBe('eval-1:AVS task 2');
    });
    await waitFor(() => expect(screen.queryByText(/1 selected/)).toBeNull());
  });

  const ExternalTaskChangeHarness = () => {
    const [selectedTask, setSelectedTask] = React.useState(task);
    return (
      <>
        <button
          type="button"
          onClick={() => setSelectedTask({ ...task, taskName: 'AVS task 2' })}
        >
          Simulate external task change
        </button>
        <AvsWorkspace
          connectedUserId="team-a"
          evaluations={avsEvaluations}
          selectedTask={selectedTask}
          setSelectedTask={setSelectedTask}
          onInspect={jest.fn()}
          onSessionRejected={jest.fn()}
        />
      </>
    );
  };

  test('unexpected task-scope change preserves the old basket and blocks mutation', async () => {
    getCurrentDresTask.mockImplementation(async (_userId, { taskName }) => ({
      user_id: 'team-a',
      evaluation_id: 'eval-1',
      task_scope_key: taskName === 'AVS task 2' ? 'scope-2' : 'scope-1',
      task_name: taskName,
      task_group: 'AVS',
      task_type: 'AVS',
      duration: 300,
    }));
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
    render(<ExternalTaskChangeHarness />);
    await runSearch('seafood');
    fireEvent.click(screen.getByRole('checkbox'));

    fireEvent.click(screen.getByRole('button', { name: 'Simulate external task change' }));

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toMatch(/task scope changed/i);
    expect(screen.getByRole('checkbox').disabled).toBe(true);
    fireEvent.click(screen.getByRole('button', { name: 'Review selections' }));
    expect(screen.getByRole('button', { name: 'Remove f1 from selection' })).toBeTruthy();
  });

  const selectOneForSubmission = async () => {
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V1', 1000)]));
    renderWorkspace();
    await runSearch('seafood');
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: 'Submit 1 answer' }));
    await screen.findByRole('dialog', { name: 'Submit AVS answers' });
  };

  test('RECORDED moves the submitted candidate out of pending state', async () => {
    submitDresAnswers.mockResolvedValueOnce({
      state: 'RECORDED', recorded: true, verdict: 'CORRECT', message: 'recorded',
    });
    await selectOneForSubmission();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

    await screen.findByText('Submitted');
    expect(screen.getByRole('checkbox').disabled).toBe(true);
    expect(screen.queryByRole('button', { name: 'Submit 1 answer' })).toBeNull();
    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  });

  test('NOT_RECORDED leaves the selected candidate pending', async () => {
    submitDresAnswers.mockResolvedValueOnce({
      state: 'NOT_RECORDED', recorded: false, reason: 'DRES_REJECTED', message: 'rejected',
    });
    await selectOneForSubmission();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

    await screen.findByText('rejected');
    expect(screen.getByRole('checkbox').checked).toBe(true);
    expect(screen.getByRole('button', { name: 'Submit 1 answer' })).toBeTruthy();
    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  });

  test('UNKNOWN freezes normal mutation and never retries automatically', async () => {
    submitDresAnswers.mockResolvedValueOnce({
      state: 'UNKNOWN', recorded: null, message: 'check DRES',
    });
    await selectOneForSubmission();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

    await screen.findByText('check DRES');
    expect(screen.getByRole('checkbox').disabled).toBe(true);
    expect(screen.getByRole('button', { name: 'Retry after verification' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Mark recorded after verification' })).toBeTruthy();
    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
  });

  test('UNKNOWN retries only after the explicit verification action', async () => {
    window.confirm = jest.fn().mockReturnValue(true);
    submitDresAnswers
      .mockResolvedValueOnce({ state: 'UNKNOWN', recorded: null, message: 'check DRES' })
      .mockResolvedValueOnce({ state: 'RECORDED', recorded: true, verdict: 'CORRECT', message: 'recorded' });
    await selectOneForSubmission();
    fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));
    await screen.findByRole('button', { name: 'Retry after verification' });

    fireEvent.click(screen.getByRole('button', { name: 'Retry after verification' }));

    expect(window.confirm).toHaveBeenCalledWith(
      'I verified DRES state and want to retry this exact batch.',
    );
    await waitFor(() => expect(submitDresAnswers).toHaveBeenCalledTimes(2));
    await screen.findByText('Submitted');
  });

  test('mark recorded after verification performs no second network request', async () => {
    submitDresAnswers.mockResolvedValueOnce({
      state: 'UNKNOWN', recorded: null, message: 'check DRES',
    });
    await selectOneForSubmission();
    fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));
    await screen.findByRole('button', { name: 'Mark recorded after verification' });

    fireEvent.click(screen.getByRole('button', { name: 'Mark recorded after verification' }));

    expect(submitDresAnswers).toHaveBeenCalledTimes(1);
    await screen.findByText('Submitted');
  });

  test('TASK_SCOPE_MISMATCH preserves pending selection and surfaces conflict', async () => {
    submitDresAnswers.mockRejectedValueOnce(Object.assign(new Error('scope changed'), {
      status: 409,
      code: 'TASK_SCOPE_MISMATCH',
    }));
    await selectOneForSubmission();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm submit' }));

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toMatch(/task scope changed/i);
    expect(screen.getByRole('checkbox').checked).toBe(true);
  });
});
