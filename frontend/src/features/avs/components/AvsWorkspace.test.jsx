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
});
