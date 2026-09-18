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

describe('AvsWorkspace', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders query controls and executes search with query and userId', async () => {
    searchAvs.mockResolvedValueOnce(responseWith([candidate('f1', 'V01', 1000)]));

    render(
      <AvsWorkspace
        connectedUserId="team-a"
        selectedTask={task}
        setSelectedTask={jest.fn()}
        evaluations={[]}
      />
    );

    const input = screen.getByRole('textbox', { name: /avs query/i });
    fireEvent.change(input, { target: { value: 'red car' } });
    fireEvent.click(screen.getByRole('button', { name: /^search$/i }));

    await waitFor(() => {
      expect(searchAvs).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'red car',
          pageSize: 80,
          userId: 'team-a',
        })
      );
    });
  });
});
