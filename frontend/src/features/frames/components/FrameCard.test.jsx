import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import FrameCard, { getAnswerFrameClassName } from './FrameCard';
import { useOptionalAnswerWorkspace } from '../../answer-workspace/contexts/AnswerWorkspaceContext';

jest.mock('../../answer-workspace/contexts/AnswerWorkspaceContext', () => ({
  useOptionalAnswerWorkspace: jest.fn(),
}));

beforeEach(() => {
  useOptionalAnswerWorkspace.mockReturnValue(null);
});

test('does not expose a direct submission action on a result card', () => {
  const frame = {
    frame_id: 'internal-frame-1',
    video_id: 'L21_V001',
    frame_idx: 17794,
    caption: 'A sample frame',
  };
  const onSubmit = jest.fn();
  render(<FrameCard frame={frame} onClick={jest.fn()} onSubmit={onSubmit} />);

  expect(screen.queryByRole('button', { name: /submit/i })).toBeNull();
  expect(onSubmit).not.toHaveBeenCalled();
});

test('shows the raw alignment score and representative alignment path', () => {
  const frame = {
    frame_id: 'representative-frame',
    video_id: 'L21_V001',
    frame_idx: 17794,
    score: 2.34567,
    frame_ids: ['f1', 'f2'],
    timestamps_ms: [1200, 2400],
    metadata: { caption: 'A sample frame' },
  };
  render(
    <FrameCard
      frame={frame}
      events={['hold', 'roll']}
      onClick={jest.fn()}
    />,
  );

  expect(screen.getByText('Alignment score: 2.346')).toBeTruthy();
  expect(screen.queryByText(/score details/i)).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: /alignment/i }));
  expect(screen.getByText('roll')).toBeTruthy();
});

test('shows a loading placeholder while page details are fetched', () => {
  render(
    <FrameCard
      frame={{
        frame_id: 'page-frame',
        video_id: 'L21_a_topic.video-1',
        frame_idx: 1,
        timestamp_ms: 100,
      }}
      detailStatus="loading"
    />,
  );

  expect(screen.getByText('Loading frame…')).toBeTruthy();
});

test('does not render a submitted control from query-history styling', () => {
  const frame = {
    frame_id: 'internal-frame-1',
    video_id: 'L21_V001',
    frame_idx: 17794,
    caption: 'A sample frame',
  };
  render(<FrameCard frame={frame} className="submitted" onClick={jest.fn()} onSubmit={jest.fn()} />);

  expect(screen.queryByRole('button', { name: /submit/i })).toBeNull();
});

test('answer-workspace membership adds a candidate class without implying submission', () => {
  const frame = {
    frame_id: 'workspace-frame',
    video_id: 'V01',
    timestamp_ms: 12_345,
  };
  const candidate = {
    kind: 'FRAME',
    source_frame_id: 'workspace-frame',
    video_id: 'V01',
    timestamp_ms: 12_345,
  };

  expect(getAnswerFrameClassName(frame, [candidate])).toBe('candidate');
  expect(getAnswerFrameClassName(frame, [{ ...candidate, source_frame_id: 'different-frame' }]))
    .toBe('');
});

test('DRES submitted styling comes from answer-workspace submission state', () => {
  const frame = { frame_id: 'workspace-frame', video_id: 'V01', timestamp_ms: 12_345 };
  const candidate = {
    kind: 'FRAME',
    source_frame_id: 'workspace-frame',
    video_id: 'V01',
    timestamp_ms: 12_345,
    submitted_at_ms: 1_700_000_000_000,
    submitted_by_user_id: 'team-a',
    dres_status: 'SUBMITTED',
  };

  expect(getAnswerFrameClassName(frame, [candidate])).toBe('submitted');
});

test('FrameCard combines viewed history styling with current answer-workspace candidate state', () => {
  useOptionalAnswerWorkspace.mockReturnValue({
    candidates: [{
      kind: 'FRAME',
      source_frame_id: 'workspace-frame',
      video_id: 'V01',
      timestamp_ms: 12_345,
    }],
  });
  const { rerender } = render(
    <FrameCard
      frame={{ frame_id: 'workspace-frame', video_id: 'V01', timestamp_ms: 12_345 }}
      className="viewed"
    />,
  );
  const card = screen.getByText('No caption available').closest('.frame-card');
  expect(card.className).toContain('viewed');
  expect(card.className).toContain('candidate');
  expect(card.className).not.toContain('submitted');

  useOptionalAnswerWorkspace.mockReturnValue({
    candidates: [{
      kind: 'FRAME',
      source_frame_id: 'workspace-frame',
      video_id: 'V01',
      timestamp_ms: 12_345,
      submitted_at_ms: 1_700_000_000_000,
    }],
  });
  rerender(
    <FrameCard
      frame={{ frame_id: 'workspace-frame', video_id: 'V01', timestamp_ms: 12_345 }}
      className="viewed"
    />,
  );
  expect(screen.getByText('No caption available').closest('.frame-card').className)
    .toContain('submitted');
});

test('adds the exact result timestamp to the review workspace without using frame_idx', () => {
  const frame = {
    frame_id: 'internal-frame-12000',
    video_id: 'V01',
    frame_idx: 4,
    timestamp_ms: 12_000,
  };
  const onAddCandidate = jest.fn();
  const onClick = jest.fn();
  render(
    <FrameCard
      frame={frame}
      workspaceAction="add-candidate"
      onAddCandidate={onAddCandidate}
      onClick={onClick}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: 'Add frame to answer workspace' }));

  expect(onAddCandidate).toHaveBeenCalledWith({ kind: 'FRAME', videoId: 'V01', timestampMs: 12_000 });
  expect(onClick).not.toHaveBeenCalled();
});

test('keeps the original frame timestamp when fetched display details differ', () => {
  const frame = { frame_id: 'canonical-frame', video_id: 'V01', frame_idx: 3, timestamp_ms: 12_345 };
  const onAddCandidate = jest.fn();
  render(
    <FrameCard
      frame={frame}
      detail={{ video_id: 'V99', timestamp_ms: 99_000 }}
      workspaceAction="add-candidate"
      onAddCandidate={onAddCandidate}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: 'Add frame to answer workspace' }));
  expect(onAddCandidate).toHaveBeenCalledWith({ kind: 'FRAME', videoId: 'V01', timestampMs: 12_345 });
});
