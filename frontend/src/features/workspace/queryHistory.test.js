import {
  activityStateForFrame,
  buildKisSnapshot,
  buildTrakeSnapshot,
  getSnapshotKind,
  normalizeFrameActivity,
  withSubmittedFrames,
  withViewedFrame,
} from './queryHistory';

test('builds a replayable KIS snapshot with the complete live-search result', () => {
  expect(buildKisSnapshot([{
    rank: 1,
    frame_id: 'frame-1',
    video_id: 'V01',
    frame_idx: 10,
    timestamp_ms: 1000,
    fps: 29.97,
    folder_id: 'L21',
    frame_ids: ['frame-1'],
    timestamps_ms: [1000],
    score: 0.93,
    metadata: {
      title: 'video title',
      caption: 'caption',
      ocr: 'text in frame',
      objects: ['person'],
      asr: 'spoken text',
    },
  }], {
    events: ['event'],
    latency: { total_ms: 7 },
    warnings: ['warning'],
  })).toEqual({
    events: ['event'],
    latency: { total_ms: 7 },
    warnings: ['warning'],
    results: [{
      rank: 1,
      frame_id: 'frame-1',
      video_id: 'V01',
      frame_idx: 10,
      timestamp_ms: 1000,
      fps: 29.97,
      folder_id: 'L21',
      score: 0.93,
      frame_ids: ['frame-1'],
      timestamps_ms: [1000],
      caption: 'caption',
      metadata: {
        title: 'video title',
        caption: 'caption',
        ocr: 'text in frame',
        objects: ['person'],
        asr: 'spoken text',
      },
    }],
  });
});

test('uses the current or legacy final score field for KIS', () => {
  expect(buildKisSnapshot([{
    frame_id: 'frame-1',
    video_id: 'V01',
    frame_idx: 10,
    timestamp_ms: 1000,
    scores: { final: 0.8 },
  }], {
    events: ['event'],
    latency: { total_ms: 1 },
  })).toEqual({
    results: [{
      frame_id: 'frame-1',
      video_id: 'V01',
      frame_idx: 10,
      timestamp_ms: 1000,
      scores: { final: 0.8 },
      score: 0.8,
      frame_ids: ['frame-1'],
      timestamps_ms: [1000],
      caption: null,
      metadata: {},
    }],
    events: ['event'],
    latency: { total_ms: 1 },
    warnings: [],
  });
});

test('preserves TRAKE display data and event-frame order', () => {
  expect(buildTrakeSnapshot([{
    video_id: 'V01',
    score: 2.4,
    frame_ids: ['third', 'first', 'second'],
    frame_idxs: [30, 10, 20],
    timestamps_ms: [3, 1, 2],
  }], {
    events: ['first', 'second', 'third'],
    latency: { total_ms: 8 },
  })).toEqual({
    events: ['first', 'second', 'third'],
    latency: { total_ms: 8 },
    warnings: [],
    paths: [{
      video_id: 'V01',
      score: 2.4,
      frame_ids: ['third', 'first', 'second'],
      frame_idxs: [30, 10, 20],
      timestamps_ms: [3, 1, 2],
    }],
  });
});

test('distinguishes only unambiguous snapshot discriminators', () => {
  expect(getSnapshotKind({ results: [] })).toBe('kis');
  expect(getSnapshotKind({ paths: [] })).toBe('trake');
  expect(() => getSnapshotKind({ results: [], paths: [] })).toThrow(/exactly one/);
  expect(() => getSnapshotKind({})).toThrow(/exactly one/);
});

test('activity uses canonical ids and gives submitted state priority', () => {
  const activity = normalizeFrameActivity({
    viewed_frame_ids: ['frame-1', 'frame-2'],
    submitted_frame_ids: ['frame-2'],
  });
  expect(activityStateForFrame('frame-2', activity)).toBe('submitted');
  expect(activityStateForFrame('frame-1', activity)).toBe('viewed');
  expect(activityStateForFrame('frame-3', activity)).toBe('neutral');
  expect(withViewedFrame(activity, 'frame-3').viewedFrameIds).toEqual(
    new Set(['frame-1', 'frame-2', 'frame-3']),
  );
  expect(withSubmittedFrames(activity, ['frame-2', 'frame-4']).submittedFrameIds).toEqual(
    new Set(['frame-2', 'frame-4']),
  );
});
