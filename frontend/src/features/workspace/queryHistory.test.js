import {
  activityStateForFrame,
  buildKisSnapshot,
  getSnapshotKind,
  normalizeFrameActivity,
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

test('preserves full KIS semantic intent graph with entities, events, and temporal edges', () => {
  const mockIntent = {
    revision: 2,
    inputs: ['A woman enters the kitchen', 'She takes a white plate'],
    language: 'en',
    query_text: 'A woman enters the kitchen and takes a white plate',
    entities: [
      { id: 'X1', kind: 'person', description: 'woman' },
      { id: 'X2', kind: 'object', description: 'white plate' },
    ],
    events: [
      { id: 'E1', text: 'woman enters kitchen' },
      { id: 'E2', text: 'woman takes white plate' },
    ],
    temporal_edges: [
      { source: 'E1', relation: 'before', target: 'E2' },
    ],
  };

  const snapshot = buildKisSnapshot([{
    frame_id: 'frame-1',
    video_id: 'V01',
    frame_idx: 10,
    timestamp_ms: 1000,
    score: 0.95,
    frame_ids: ['frame-1', 'frame-2'],
    timestamps_ms: [1000, 2000],
    metadata: { caption: 'Kitchen scene' },
  }], {
    intent: mockIntent,
    latency: { total_ms: 15 },
  });

  expect(snapshot.intent).toEqual(mockIntent);
  expect(snapshot.intent.inputs).toEqual(['A woman enters the kitchen', 'She takes a white plate']);
  expect(snapshot.intent.entities).toEqual(mockIntent.entities);
  expect(snapshot.intent.events).toEqual(mockIntent.events);
  expect(snapshot.intent.temporal_edges).toEqual(mockIntent.temporal_edges);
  expect(snapshot.events).toBeUndefined();
  expect(snapshot.results[0].frame_id).toBe('frame-1');
  expect(snapshot.results[0].timestamps_ms).toEqual([1000, 2000]);
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

test('classifies KIS snapshots and keeps legacy path snapshots unsupported', () => {
  expect(getSnapshotKind({ results: [] })).toBe('kis');
  expect(getSnapshotKind({ paths: [] })).toBe('unsupported');
  expect(() => getSnapshotKind({ results: [], paths: [] })).toThrow(/exactly one/);
  expect(() => getSnapshotKind({})).toThrow(/exactly one/);
});

test('query history tracks viewed state only and ignores legacy submitted ids', () => {
  const activity = normalizeFrameActivity({
    viewed_frame_ids: ['frame-1', 'frame-2'],
  });
  expect(activity).toEqual({ viewedFrameIds: new Set(['frame-1', 'frame-2']) });
  expect(activityStateForFrame('frame-2', activity)).toBe('viewed');
  expect(activityStateForFrame('frame-1', activity)).toBe('viewed');
  expect(activityStateForFrame('frame-3', activity)).toBe('neutral');
  expect(withViewedFrame(activity, 'frame-3').viewedFrameIds).toEqual(
    new Set(['frame-1', 'frame-2', 'frame-3']),
  );
});
