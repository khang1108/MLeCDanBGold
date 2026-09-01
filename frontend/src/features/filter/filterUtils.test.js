import {
  FILTER_FOLDER_IDS,
  filterResultsByScope,
  getFrameFolderId,
} from './filterUtils';

const frames = [
  { frame_id: 'a1', video_id: 'L26_topic.video-1', folder_id: 'L26' },
  { frame_id: 'a2', video_id: 'L26_topic.video-1', folder_id: 'L26' },
  { frame_id: 'b1', video_id: 'L27_topic.video-1', folder_id: 'L27' },
];

test('lists the current archive folders', () => {
  expect(FILTER_FOLDER_IDS).toEqual([
    'L21', 'L22', 'L23', 'L24', 'L25',
    'L26', 'L27', 'L28', 'L29', 'L30',
  ]);
});

test('filters L26 and L27 independently', () => {
  expect(filterResultsByScope(frames, { folderId: 'L26' }).map((frame) => frame.frame_id))
    .toEqual(['a1', 'a2']);
  expect(filterResultsByScope(frames, { folderId: 'L27' }).map((frame) => frame.frame_id))
    .toEqual(['b1']);
});

test('falls back to the folder prefix when folder metadata is absent', () => {
  expect(getFrameFolderId({ video_id: 'L26_topic.video-1' })).toBe('L26');
  expect(getFrameFolderId({ video_id: 'L26_topic.video-1', folder_id: 'L27' })).toBe('L27');
});
