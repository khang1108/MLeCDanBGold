import { getFrameFolderId, normalizeFolderId } from './filterUtils';


test('normalizes arbitrary free-text folder scopes', () => {
  expect(normalizeFolderId('/dataset/team_a.zip')).toBe('TEAM_A');
  expect(normalizeFolderId('')).toBeNull();
});


test('derives folder scope from canonical video identity', () => {
  expect(getFrameFolderId({ video_id: 'L21_V001' })).toBe('L21');
  expect(getFrameFolderId({ folder_id: 'custom', video_id: 'L21_V001' })).toBe('CUSTOM');
});
