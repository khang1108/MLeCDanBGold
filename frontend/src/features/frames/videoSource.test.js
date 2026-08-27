import {
  displayVideoId,
  getYouTubeEmbedUrl,
  getYouTubeVideoId,
  getYouTubeWatchUrl,
  timestampSeconds,
} from './videoSource';

test('looks up the YouTube URL using the canonical leaf video ID', () => {
  expect(getYouTubeWatchUrl('folder_1.folder2.L21_V001')).toBe(
    'https://youtube.com/watch?v=Rzpw5WR7nAY',
  );
});

test('extracts video IDs from watch and embed URLs', () => {
  expect(getYouTubeVideoId('https://youtube.com/watch?v=Rzpw5WR7nAY')).toBe('Rzpw5WR7nAY');
  expect(getYouTubeVideoId('https://www.youtube.com/embed/Rzpw5WR7nAY')).toBe('Rzpw5WR7nAY');
});

test('builds an API-enabled embed URL', () => {
  const embedUrl = getYouTubeEmbedUrl('L21_V001');
  expect(embedUrl).toContain('https://www.youtube.com/embed/Rzpw5WR7nAY?');
  expect(embedUrl).toContain('autoplay=0');
  expect(embedUrl).toContain('enablejsapi=1');
  expect(embedUrl).toContain('origin=');
});

test('returns null for unknown video metadata', () => {
  expect(getYouTubeWatchUrl('unknown-video')).toBeNull();
  expect(getYouTubeVideoId('unknown-video')).toBeNull();
  expect(getYouTubeEmbedUrl('unknown-video')).toBeNull();
});

test('uses only the canonical timestamp for exact source seeking', () => {
  expect(timestampSeconds(5_000)).toBe(5);
  expect(timestampSeconds(undefined)).toBeNull();
});

test('only displays the leaf video ID', () => {
  expect(displayVideoId('folder_1.folder2.L21_V001')).toBe('L21_V001');
});
