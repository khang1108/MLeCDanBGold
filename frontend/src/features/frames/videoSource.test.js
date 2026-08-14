import {
  displayVideoId,
  frameIndexAt,
  getS3VideoUrl,
  s3VideoObjectKey,
  targetTimeSeconds,
} from './videoSource';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

jest.mock('@aws-sdk/client-s3', () => ({
  S3Client: jest.fn().mockImplementation((config) => ({ config })),
  GetObjectCommand: jest.fn().mockImplementation((input) => ({ input })),
}));
jest.mock('@aws-sdk/s3-request-presigner', () => ({ getSignedUrl: jest.fn() }));

beforeEach(() => getSignedUrl.mockResolvedValue('https://signed.example/video.mp4'));

test('creates a presigned URL for the canonical path-bearing video ID', async () => {
  await expect(getS3VideoUrl('folder_1.folder2.L21_V001', {
    bucket: 'hcmai-video-bucket',
    region: 'ap-southeast-2',
    accessKeyId: 'access',
    secretAccessKey: 'secret',
  })).resolves.toBe('https://signed.example/video.mp4');
  expect(s3VideoObjectKey('folder_1.folder2.L21_V001')).toBe('data/folder_1/folder2/L21_V001.mp4');
});

test('does not create an S3 URL without complete credentials', async () => {
  await expect(getS3VideoUrl('folder.video', { bucket: '', region: 'ap-southeast-2' })).resolves.toBeNull();
});

test('uses the backend FPS for seeking and current frame display', () => {
  expect(targetTimeSeconds(125, 25)).toBe(5);
  expect(frameIndexAt(12.48, 25)).toBe(312);
});

test('only displays the leaf video ID', () => {
  expect(displayVideoId('folder_1.folder2.L21_V001')).toBe('L21_V001');
});
