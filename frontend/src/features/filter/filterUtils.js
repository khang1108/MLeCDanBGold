const VIDEO_ID_PATTERN = /(L(?:2[1-9]|30))(?:[._-]|$)/i;
const FOLDER_ID_PATTERN = /^(L(?:2[1-9]|30))(?:[_ .-]|$)/i;

/**
 * Folder options supplied by the current BTC archive.
 */
export const FILTER_FOLDER_IDS = Object.freeze([
  'L21',
  'L22',
  'L23',
  'L24',
  'L25',
  'L26',
  'L27',
  'L28',
  'L29',
  'L30',
]);

export const normalizeFolderId = (folderId) => {
  const value = String(folderId || '')
    .trim()
    .replace(/^.*[\\/]/, '')
    .replace(/^_+/, '')
    .replace(/\.zip$/i, '');
  if (!value) return null;

  // Keep the scope compatible with folder IDs that may still carry a legacy
  // topic suffix while exposing only the current L21-L30 folders in the UI.
  return value.match(FOLDER_ID_PATTERN)?.[1]?.toUpperCase() || value.toUpperCase();
};

/**
 * Read folder scope without changing canonical video/frame identities.
 * Explicit backend folder metadata always wins over the compatibility prefix.
 */
export const getFrameFolderId = (frame) => (
  normalizeFolderId(frame?.folder_id)
  || normalizeFolderId(String(frame?.video_id || '').match(VIDEO_ID_PATTERN)?.[1])
);

export const filterResultsByScope = (results, { folderId = null, videoId = '' } = {}) => (
  results.filter((frame) => (
    (!folderId || getFrameFolderId(frame) === normalizeFolderId(folderId))
    && (!videoId || frame.video_id === videoId)
  ))
);
