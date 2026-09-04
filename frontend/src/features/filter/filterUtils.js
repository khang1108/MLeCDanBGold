/** Normalize an optional free-text folder scope without assuming dataset IDs. */
export const normalizeFolderId = (folderId) => {
  const value = String(folderId || '')
    .trim()
    .replace(/^.*[\\/]/, '')
    .replace(/\.zip$/i, '');
  return value ? value.toUpperCase() : null;
};


/** Read folder scope while preserving canonical video and frame identities. */
export const getFrameFolderId = (frame) => (
  normalizeFolderId(frame?.folder_id)
  || normalizeFolderId(String(frame?.video_id || '').split('_')[0])
);


/** Filter an already-loaded result list for presentation-only consumers. */
export const filterResultsByScope = (results, { folderId = null, videoId = '' } = {}) => (
  results.filter((frame) => (
    (!folderId || getFrameFolderId(frame) === normalizeFolderId(folderId))
    && (!videoId || frame.video_id === videoId)
  ))
);
