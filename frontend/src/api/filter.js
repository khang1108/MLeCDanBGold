import { requestJson } from './client';
import { DEFAULT_FRAMES_PER_PAGE } from '../features/filter/filterPagination';
import { normalizeFolderId } from '../features/filter/filterUtils';


/** Build the literal Filter request; matching and normalization belong to BE. */
export const buildFilterRequest = (
  query,
  {
    folderId = null,
    videoId = null,
    framesPerPage = DEFAULT_FRAMES_PER_PAGE,
    pageId = 1,
  } = {},
) => ({
  query: String(query || '').trim(),
  folder_id: normalizeFolderId(folderId),
  video_id: String(videoId || '').trim() || null,
  frames_per_pages: framesPerPage,
  page_id: pageId,
});


/** Request one backend-owned page of direct evidence-text matches. */
export const filterFrames = async ({
  query,
  folderId = null,
  videoId = null,
  framesPerPage = DEFAULT_FRAMES_PER_PAGE,
  pageId = 1,
  signal,
} = {}) => requestJson('/api/v1/filter', {
  method: 'POST',
  body: buildFilterRequest(query, {
    folderId,
    videoId,
    framesPerPage,
    pageId,
  }),
  signal,
});
