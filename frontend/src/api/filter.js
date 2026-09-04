import { requestJson } from './client';
import { FRAMES_PER_PAGE } from '../features/filter/filterPagination';


const FILTER_TEXT_FIELDS = ['title', 'asr', 'caption', 'ocr'];


/** Normalize the optional backend-only folder scope without filtering results. */
const normalizeFolderId = (folderId) => {
  const value = String(folderId || '')
    .trim()
    .replace(/^.*[\\/]/, '')
    .replace(/\.zip$/i, '');
  return value ? value.toUpperCase() : null;
};


/** Normalize an object label to the same literal key used by the backend. */
const normalizeObjectLabel = (value) => String(value || '')
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .replace(/đ/g, 'd')
  .replace(/Đ/g, 'D')
  .toLowerCase()
  .trim()
  .replace(/\s+/g, ' ');


/** Convert repeatable ``name: count`` rows into minimum object thresholds. */
export const serializeObjectFilters = (objects = []) => objects.reduce((filters, row) => {
  const separator = String(row?.value || '').lastIndexOf(':');
  if (separator < 0) return filters;

  const label = normalizeObjectLabel(String(row.value).slice(0, separator));
  const minimumCount = Number(String(row.value).slice(separator + 1).trim());
  if (!label || !Number.isInteger(minimumCount) || minimumCount < 1) return filters;

  // Duplicate conditions are AND predicates, so only the strictest threshold
  // changes the result set. Adding them would incorrectly require more objects.
  filters[label] = Math.max(filters[label] || 0, minimumCount);
  return filters;
}, {});


/** Build the FE-owned fixed-size request for independent evidence fields. */
export const buildFilterRequest = (
  filters = {},
  { folderId = null, videoId = null, pageId = 1 } = {},
) => ({
  metadata_filters: {
    ...Object.fromEntries(FILTER_TEXT_FIELDS.map((field) => {
      const value = String(filters[field] || '').trim();
      return [field, value || null];
    })),
    objects: serializeObjectFilters(filters.objects),
  },
  // These scopes are request predicates only. The UI must never rescope the
  // page returned by the backend after pagination has already been applied.
  folder_id: normalizeFolderId(folderId),
  video_id: String(videoId || '').trim() || null,
  frames_per_pages: FRAMES_PER_PAGE,
  page_id: pageId,
});


/** Request one backend-owned Filter page using the invariant page size. */
export const filterFrames = async ({
  filters = {},
  folderId = null,
  videoId = null,
  pageId = 1,
  signal,
} = {}) => {
  if (!Number.isInteger(pageId) || pageId < 1) {
    throw new Error('Filter request page_id must be a positive integer');
  }

  const payload = await requestJson('/api/v1/filter', {
    method: 'POST',
    body: buildFilterRequest(filters, { folderId, videoId, pageId }),
    signal,
  });

  if (payload?.frames_per_pages !== FRAMES_PER_PAGE) {
    throw new Error('Filter server returned a page size other than 20');
  }

  return payload;
};
