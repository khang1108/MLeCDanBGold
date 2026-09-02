import { requestJson, resolveApiUrl } from './client';
import { DEFAULT_FRAMES_PER_PAGE } from '../features/filter/filterPagination';
import { normalizeFolderId } from '../features/filter/filterUtils';

const FILTER_TEXT_FIELDS = ['title', 'asr', 'caption', 'ocr'];

// The backend owns matching semantics. The FE only makes text input stable
// across casing and Vietnamese diacritic variants before it is sent.
export const normalizeFilterText = (value) => String(value || '')
  .normalize('NFD')
  .replace(/[\u0300-\u036f]/g, '')
  .replace(/đ/g, 'd')
  .replace(/Đ/g, 'D')
  .toLowerCase()
  .trim()
  .replace(/\s+/g, ' ');

const objectEntries = (objects) => {
  if (Array.isArray(objects)) {
    return objects.map((row) => {
      if (row?.value !== undefined) {
        const separatorIndex = String(row.value).lastIndexOf(':');
        if (separatorIndex < 0) return [row.value, ''];
        return [
          String(row.value).slice(0, separatorIndex),
          String(row.value).slice(separatorIndex + 1),
        ];
      }
      return [row?.name, row?.count];
    });
  }
  return Object.entries(objects || {});
};

/**
 * Serialize object rows into the additive backend contract.
 *
 * Empty rows are ignored. Repeated normalized names are merged by adding
 * their numeric counts, while the backend remains responsible for deciding
 * whether a count means exact, minimum, or another retrieval constraint.
 */
export const serializeObjectFilters = (objects = []) => objectEntries(objects)
  .reduce((serialized, [rawName, rawCount]) => {
    const name = normalizeFilterText(rawName);
    const countText = String(rawCount ?? '').trim();
    const count = Number(countText);
    if (!name || !countText || !Number.isFinite(count)) return serialized;

    serialized[name] = (serialized[name] || 0) + count;
    return serialized;
  }, {});

/**
 * Build the stable Filter payload for POST /api/v1/filter.
 * Folder/video scope is sent at the top level so BE owns pagination scope.
 */
export const buildFilterRequest = (
  filters = {},
  {
    folderId = null,
    videoId = null,
    framesPerPage = DEFAULT_FRAMES_PER_PAGE,
    pageId = 1,
  } = {},
) => ({
  metadata_filters: {
    ...Object.fromEntries(FILTER_TEXT_FIELDS.map((field) => [
      field,
      normalizeFilterText(filters[field]) || null,
    ])),
    objects: serializeObjectFilters(filters.objects),
  },
  folder_id: normalizeFolderId(folderId),
  video_id: String(videoId || '').trim() || null,
  frames_per_pages: framesPerPage,
  page_id: pageId,
});

const normalizeFilterResult = (frame) => {
  if (!frame || typeof frame.frame_id !== 'string' || !frame.frame_id.trim()) {
    throw new Error('Filter server returned a frame without canonical frame_id');
  }
  if (typeof frame.video_id !== 'string' || !frame.video_id.trim()) {
    throw new Error('Filter server returned a frame without canonical video_id');
  }
  if (!Number.isInteger(frame.frame_idx) || frame.frame_idx < 0) {
    throw new Error('Filter server returned an invalid canonical frame_idx');
  }

  const timestampMs = frame.timestamp_ms ?? frame.timestamp;
  if (!Number.isInteger(timestampMs) || timestampMs < 0) {
    throw new Error('Filter server returned an invalid canonical timestamp');
  }
  if (typeof frame.folder_id !== 'string' || !frame.folder_id.trim()) {
    throw new Error('Filter server returned an invalid folder_id');
  }
  if ('image_path' in frame || 'thumbnail_path' in frame || 'database_path' in frame) {
    throw new Error('Filter server returned a forbidden filesystem path');
  }

  ['title', 'caption', 'ocr', 'asr'].forEach((field) => {
    if (!Object.prototype.hasOwnProperty.call(frame, field)
        || (frame[field] !== null && typeof frame[field] !== 'string')) {
      throw new Error(`Filter server returned invalid ${field} metadata`);
    }
  });
  if (!frame.objects || Array.isArray(frame.objects) || typeof frame.objects !== 'object'
      || Object.entries(frame.objects).some(([label, count]) => (
        !label.trim() || !Number.isInteger(count) || count < 0
      ))) {
    throw new Error('Filter server returned invalid object metadata');
  }

  return {
    ...frame,
    timestamp_ms: timestampMs,
    ...(frame.frame_url ? { frame_url: resolveApiUrl(frame.frame_url) } : {}),
    ...(frame.thumbnail_url
      ? { thumbnail_url: resolveApiUrl(frame.thumbnail_url) }
      : {}),
  };
};

/**
 * Execute one page of the metadata-filter request and validate its response.
 */
export const filterFrames = async ({
  filters,
  folderId = null,
  videoId = null,
  framesPerPage = DEFAULT_FRAMES_PER_PAGE,
  pageId = 1,
  signal,
} = {}) => {
  if (!Number.isInteger(framesPerPage) || framesPerPage < 1
      || !Number.isInteger(pageId) || pageId < 1) {
    throw new Error('Filter request pagination values must be positive integers');
  }

  const payload = await requestJson('/api/v1/filter', {
    method: 'POST',
    body: buildFilterRequest(filters, {
      folderId,
      videoId,
      framesPerPage,
      pageId,
    }),
    signal,
  });

  if (!Number.isInteger(payload?.page_id) || payload.page_id < 1) {
    throw new Error('Filter server returned an invalid page_id value');
  }
  if (payload.page_id !== pageId) {
    throw new Error(
      `Filter server returned page_id ${payload.page_id} for requested page ${pageId}`,
    );
  }
  if (!Number.isInteger(payload?.frames_per_pages)
      || payload.frames_per_pages < 1 || payload.frames_per_pages > 48) {
    throw new Error('Filter server returned an invalid frames_per_pages value');
  }
  if (payload.frames_per_pages !== framesPerPage) {
    throw new Error(
      `Filter server returned frames_per_pages ${payload.frames_per_pages} `
      + `for requested size ${framesPerPage}`,
    );
  }
  if (!Number.isInteger(payload?.total_pages) || payload.total_pages < 0) {
    throw new Error('Filter server returned an invalid total_pages value');
  }
  if (!Array.isArray(payload?.results)) {
    throw new Error('Filter server returned an invalid response contract');
  }
  if (!Number.isInteger(payload?.total_results) || payload.total_results < 0) {
    throw new Error('Filter server returned an invalid result count');
  }

  return {
    ...payload,
    results: payload.results.map(normalizeFilterResult),
  };
};
