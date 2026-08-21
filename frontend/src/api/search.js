import { API_BASE_URL, requestJson } from "./client";

export const resolveApiUrl = (value) => {
  if (!value || /^(?:https?:|data:)/i.test(value)) return value;
  return `${API_BASE_URL}/${value.replace(/^\/+/, "")}`;
};

export const frameAssetUrl = (frameId, asset) => (
  frameId
    ? resolveApiUrl(`/api/v1/frames/${encodeURIComponent(frameId)}/${asset}`)
    : undefined
);

export const materializeFrameAssets = (frame) => {
  if (!frame || typeof frame.frame_id !== 'string' || !frame.frame_id.trim()) {
    throw new Error('Backend frame response is missing canonical frame_id');
  }
  return {
    ...frame,
    // Asset lookup always uses the internal identity. Backend frame_idx is a
    // BTC submission coordinate and must never become a thumbnail key.
    thumbnail_url: frameAssetUrl(frame.frame_id, 'thumbnail'),
    frame_url: frameAssetUrl(frame.frame_id, 'image'),
  };
};

const withAssetUrls = (payload) => ({
  ...payload,
  results: payload.results.map(materializeFrameAssets),
});

// Executes one standalone competition-task search.
export const searchFrames = async ({
  query,
  topK,
  queryType,
  searchId,
  signal,
}) => {


  const body = {
    query: query.trim(),
    top_k: topK,
  };
  if (queryType) body.query_type = queryType;
  if (searchId) body.search_id = searchId;

  const payload = await requestJson("/api/v1/search", {
    method: "POST",
    body,
    signal,
  });
  if (!Array.isArray(payload?.results) || !payload.latency_ms) {
    throw new Error("Search server returned an invalid response contract");
  }
  return withAssetUrls(payload);
};

export const searchVqa = async ({
  eventDescription,
  question,
  topK,
  searchId,
  signal,
}) => {
  const payload = await requestJson('/api/v1/vqa', {
    method: 'POST',
    body: {
      query_type: 'vqa',
      event_description: eventDescription.trim(),
      question: question.trim(),
      top_k: topK,
      ...(searchId ? { search_id: searchId } : {}),
    },
    signal,
  });
  if (!Array.isArray(payload?.submissions)
      || typeof payload?.latency_ms !== 'number') {
    throw new Error('VQA server returned an invalid response contract');
  }
  payload.submissions = payload.submissions.map(materializeFrameAssets);
  return payload;
};

export const searchTrake = async ({ events, topK, signal }) => {
  const orderedEvents = events.map((event) => event.trim()).filter(Boolean);
  if (orderedEvents.length < 2) {
    throw new Error('TRAKE requires at least two non-empty ordered events');
  }
  const payload = await requestJson('/api/v1/trake', {
    method: 'POST',
    body: {
      query_type: 'trake',
      query: orderedEvents
        .map((event, index) => `E${index + 1}: ${event}`)
        .join('\n'),
      events: orderedEvents,
      top_k: topK,
    },
    signal,
  });
  if (!Array.isArray(payload?.events)
      || !Array.isArray(payload?.submissions)
      || typeof payload?.total_results !== 'number') {
    throw new Error('TRAKE server returned an invalid response contract');
  }
  return payload;
};

export const submitCsvFiles = async (files) => {
  return requestJson('/api/v1/submission', {
    method: 'POST',
    body: { files },
  });
};

export const suggestQueries = async ({ count = 5, query = '', signal } = {}) => {
  const payload = await requestJson('/api/v1/suggest-query', {
    method: 'POST',
    body: { count, current_query: query },
    signal,
  });
  if (Array.isArray(payload)) {
    return payload;
  }
  if (Array.isArray(payload?.suggestions)) {
    return payload.suggestions;
  }
  if (Array.isArray(payload?.queries)) {
    return payload.queries;
  }
  if (Array.isArray(payload?.results)) {
    return payload.results;
  }
  return [];
};

export const uploadQueryFiles = async (fileList) => {
  const formData = new FormData();
  Array.from(fileList).forEach((file) => {
    formData.append('files', file);
  });

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/parse-query-files`, {
      method: 'POST',
      body: formData,
    });
    if (response.ok) {
      const payload = await response.json();
      if (Array.isArray(payload?.files)) {
        return payload.files.map((f, idx) => {
          const name = typeof f === 'string' ? f : f.name;
          const csvName = name.endsWith('.csv') ? name : `${name.replace(/\.[^/.]+$/, '')}.csv`;
          const fallbackName = fileList[idx]?.name || name;
          return {
            id: csvName,
            name: csvName,
            originalName: typeof f === 'object' ? f.originalName || fallbackName : fallbackName,
            content: typeof f === 'object' ? f.content || '' : '',
          };
        });
      }
      if (Array.isArray(payload)) {
        return payload.map(f => {
          const name = typeof f === 'string' ? f : f.name;
          const csvName = name.endsWith('.csv') ? name : `${name.replace(/\.[^/.]+$/, '')}.csv`;
          return {
            id: csvName,
            name: csvName,
            content: typeof f === 'object' ? f.content || '' : '',
          };
        });
      }
    }
  } catch (e) {
    console.warn('Backend parse-query-files endpoint not reachable, performing client parsing', e);
  }

  // Client parsing fallback: map each uploaded query file name to <filename>.csv
  return Array.from(fileList).map((file) => {
    const baseName = file.name.replace(/\.[^/.]+$/, '');
    const csvName = `${baseName}.csv`;
    return {
      id: csvName,
      name: csvName,
      originalName: file.name,
      content: '',
    };
  });
};
