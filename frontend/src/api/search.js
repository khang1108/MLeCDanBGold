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

const withAssetUrls = (payload) => ({
  ...payload,
  results: payload.results.map((frame) => ({
    ...frame,
    thumbnail_url: resolveApiUrl(frame.thumbnail_url),
    frame_url: resolveApiUrl(frame.frame_url),
  })),
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
  payload.submissions = payload.submissions.map((sub) => ({
    ...sub,
    thumbnail_url: resolveApiUrl(sub.thumbnail_url)
      || frameAssetUrl(sub.frame_id, 'thumbnail'),
    frame_url: resolveApiUrl(sub.frame_url)
      || frameAssetUrl(sub.frame_id, 'image'),
  }));
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
      query: orderedEvents.join(' | '),
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
