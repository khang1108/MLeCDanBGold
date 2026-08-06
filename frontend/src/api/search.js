import { API_BASE_URL, requestJson } from "./client";

export const resolveApiUrl = (value) => {
  if (!value || /^(?:https?:|data:)/i.test(value)) return value;
  return `${API_BASE_URL}/${value.replace(/^\/+/, "")}`;
};

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
  signal,
}) => {
  const body = {
    query: query.trim(),
    top_k: topK,
  };
  if (queryType) body.query_type = queryType;

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
