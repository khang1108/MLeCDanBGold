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

// Mock Data Generation
const generateMockFrames = (topK) => Array.from({ length: topK }).map((_, i) => ({
  video_id: `demo-video-${(i % 3) + 1}`,
  frame_idx: i * 30 + 100,
  timestamp_ms: i * 30 * 1000,
  scores: { final: 0.99 - (i * 0.01), visual: 0.9, text: 0.8 },
  caption: `This is a mock caption for frame ${i * 30 + 100}. It provides contextual information about the scene.`,
  thumbnail_url: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMTY5Ij48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzQ5OGRiIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5Nb2NrIEZyYW1lPC90ZXh0Pjwvc3ZnPg==",
  frame_url: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMTY5Ij48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzQ5OGRiIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5Nb2NrIEZyYW1lPC90ZXh0Pjwvc3ZnPg==",
}));

const generateMockSubmissions = (topK) => Array.from({ length: topK }).map((_, i) => ({
  rank: i + 1,
  video_id: `demo-video-${(i % 3) + 1}`,
  frame_id: `demo-frame-${i}`,
  frame_idx: i * 30 + 100,
  timestamp_ms: i * 30 * 1000,
  answer: i === 0 ? "This is a mock answer that demonstrates UI behavior." : `Mock answer ${i}`,
  normalized_answer: `mock answer ${i}`,
  joint_score: 0.95 - (i * 0.01),
  retrieval_score: 0.9,
  localization_score: 0.9,
  answer_confidence: 0.99,
  evidence_consistency_score: 1.0,
  provenance: "mock-vlm",
  evidence_summary: `This is a mock evidence summary for frame ${i * 30 + 100}. It explains why this answer was chosen.`,
  thumbnail_url: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMTY5Ij48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzQ5OGRiIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5Nb2NrIEZyYW1lPC90ZXh0Pjwvc3ZnPg==",
  frame_url: "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMTY5Ij48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjMzQ5OGRiIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMjAiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGR5PSIuM2VtIj5Nb2NrIEZyYW1lPC90ZXh0Pjwvc3ZnPg==",
}));

const generateMockSuggestions = (query, count) => Array.from({ length: count }).map((_, i) => ({
  suggestion_id: `s-${i}`,
  query: `${query} ${['driving fast', 'in rain', 'at night', 'with people', 'near a building', 'turning left', 'parked', 'on fire'][i % 8]}`,
  focus: ['action', 'environment', 'object', 'attribute'][i % 4]
}));

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

  try {
    const payload = await requestJson("/api/v1/search", {
      method: "POST",
      body,
      signal,
    });
    if (!Array.isArray(payload?.results) || !payload.latency_ms) {
      throw new Error("Search server returned an invalid response contract");
    }
    return withAssetUrls(payload);
  } catch (err) {
    if (err.message.startsWith('Could not reach the backend:')) {
      console.warn("Backend unreachable, returning mock searchFrames data.");
      await new Promise(r => setTimeout(r, 800)); // simulate delay
      return {
        latency_ms: { total: 150 },
        warnings: ["MOCK DATA: Backend is unreachable."],
        results: generateMockFrames(topK)
      };
    }
    throw err;
  }
};

export const searchVqa = async ({
  eventDescription,
  question,
  topK,
  signal,
}) => {
  try {
    const payload = await requestJson('/api/v1/vqa', {
      method: 'POST',
      body: {
        query_type: 'vqa',
        event_description: eventDescription.trim(),
        question: question.trim(),
        top_k: topK,
      },
      signal,
    });
    if (!Array.isArray(payload?.submissions)
        || typeof payload?.latency_ms !== 'number') {
      throw new Error('VQA server returned an invalid response contract');
    }
    // Process asset URLs if the backend returns them for VQA
    payload.submissions = payload.submissions.map((sub) => ({
      ...sub,
      thumbnail_url: resolveApiUrl(sub.thumbnail_url),
      frame_url: resolveApiUrl(sub.frame_url),
    }));
    return payload;
  } catch (err) {
    if (err.message.startsWith('Could not reach the backend:')) {
      console.warn("Backend unreachable, returning mock searchVqa data.");
      await new Promise(r => setTimeout(r, 1200));
      return {
        latency_ms: 450,
        warnings: ["MOCK DATA: Backend is unreachable."],
        submissions: generateMockSubmissions(topK)
      };
    }
    throw err;
  }
};

export const searchKisc = async ({
  history = [],
  currentMessage,
  previousState = null,
  feedback = {},
  topK = 20,
  filters = null,
  signal,
}) => {
  try {
    const payload = await requestJson('/api/v1/kisc/search', {
      method: 'POST',
      body: {
        history,
        current_message: currentMessage.trim(),
        previous_state: previousState,
        feedback,
        top_k: topK,
        filters,
      },
      signal,
    });
    if (!payload?.interpreted_state || !Array.isArray(payload?.search?.results)) {
      throw new Error('KISC server returned an invalid response contract');
    }
    return { ...payload, search: withAssetUrls(payload.search) };
  } catch (err) {
    if (err.message.startsWith('Could not reach the backend:')) {
      console.warn("Backend unreachable, returning mock searchKisc data.");
      await new Promise(r => setTimeout(r, 800));
      return {
        interpreted_state: { mode: "mock", intent: "demo" },
        search: {
          latency_ms: { total: 150 },
          warnings: ["MOCK DATA: Backend is unreachable."],
          results: generateMockFrames(topK)
        }
      };
    }
    throw err;
  }
};

export const suggestQueries = async ({ query, count = 5, signal }) => {
  try {
    const payload = await requestJson('/api/v1/suggest', {
      method: 'POST',
      body: {
        query: query.trim(),
        count,
      },
      signal,
    });
    if (!Array.isArray(payload?.suggestions)) {
      throw new Error('Suggest server returned an invalid response contract');
    }
    return payload;
  } catch (err) {
    if (err.message.startsWith('Could not reach the backend:')) {
      console.warn("Backend unreachable, returning mock suggestQueries data.");
      await new Promise(r => setTimeout(r, 500));
      return {
        suggestions: generateMockSuggestions(query, count)
      };
    }
    throw err;
  }
};

