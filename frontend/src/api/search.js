import { requestJson } from './client';

const hasSearchLatency = (latency) => (
  latency
  && typeof latency === 'object'
  && typeof latency.total_ms === 'number'
);

export const searchFrames = async ({
  query,
  topK,
  useDense = true,
  useBm25 = true,
  signal,
}) => {
  if (!useDense && !useBm25) {
    throw new Error('Enable at least one retrieval source');
  }

  const payload = await requestJson('/api/v1/search', {
    method: 'POST',
    body: {
      query: query.trim(),
      top_k: topK,
      use_dense: useDense,
      use_bm25: useBm25,
    },
    signal,
  });

  if (
    !Array.isArray(payload?.events)
    || !Array.isArray(payload?.results)
    || !hasSearchLatency(payload?.latency)
  ) {
    throw new Error('Search server returned an invalid response contract');
  }

  return payload;
};

export const searchTrake = async ({
  events,
  topK,
  useDense = true,
  useBm25 = true,
  signal,
}) => {
  const orderedEvents = events.map((event) => event.trim()).filter(Boolean);
  if (orderedEvents.length < 1) {
    throw new Error('TRAKE requires at least one non-empty ordered event');
  }
  if (!useDense && !useBm25) {
    throw new Error('Enable at least one retrieval source');
  }

  const payload = await requestJson('/api/v1/trake', {
    method: 'POST',
    body: {
      events: orderedEvents,
      top_k: topK,
      use_dense: useDense,
      use_bm25: useBm25,
    },
    signal,
  });

  if (
    !Array.isArray(payload?.events)
    || !Array.isArray(payload?.paths)
    || !hasSearchLatency(payload?.latency)
  ) {
    throw new Error('TRAKE server returned an invalid response contract');
  }

  return payload;
};
