import { fetchQuerySuggestions } from "./querySuggestions";

beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.restoreAllMocks();
});

test("posts the bounded query-suggestion request", async () => {
  global.fetch.mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({
      request_id: "suggestion-request-1",
      original_query: "a red car",
      suggestions: [
        {
          suggestion_id: "suggestion-1",
          query: "a red vehicle driving on a road",
          language: "en",
          focus: "action",
        },
      ],
      provider: "gpu_inference",
      model: "test-model",
      revision: null,
      generation_latency_ms: 12,
      warnings: [],
    }),
  });

  await fetchQuerySuggestions({ query: "  a red car  ", count: 5 });

  expect(global.fetch).toHaveBeenCalledWith(
    "http://127.0.0.1:8000/api/v1/query-suggestions",
    expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ query: "a red car", count: 5 }),
    }),
  );
});

test("rejects a malformed suggestion response", async () => {
  global.fetch.mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => ({ suggestions: [{ suggestion_id: "missing-query" }] }),
  });

  await expect(
    fetchQuerySuggestions({ query: "a red car", count: 5 }),
  ).rejects.toThrow("invalid response contract");
});
