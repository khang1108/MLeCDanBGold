import { requestJson } from "./client";

export const fetchQuerySuggestions = async ({ query, count, signal }) => {
  const payload = await requestJson("/api/v1/query-suggestions", {
    method: "POST",
    body: { query: query.trim(), count },
    signal,
  });

  if (
    !Array.isArray(payload?.suggestions) ||
    payload.suggestions.some(
      (item) =>
        !item?.suggestion_id ||
        typeof item?.query !== "string" ||
        !item.query.trim(),
    )
  ) {
    throw new Error("Query-suggestion server returned an invalid response contract");
  }

  return payload;
};
