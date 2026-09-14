/** HTTP contracts for replayable KIS query history and viewed-frame activity. */

import { requestJson } from './client';

const requireText = (value, field) => {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${field} must be a non-blank string`);
  }
  return value;
};

const validateHistoryItem = (item, index) => {
  if (!item || typeof item !== 'object') {
    throw new Error(`Query history item ${index} is not an object`);
  }
  ['query_id', 'query_text', 'result_snapshot', 'frame_activity'].forEach((field) => {
    if (!Object.prototype.hasOwnProperty.call(item, field)) {
      throw new Error(`Query history item ${index} is missing ${field}`);
    }
  });
  requireText(item.query_id, `history[${index}].query_id`);
  if (typeof item.query_text !== 'string') {
    throw new Error(`history[${index}].query_text must be a string`);
  }
  if (!item.result_snapshot || typeof item.result_snapshot !== 'object') {
    throw new Error(`history[${index}].result_snapshot must be an object`);
  }
  if (!item.frame_activity || typeof item.frame_activity !== 'object'
      || !Array.isArray(item.frame_activity.viewed_frame_ids)) {
    throw new Error(`history[${index}].frame_activity must contain viewed_frame_ids`);
  }
  return item;
};

/** Save one completed retrieval snapshot for the selected local history identity. */
export const createQueryHistory = async ({
  queryId, userId, queryText, resultSnapshot, signal,
} = {}) => {
  requireText(queryId, 'queryId');
  requireText(userId, 'userId');
  requireText(queryText, 'queryText');
  if (!resultSnapshot || typeof resultSnapshot !== 'object') {
    throw new Error('resultSnapshot must be an object');
  }

  return requestJson('/api/v1/query-history', {
    method: 'POST',
    body: {
      query_id: queryId,
      user_id: userId,
      query_text: queryText,
      result_snapshot: resultSnapshot,
    },
    signal,
  });
};

/** Load and validate one user's replayable KIS history rows. */
export const getQueryHistory = async ({ userId, signal } = {}) => {
  requireText(userId, 'userId');
  const payload = await requestJson(
    `/api/v1/query-history?user_id=${encodeURIComponent(userId)}`,
    { signal },
  );
  if (!payload || !Array.isArray(payload.items)) {
    throw new Error('Query history server returned an invalid items response');
  }
  return {
    ...payload,
    items: payload.items.map(validateHistoryItem),
  };
};

/** Record that a canonical frame was opened from a query-history result. */
export const markFrameViewed = async ({ queryId, frameId, signal } = {}) => {
  requireText(queryId, 'queryId');
  requireText(frameId, 'frameId');
  return requestJson(`/api/v1/query-history/${encodeURIComponent(queryId)}/viewed-frame`, {
    method: 'PATCH',
    body: { frame_id: frameId },
    signal,
  });
};
