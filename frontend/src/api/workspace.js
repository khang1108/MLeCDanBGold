/** Workspace HTTP and WebSocket transport contracts. */

import { API_BASE_URL, requestJson } from './client';

const requireText = (value, field) => {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${field} must be a non-blank string`);
  }
  return value;
};

const requireFrameIds = (frameIds) => {
  if (!Array.isArray(frameIds) || frameIds.length === 0) {
    throw new Error('frameIds must be a non-empty array');
  }
  frameIds.forEach((frameId, index) => {
    requireText(frameId, `frameIds[${index}]`);
  });
  return frameIds;
};

const validateHistoryItem = (item, index) => {
  if (!item || typeof item !== 'object') {
    throw new Error(`Query history item ${index} is not an object`);
  }
  ['query_id', 'query_text', 'submission_files', 'result_snapshot', 'frame_activity']
    .forEach((field) => {
      if (!Object.prototype.hasOwnProperty.call(item, field)) {
        throw new Error(`Query history item ${index} is missing ${field}`);
      }
    });
  requireText(item.query_id, `history[${index}].query_id`);
  if (typeof item.query_text !== 'string') {
    throw new Error(`history[${index}].query_text must be a string`);
  }
  if (!Array.isArray(item.submission_files)) {
    throw new Error(`history[${index}].submission_files must be an array`);
  }
  if (!item.result_snapshot || typeof item.result_snapshot !== 'object') {
    throw new Error(`history[${index}].result_snapshot must be an object`);
  }
  if (!item.frame_activity || typeof item.frame_activity !== 'object'
      || !Array.isArray(item.frame_activity.viewed_frame_ids)
      || !Array.isArray(item.frame_activity.submitted_frame_ids)) {
    throw new Error(`history[${index}].frame_activity must contain both activity arrays`);
  }
  return item;
};

const validateFile = (file, index) => {
  if (!file || typeof file !== 'object') {
    throw new Error(`Submission file ${index} is not an object`);
  }
  if (typeof file.name !== 'string' || !file.name.trim()) {
    throw new Error(`Submission file ${index} has an invalid name`);
  }
  if (typeof file.content !== 'string') {
    throw new Error(`Submission file ${index} has invalid content`);
  }
  if (typeof file.is_validated !== 'boolean') {
    throw new Error(`Submission file ${index} has invalid is_validated`);
  }
  if (!Number.isInteger(file.revision) || file.revision < 0) {
    throw new Error(`Submission file ${index} has invalid revision`);
  }
  return {
    name: file.name,
    content: file.content,
    is_validated: file.is_validated,
    revision: file.revision,
  };
};

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

export const markFrameViewed = async ({ queryId, frameId, signal } = {}) => {
  requireText(queryId, 'queryId');
  requireText(frameId, 'frameId');
  return requestJson(`/api/v1/query-history/${encodeURIComponent(queryId)}/viewed-frame`, {
    method: 'PATCH',
    body: { frame_id: frameId },
    signal,
  });
};

export const markFramesSubmitted = async ({
  queryId, submissionFileName, submissionLine, frameIds, signal,
} = {}) => {
  requireText(queryId, 'queryId');
  requireText(submissionFileName, 'submissionFileName');
  requireText(submissionLine, 'submissionLine');
  requireFrameIds(frameIds);
  return requestJson(`/api/v1/query-history/${encodeURIComponent(queryId)}/submission`, {
    method: 'PATCH',
    body: {
      submission_file_name: submissionFileName,
      submission_line: submissionLine,
      frame_ids: frameIds,
    },
    signal,
  });
};

export const getSubmissionFiles = async ({ signal } = {}) => {
  const payload = await requestJson('/api/v1/submission-files', { signal });
  if (!payload || !Array.isArray(payload.files)) {
    throw new Error('Submission files server returned an invalid files response');
  }
  return {
    ...payload,
    files: payload.files.map(validateFile),
  };
};

export const workspaceWebSocketUrl = () => {
  const url = new URL('/api/v1/workspace/ws', API_BASE_URL);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
};
