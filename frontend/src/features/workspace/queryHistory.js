/** Build replayable Query-history snapshots and canonical activity state.

History snapshots retain the fields needed by the live KIS/TRAKE result
components. They intentionally do not contain images or model embeddings.
*/

const requireText = (value, field) => {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(`${field} must be a non-blank string`);
  }
  return value;
};

const resolveScore = (item, field) => {
  const score = item.score ?? item.scores?.final;
  if (typeof score !== 'number' || !Number.isFinite(score)) {
    throw new Error(`${field} is missing a numeric score`);
  }
  return score;
};

const normalizeFrameIds = (frameIds, field) => {
  if (!Array.isArray(frameIds) || frameIds.length === 0) {
    throw new Error(`${field} must be a non-empty array`);
  }
  frameIds.forEach((frameId, index) => requireText(frameId, `${field}[${index}]`));
  return frameIds.slice();
};

const normalizeNonNegativeIntegers = (values, field, expectedLength) => {
  if (!Array.isArray(values) || values.length !== expectedLength) {
    throw new Error(`${field} must contain ${expectedLength} values`);
  }
  values.forEach((value, index) => {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new Error(`${field}[${index}] must be a non-negative integer`);
    }
  });
  return values.slice();
};

const normalizeEvents = (events, field) => {
  if (!Array.isArray(events)) throw new Error(`${field} must be an array`);
  return events.map((event, index) => requireText(event, `${field}[${index}]`));
};

const normalizeLatency = (latency, field) => {
  if (!latency || typeof latency !== 'object' || Array.isArray(latency)) {
    throw new Error(`${field} must be an object`);
  }
  return { ...latency };
};

const normalizeCaption = (result, field) => {
  const caption = result?.caption ?? result?.metadata?.caption ?? null;
  if (caption !== null && typeof caption !== 'string') {
    throw new Error(`${field} must be a string or null`);
  }
  return caption;
};

const normalizeMetadata = (metadata, field) => {
  if (metadata === undefined || metadata === null) return {};
  if (typeof metadata !== 'object' || Array.isArray(metadata)) {
    throw new Error(`${field} must be an object`);
  }
  return { ...metadata };
};

const normalizeSnapshotOptions = (options, field) => {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new Error(`${field} options must be an object`);
  }
  return {
    events: normalizeEvents(options.events, `${field}.events`),
    latency: normalizeLatency(options.latency, `${field}.latency`),
    warnings: options.warnings === undefined
      ? []
      : normalizeEvents(options.warnings, `${field}.warnings`),
  };
};

export const buildKisSnapshot = (results, options) => {
  if (!Array.isArray(results)) throw new Error('KIS results must be an array');
  const { events, latency, warnings } = normalizeSnapshotOptions(options, 'KIS snapshot');
  return {
    events,
    latency,
    warnings,
    results: results.map((result, index) => {
      const frameId = requireText(result?.frame_id, `results[${index}].frame_id`);
      const frameIds = normalizeFrameIds(
        result.frame_ids || [frameId],
        `results[${index}].frame_ids`,
      );
      const timestampsMs = normalizeNonNegativeIntegers(
        result.timestamps_ms || [result.timestamp_ms],
        `results[${index}].timestamps_ms`,
        frameIds.length,
      );
      const frameIdx = normalizeNonNegativeIntegers(
        [result.frame_idx],
        `results[${index}].frame_idx`,
        1,
      )[0];
      const timestampMs = normalizeNonNegativeIntegers(
        [result.timestamp_ms],
        `results[${index}].timestamp_ms`,
        1,
      )[0];
      const metadata = normalizeMetadata(result.metadata, `results[${index}].metadata`);
      return {
        // Keep optional fields from the live /search result (for example fps,
        // folder_id, or rank) so Replay receives the same frame object.
        ...result,
        frame_id: frameId,
        video_id: requireText(result.video_id, `results[${index}].video_id`),
        frame_idx: frameIdx,
        timestamp_ms: timestampMs,
        score: resolveScore(result, `results[${index}].score`),
        frame_ids: frameIds,
        timestamps_ms: timestampsMs,
        caption: normalizeCaption(result, `results[${index}].caption`),
        metadata,
      };
    }),
  };
};

export const buildTrakeSnapshot = (paths, options) => {
  if (!Array.isArray(paths)) throw new Error('TRAKE paths must be an array');
  const { events, latency, warnings } = normalizeSnapshotOptions(options, 'TRAKE snapshot');
  return {
    events,
    latency,
    warnings,
    paths: paths.map((path, index) => ({
      video_id: requireText(path?.video_id, `paths[${index}].video_id`),
      score: resolveScore(path, `paths[${index}].score`),
      frame_ids: (() => {
        const frameIds = normalizeFrameIds(path.frame_ids, `paths[${index}].frame_ids`);
        normalizeNonNegativeIntegers(
          path.frame_idxs,
          `paths[${index}].frame_idxs`,
          frameIds.length,
        );
        normalizeNonNegativeIntegers(
          path.timestamps_ms,
          `paths[${index}].timestamps_ms`,
          frameIds.length,
        );
        return frameIds;
      })(),
      frame_idxs: path.frame_idxs.slice(),
      timestamps_ms: path.timestamps_ms.slice(),
    })),
  };
};

export const getSnapshotKind = (resultSnapshot) => {
  if (!resultSnapshot || typeof resultSnapshot !== 'object') {
    throw new Error('resultSnapshot must be an object');
  }
  const hasResults = Array.isArray(resultSnapshot.results);
  const hasPaths = Array.isArray(resultSnapshot.paths);
  if (hasResults === hasPaths) {
    throw new Error('resultSnapshot must contain exactly one of results or paths');
  }
  return hasResults ? 'kis' : 'trake';
};

const toSet = (values, field) => {
  const entries = values instanceof Set ? Array.from(values) : values;
  if (!Array.isArray(entries)) throw new Error(`${field} must be an array`);
  entries.forEach((frameId, index) => requireText(frameId, `${field}[${index}]`));
  return new Set(entries);
};

export const normalizeFrameActivity = (frameActivity = {}) => ({
  viewedFrameIds: toSet(
    frameActivity.viewedFrameIds || frameActivity.viewed_frame_ids || [],
    'viewed_frame_ids',
  ),
  submittedFrameIds: toSet(
    frameActivity.submittedFrameIds || frameActivity.submitted_frame_ids || [],
    'submitted_frame_ids',
  ),
});

export const activityStateForFrame = (frameId, frameActivity) => {
  requireText(frameId, 'frameId');
  const normalized = normalizeFrameActivity(frameActivity);
  if (normalized.submittedFrameIds.has(frameId)) return 'submitted';
  if (normalized.viewedFrameIds.has(frameId)) return 'viewed';
  return 'neutral';
};

export const withViewedFrame = (frameActivity, frameId) => {
  requireText(frameId, 'frameId');
  const normalized = normalizeFrameActivity(frameActivity);
  normalized.viewedFrameIds.add(frameId);
  return normalized;
};

export const withSubmittedFrames = (frameActivity, frameIds) => {
  const normalized = normalizeFrameActivity(frameActivity);
  normalizeFrameIds(frameIds, 'frameIds').forEach((frameId) => {
    normalized.submittedFrameIds.add(frameId);
  });
  return normalized;
};
