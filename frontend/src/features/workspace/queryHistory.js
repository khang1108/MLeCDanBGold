/** Build replayable KIS Query-history snapshots and canonical activity state.

History snapshots retain the fields needed by the live KIS result components.
They intentionally do not contain images or model embeddings.
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

const roundLatencyVal = (val) => {
  if (typeof val !== 'number' || !Number.isFinite(val)) return val;
  return Math.round(val * 100) / 100;
};

const normalizeLatency = (latency, field) => {
  if (!latency || typeof latency !== 'object' || Array.isArray(latency)) {
    throw new Error(`${field} must be an object`);
  }
  const normalized = { ...latency };
  Object.keys(normalized).forEach((key) => {
    if (typeof normalized[key] === 'number') {
      normalized[key] = roundLatencyVal(normalized[key]);
    }
  });
  return normalized;
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

export const buildOperationMetadata = (meta = {}) => {
  if (!meta || typeof meta !== 'object') return null;
  const semanticRevision = typeof meta.semantic_revision === 'number'
    ? meta.semantic_revision
    : typeof meta.semanticRevision === 'number'
      ? meta.semanticRevision
      : 0;
  const operationKind = typeof meta.operation_kind === 'string'
    ? meta.operation_kind
    : typeof meta.operationKind === 'string'
      ? meta.operationKind
      : 'initial_resolve';
  const affectedEventIds = Array.isArray(meta.affected_event_ids)
    ? meta.affected_event_ids
    : Array.isArray(meta.affectedEventIds)
      ? meta.affectedEventIds
      : [];
  const imageAdded = Array.isArray(meta.image_added)
    ? meta.image_added
    : Array.isArray(meta.imageAdded)
      ? meta.imageAdded
      : [];
  const imageRemoved = Array.isArray(meta.image_removed)
    ? meta.image_removed
    : Array.isArray(meta.imageRemoved)
      ? meta.imageRemoved
      : [];
  const searchOnly = Boolean(meta.search_only ?? meta.searchOnly);
  const action = typeof meta.action === 'string' ? meta.action : null;
  const scope = typeof meta.scope === 'string' ? meta.scope : null;
  const feedbackRevision = typeof meta.feedback_revision === 'number'
    ? meta.feedback_revision
    : typeof meta.feedbackRevision === 'number'
      ? meta.feedbackRevision
      : null;

  return {
    semantic_revision: semanticRevision,
    operation_kind: operationKind,
    affected_event_ids: affectedEventIds.slice(),
    image_added: imageAdded.slice(),
    image_removed: imageRemoved.slice(),
    search_only: searchOnly,
    ...(action ? { action } : {}),
    ...(scope ? { scope } : {}),
    ...(feedbackRevision !== null ? { feedback_revision: feedbackRevision } : {}),
  };
};

const normalizeSnapshotOptions = (options, field) => {
  if (!options || typeof options !== 'object' || Array.isArray(options)) {
    throw new Error(`${field} options must be an object`);
  }

  let intent = null;
  if (options.intent !== undefined && options.intent !== null) {
    if (typeof options.intent !== 'object' || Array.isArray(options.intent)) {
      throw new Error(`${field}.intent must be an object`);
    }
    intent = JSON.parse(JSON.stringify(options.intent));
  }

  let events = null;
  if (Array.isArray(options.events) && !intent) {
    events = normalizeEvents(options.events, `${field}.events`);
  } else if (!intent && !Array.isArray(options.events)) {
    throw new Error(`${field} must contain either intent or events array`);
  }

  const rawMeta = options.operation_metadata ?? options.operationMetadata;
  const operationMetadata = rawMeta ? buildOperationMetadata(rawMeta) : null;

  return {
    intent,
    events,
    latency: normalizeLatency(options.latency, `${field}.latency`),
    warnings: options.warnings === undefined
      ? []
      : normalizeEvents(options.warnings, `${field}.warnings`),
    ...(operationMetadata ? { operation_metadata: operationMetadata } : {}),
  };
};

export const buildKisSnapshot = (results, options) => {
  if (!Array.isArray(results)) throw new Error('KIS results must be an array');
  const {
    intent,
    events,
    latency,
    warnings,
    operation_metadata: operationMetadata,
  } = normalizeSnapshotOptions(options, 'KIS snapshot');
  return {
    ...(intent ? { intent } : {}),
    ...(events ? { events } : {}),
    ...(operationMetadata ? { operation_metadata: operationMetadata } : {}),
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

export const getSnapshotKind = (resultSnapshot) => {
  if (!resultSnapshot || typeof resultSnapshot !== 'object') {
    throw new Error('resultSnapshot must be an object');
  }
  const hasResults = Array.isArray(resultSnapshot.results);
  const hasPaths = Array.isArray(resultSnapshot.paths);
  if (hasPaths && !hasResults) return 'unsupported';
  if (hasResults === hasPaths) {
    throw new Error('resultSnapshot must contain exactly one of results or paths');
  }
  return 'kis';
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
});

export const activityStateForFrame = (frameId, frameActivity) => {
  requireText(frameId, 'frameId');
  const normalized = normalizeFrameActivity(frameActivity);
  if (normalized.viewedFrameIds.has(frameId)) return 'viewed';
  return 'neutral';
};

export const withViewedFrame = (frameActivity, frameId) => {
  requireText(frameId, 'frameId');
  const normalized = normalizeFrameActivity(frameActivity);
  normalized.viewedFrameIds.add(frameId);
  return normalized;
};
