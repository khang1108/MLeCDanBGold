/** Format and parse the single editable answer line used by direct submission. */

const isSafeTimestamp = (value) => Number.isSafeInteger(value) && value >= 0;

const parseTimestamp = (value) => {
  const digits = value.trim();
  if (!/^\d+$/.test(digits)) return null;
  const timestamp = Number(digits);
  return isSafeTimestamp(timestamp) ? timestamp : null;
};

/** Format a canonical video and millisecond range as one comma-separated line. */
export const formatTemporalAnswer = ({ videoId, startMs, endMs } = {}) => {
  const normalizedVideoId = typeof videoId === 'string' ? videoId.trim() : '';
  if (!normalizedVideoId || /[,\r\n]/.test(normalizedVideoId)) {
    throw new Error('A temporal answer requires a single-line video ID');
  }
  if (!isSafeTimestamp(startMs) || !isSafeTimestamp(endMs) || endMs < startMs) {
    throw new Error('A temporal answer requires a valid non-negative millisecond range');
  }
  return `${normalizedVideoId},${startMs},${endMs}`;
};

/** Parse one editable line into a strict API answer, treating malformed CSV as text. */
export const parseAnswerLine = (value) => {
  if (typeof value !== 'string' || !value.trim() || /[\r\n]/.test(value)) {
    throw new Error('Enter one non-blank answer line');
  }

  const text = value.trim();
  const fields = text.split(',');
  if (fields.length === 3) {
    const videoId = fields[0].trim();
    const startMs = parseTimestamp(fields[1]);
    const endMs = parseTimestamp(fields[2]);
    if (videoId && startMs !== null && endMs !== null && endMs >= startMs) {
      return {
        kind: 'TEMPORAL',
        video_id: videoId,
        start_ms: startMs,
        end_ms: endMs,
      };
    }
  }

  return { kind: 'TEXT', text };
};
