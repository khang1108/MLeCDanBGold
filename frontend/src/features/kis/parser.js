/**
 * Frontend deterministic KIS composer command parser.
 *
 * Provides immediate structural preview and validation feedback mirroring the backend grammar.
 * This parser produces preview classifications but does not replace authoritative server-side resolution.
 */

const MALFORMED_PREFIX_REGEX = /^E(?=[0-9+\-: \t])[^\r\n:]*:/m;

/**
 * Extract event patches from an explicit E#: block.
 *
 * @param {string} text - Composer input text
 * @returns {Array<{ event_id: string, instruction: string }>}
 */
export const extractEventPatches = (text) => {
  const normalized = (text || '').trim();
  const headerRegex = /^E([1-9]\d*):[ \t]*(.*)$/gm;
  const headers = [];
  let match;
  while ((match = headerRegex.exec(normalized)) !== null) {
    headers.push({
      index: match.index,
      fullMatch: match[0],
      eventId: `E${match[1]}`,
      afterColon: match[2],
    });
  }

  const patches = [];
  for (let i = 0; i < headers.length; i += 1) {
    const h = headers[i];
    const nextStart = i + 1 < headers.length ? headers[i + 1].index : normalized.length;
    const instruction = (
      h.afterColon + normalized.slice(h.index + h.fullMatch.length, nextStart)
    ).trim();
    patches.push({ event_id: h.eventId, instruction });
  }
  return patches;
};

/**
 * Extract instruction from an /llm-rewrite command block.
 *
 * @param {string} text - Composer input text
 * @returns {string}
 */
export const extractGlobalRewriteInstruction = (text) => {
  const lines = (text || '').trim().split('\n');
  return lines.slice(1).join('\n').trim();
};

/**
 * Parse and validate a KIS composer draft against base intent.
 *
 * @param {string} draft - Raw composer text
 * @param {object|null} baseIntent - Current intent or null
 * @returns {{
 *   kind: 'initial_resolve' | 'patch_events' | 'global_rewrite' | 'invalid',
 *   affectedEventIds: string[],
 *   error: string | null,
 * }}
 */
export const parseComposerDraft = (draft, baseIntent = null) => {
  const normalized = (draft || '').trim();
  if (!normalized) {
    return {
      kind: 'invalid',
      affectedEventIds: [],
      error: 'KIS command must contain non-empty text',
    };
  }

  // Global rewrite command: /llm-rewrite\n<instruction>
  if (normalized.startsWith('/llm-rewrite')) {
    const lines = normalized.split('\n');
    if (lines[0].trim() !== '/llm-rewrite') {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: 'Malformed /llm-rewrite command',
      };
    }
    const instruction = lines.slice(1).join('\n').trim();
    if (!instruction) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: '/llm-rewrite requires a non-empty instruction',
      };
    }
    const affectedEventIds = Array.isArray(baseIntent?.events)
      ? baseIntent.events.map((e) => e.id)
      : [];
    return {
      kind: 'global_rewrite',
      affectedEventIds,
      error: null,
    };
  }

  // Header matching for E1..Ek:
  const headerRegex = /^E([1-9]\d*):[ \t]*(.*)$/gm;
  const headers = [];
  let match;
  while ((match = headerRegex.exec(normalized)) !== null) {
    headers.push({
      index: match.index,
      fullMatch: match[0],
      eventNum: parseInt(match[1], 10),
      eventId: `E${match[1]}`,
      afterColon: match[2],
    });
  }

  const baseEventCount = Array.isArray(baseIntent?.events) ? baseIntent.events.length : 0;

  if (headers.length === 0) {
    if (MALFORMED_PREFIX_REGEX.test(normalized)) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: 'Malformed E#: event prefix',
      };
    }
    if (baseEventCount > 0) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: 'Existing KIS intents require explicit E#: event instructions',
      };
    }
    return {
      kind: 'initial_resolve',
      affectedEventIds: [],
      error: null,
    };
  }

  if (headers[0].index !== 0) {
    return {
      kind: 'invalid',
      affectedEventIds: [],
      error: 'Explicit E#: event instructions must begin the command',
    };
  }

  // Check for any malformed event prefixes in the text that did not match clean headers
  const malformedGlobal = /^E(?=[0-9+\-: \t])[^\r\n:]*:/gm;
  const validStarts = new Set(headers.map((h) => h.index));
  let malMatch;
  while ((malMatch = malformedGlobal.exec(normalized)) !== null) {
    if (!validStarts.has(malMatch.index)) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: 'Malformed E#: event prefix',
      };
    }
  }

  // Validate instructions non-empty
  for (let i = 0; i < headers.length; i += 1) {
    const h = headers[i];
    const nextStart = i + 1 < headers.length ? headers[i + 1].index : normalized.length;
    const instruction = (
      h.afterColon + normalized.slice(h.index + h.fullMatch.length, nextStart)
    ).trim();
    if (!instruction) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: `${h.eventId} requires an instruction`,
      };
    }
  }

  // Validate event numbering
  const numbers = headers.map((h) => h.eventNum);
  if (new Set(numbers).size !== numbers.length) {
    return {
      kind: 'invalid',
      affectedEventIds: [],
      error: 'Duplicate event IDs are not allowed',
    };
  }

  for (let i = 1; i < numbers.length; i += 1) {
    if (numbers[i] <= numbers[i - 1]) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: 'Event headers must be in strictly increasing order',
      };
    }
  }

  const newNumbers = numbers.filter((n) => n > baseEventCount);
  if (newNumbers.length > 0) {
    const maxNew = newNumbers[newNumbers.length - 1];
    const expected = [];
    for (let n = baseEventCount + 1; n <= maxNew; n += 1) {
      expected.push(n);
    }
    const newSet = new Set(newNumbers);
    const missing = expected.find((n) => !newSet.has(n));
    if (missing !== undefined) {
      return {
        kind: 'invalid',
        affectedEventIds: [],
        error: `Missing E${missing} before a new event`,
      };
    }
  }

  const affectedEventIds = headers.map((h) => h.eventId);
  if (baseEventCount === 0) {
    return {
      kind: 'initial_resolve',
      affectedEventIds,
      error: null,
    };
  }

  return {
    kind: 'patch_events',
    affectedEventIds,
    error: null,
  };
};
