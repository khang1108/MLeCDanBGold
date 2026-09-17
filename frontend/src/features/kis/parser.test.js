import { parseComposerDraft } from './parser';

const BASE_INTENT = {
  revision: 3,
  query_text: 'woman enters, talks, then leaves',
  entities: [],
  events: [
    { id: 'E1', text: 'woman enters', images: [], bindings: [] },
    { id: 'E2', text: 'woman talks', images: [], bindings: [] },
    { id: 'E3', text: 'woman leaves', images: [], bindings: [] },
  ],
  temporal_edges: [
    { source: 'E1', target: 'E2', relation: 'before' },
    { source: 'E2', target: 'E3', relation: 'before' },
  ],
};

describe('parseComposerDraft', () => {
  test('previews a scoped batch', () => {
    expect(parseComposerDraft('E2: chef wears black\nE4: takes a plate', BASE_INTENT)).toEqual({
      kind: 'patch_events',
      affectedEventIds: ['E2', 'E4'],
      error: null,
    });
  });

  test('reports an illegal event gap before submit', () => {
    expect(parseComposerDraft('E5: new event', BASE_INTENT).error).toMatch(/missing E4/i);
  });

  test('previews /llm-rewrite command targeting all events', () => {
    expect(parseComposerDraft('/llm-rewrite\nrewrite everything to focus on chef', BASE_INTENT)).toEqual({
      kind: 'global_rewrite',
      affectedEventIds: ['E1', 'E2', 'E3'],
      error: null,
    });
  });

  test('reports error if /llm-rewrite has no instruction', () => {
    const result = parseComposerDraft('/llm-rewrite', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/instruction/i);
  });

  test('reports error if /llm-rewrite is malformed', () => {
    const result = parseComposerDraft('/llm-rewrite-now\nfoo', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/malformed/i);
  });

  test('previews initial natural text when no base intent exists', () => {
    expect(parseComposerDraft('woman enters, talks, then leaves', null)).toEqual({
      kind: 'initial_resolve',
      affectedEventIds: [],
      error: null,
    });
  });

  test('previews initial explicit E1..Ek mapped to initial_resolve', () => {
    expect(parseComposerDraft('E1: woman enters\nE2: woman talks', null)).toEqual({
      kind: 'initial_resolve',
      affectedEventIds: ['E1', 'E2'],
      error: null,
    });
  });

  test('reports error if initial explicit batch does not start at E1', () => {
    const result = parseComposerDraft('E2: woman talks', null);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/missing E1/i);
  });

  test('reports error on unscoped progressive text when base intent exists', () => {
    const result = parseComposerDraft('unscoped progressive text', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/explicit E#: event instructions/i);
  });

  test('reports error on duplicate event IDs', () => {
    const result = parseComposerDraft('E2: update one\nE2: update two', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/duplicate/i);
  });

  test('reports error on non-increasing event order', () => {
    const result = parseComposerDraft('E3: update three\nE2: update two', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/increasing order/i);
  });

  test('reports error if non-header text precedes explicit instructions', () => {
    const result = parseComposerDraft('some text\nE1: first', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/must begin the command/i);
  });

  test('reports error on malformed event prefix', () => {
    const result = parseComposerDraft('E0: invalid event', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/malformed/i);
  });

  test('reports error on empty draft', () => {
    const result = parseComposerDraft('   ', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/non-empty/i);
  });

  test('reports error on header without instruction', () => {
    const result = parseComposerDraft('E2:   ', BASE_INTENT);
    expect(result.kind).toBe('invalid');
    expect(result.error).toMatch(/instruction/i);
  });
});
