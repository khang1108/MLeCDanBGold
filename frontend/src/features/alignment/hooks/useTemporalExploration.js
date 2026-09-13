/**
 * Own one server-backed temporal exploration branch outside the inspector.
 *
 * The hook deliberately keeps server envelopes intact: the panel owns drafts,
 * while this hook serializes mutations and rejects responses from old branches.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  actOnExploration,
  closeExploration,
  getExploration,
  openExploration,
} from '../../../api/exploration';

const isNotFound = (error) => error?.status === 404;
const isTimeout = (error) => error?.name === 'TimeoutError';
const needsReconciliation = (error) => error?.status === 409 || isTimeout(error);
export const EXPLORATION_REQUEST_TIMEOUT_MS = 15_000;

/** Abort a transport after the client-side deadline while preserving late-open cleanup. */
const requestWithTimeout = (request, controller, onLateSuccess) => new Promise((resolve, reject) => {
  let settled = false;
  const timer = setTimeout(() => {
    if (settled) return;
    settled = true;
    controller.abort();
    const error = new Error('Exploration request timed out');
    error.name = 'TimeoutError';
    reject(error);
  }, EXPLORATION_REQUEST_TIMEOUT_MS);

  request.then((value) => {
    if (settled) {
      onLateSuccess?.(value);
      return;
    }
    settled = true;
    clearTimeout(timer);
    resolve(value);
  }, (error) => {
    if (settled) return;
    settled = true;
    clearTimeout(timer);
    reject(error);
  });
});

const validSnapshot = (snapshot) => (
  snapshot
  && typeof snapshot.query === 'string'
  && snapshot.query.trim()
  && Array.isArray(snapshot.events)
  && snapshot.events.length > 0
  && typeof snapshot.use_dense === 'boolean'
  && typeof snapshot.use_bm25 === 'boolean'
  && (snapshot.use_dense || snapshot.use_bm25)
  && (!snapshot.use_dense || (
    Array.isArray(snapshot.dense_events)
    && snapshot.dense_events.length === snapshot.events.length
  ))
  && (!snapshot.use_bm25 || (
    Array.isArray(snapshot.bm25_caption_events)
    && snapshot.bm25_caption_events.length === snapshot.events.length
  ))
);

const sessionKey = (snapshot, videoId) => `${snapshot.query}\u0000${videoId}`;

/** Build the backend contract without inferring events or timestamp coordinates. */
export const buildExplorationOpenBody = ({ snapshot, videoId, durationSeconds }) => {
  const duration = Number(durationSeconds);
  if (!validSnapshot(snapshot) || !videoId || !Number.isFinite(duration) || duration <= 0) return null;

  return {
    query: snapshot.query,
    events: snapshot.events,
    retrieval_events: snapshot.use_dense ? snapshot.dense_events : snapshot.events,
    caption_events: snapshot.use_bm25 ? snapshot.bm25_caption_events : null,
    use_dense: snapshot.use_dense,
    use_bm25: snapshot.use_bm25,
    video_id: videoId,
    window: [0, Math.floor(duration * 1000)],
  };
};

/** Manage one active branch and reconcile uncertain server mutation outcomes. */
export const useTemporalExploration = () => {
  const [session, setSession] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const generationRef = useRef(0);
  const sessionRef = useRef(null);
  const pendingRef = useRef(false);
  const controllerRef = useRef(null);
  const keyRef = useRef(null);
  const unsyncedRef = useRef(false);

  const setCurrentSession = useCallback((nextSession, key = null) => {
    sessionRef.current = nextSession;
    keyRef.current = nextSession ? key : null;
    setSession(nextSession);
  }, []);

  const invalidate = useCallback(() => {
    generationRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    pendingRef.current = false;
    unsyncedRef.current = false;
    setPending(false);
    setError(null);
  }, []);

  const cleanupLateOpen = useCallback((envelope) => {
    const revision = envelope?.view?.revision;
    if (envelope?.handle && Number.isInteger(revision)) {
      closeExploration(envelope.handle, revision).catch(() => undefined);
    }
  }, []);

  const refresh = useCallback(async ({ allowPending = false } = {}) => {
    const current = sessionRef.current;
    if (!current?.handle || (pendingRef.current && !allowPending)) return null;
    const generation = generationRef.current;
    const revision = current.view?.revision;
    const controller = new AbortController();
    controllerRef.current = controller;
    pendingRef.current = true;
    setPending(true);
    try {
      const refreshed = await requestWithTimeout(
        getExploration(current.handle, { signal: controller.signal }),
        controller,
      );
      if (
        generation !== generationRef.current
        || sessionRef.current?.handle !== current.handle
        || sessionRef.current?.view?.revision !== revision
      ) return null;
      setCurrentSession(refreshed, keyRef.current);
      unsyncedRef.current = false;
      setError(null);
      return refreshed;
    } catch (refreshError) {
      if (controller.signal.aborted && !isTimeout(refreshError)) return null;
      if (
        generation === generationRef.current
        && sessionRef.current?.handle === current.handle
        && sessionRef.current?.view?.revision === revision
      ) {
        if (isNotFound(refreshError)) {
          setCurrentSession(null);
          unsyncedRef.current = false;
          setError(null);
        } else {
          unsyncedRef.current = true;
          setError(`Exploration is not synchronized: ${refreshError.message || 'refresh failed'}`);
        }
      }
      return null;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      if (generation === generationRef.current) {
        pendingRef.current = false;
        setPending(false);
      }
    }
  }, [setCurrentSession]);

  const close = useCallback(async ({ suppressError = false } = {}) => {
    const current = sessionRef.current;
    invalidate();
    const generation = generationRef.current;
    setCurrentSession(null);
    if (!current?.handle || !Number.isInteger(current?.view?.revision)) return;
    try {
      await closeExploration(current.handle, current.view.revision);
    } catch (closeError) {
      // A stale/missing close still must not revive a discarded local branch.
      if (!suppressError && !isNotFound(closeError) && generation === generationRef.current) {
        setError(`Could not close exploration: ${closeError.message || 'request failed'}`);
      }
    }
  }, [invalidate, setCurrentSession]);

  const open = useCallback(async ({ snapshot, videoId, durationSeconds }) => {
    const body = buildExplorationOpenBody({ snapshot, videoId, durationSeconds });
    if (!body) {
      setError('Explore requires a live KIS scoring snapshot and a valid video duration.');
      return null;
    }
    const key = sessionKey(snapshot, videoId);
    if (sessionRef.current && keyRef.current === key) return sessionRef.current;

    const previous = sessionRef.current;
    invalidate();
    setCurrentSession(null);
    if (previous?.handle && Number.isInteger(previous?.view?.revision)) {
      closeExploration(previous.handle, previous.view.revision).catch(() => undefined);
    }

    const generation = generationRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    pendingRef.current = true;
    setPending(true);
    try {
      const opened = await requestWithTimeout(
        openExploration(body, { signal: controller.signal }),
        controller,
        cleanupLateOpen,
      );
      if (generation !== generationRef.current || controller.signal.aborted) {
        cleanupLateOpen(opened);
        return null;
      }
      setCurrentSession(opened, key);
      setError(null);
      return opened;
    } catch (openError) {
      if ((!controller.signal.aborted || isTimeout(openError)) && generation === generationRef.current) {
        setError(openError.message || 'Could not open exploration');
      }
      return null;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      if (generation === generationRef.current) {
        pendingRef.current = false;
        setPending(false);
      }
    }
  }, [cleanupLateOpen, invalidate, setCurrentSession]);

  const act = useCallback(async ({ action, event_index: eventIndex, interval } = {}) => {
    const current = sessionRef.current;
    if (!current?.handle || pendingRef.current || unsyncedRef.current) return null;
    const revision = current?.view?.revision;
    const eventVersion = current?.view?.event_version;
    if (!Number.isInteger(revision) || !eventVersion || !current.scoring_revision) return null;

    const generation = generationRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    pendingRef.current = true;
    setPending(true);
    try {
      const next = await requestWithTimeout(actOnExploration(current.handle, {
        expected_revision: revision,
        event_version: eventVersion,
        scoring_revision: current.scoring_revision,
        action,
        ...(eventIndex === undefined ? {} : { event_index: eventIndex }),
        ...(interval === undefined ? {} : { interval }),
      }, { signal: controller.signal }), controller);
      if (
        generation !== generationRef.current
        || sessionRef.current?.handle !== current.handle
        || sessionRef.current?.view?.revision !== revision
      ) return null;
      setCurrentSession(next, keyRef.current);
      setError(null);
      return next;
    } catch (actionError) {
      if (controller.signal.aborted && !isTimeout(actionError)) return null;
      if (
        generation === generationRef.current
        && sessionRef.current?.handle === current.handle
        && sessionRef.current?.view?.revision === revision
      ) {
        if (isNotFound(actionError)) {
          setCurrentSession(null);
          unsyncedRef.current = false;
          setError(null);
        } else if (needsReconciliation(actionError)) {
          setError('Exploration action outcome is uncertain; synchronizing current view.');
          await refresh({ allowPending: true });
        } else {
          setError(actionError.message || 'Could not update exploration');
        }
      }
      return null;
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      if (generation === generationRef.current) {
        pendingRef.current = false;
        setPending(false);
      }
    }
  }, [refresh, setCurrentSession]);

  const undo = useCallback(() => act({ action: 'undo' }), [act]);

  useEffect(() => () => {
    controllerRef.current?.abort();
  }, []);

  return {
    session,
    pending,
    error,
    open,
    act,
    undo,
    close,
    refresh,
  };
};
