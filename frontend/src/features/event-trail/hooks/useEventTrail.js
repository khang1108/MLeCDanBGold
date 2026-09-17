/**
 * Hook managing an interactive EventTrail exploration session.
 *
 * Coordinates transactional feedback mutations, revision conflicts,
 * expiration states, and branch cancellation cleanly.
 */
import { useCallback, useRef, useState } from 'react';
import {
  openEventTrail,
  getEventTrail,
  actOnEventTrail,
  closeEventTrail,
} from '../../../api/eventTrail';

export const computeSessionKey = (context) => {
  if (
    !context
    || !context.snapshotId
    || !context.resultId
    || !Number.isInteger(context.kisRevision)
  ) {
    return null;
  }
  return JSON.stringify([context.snapshotId, context.resultId, context.kisRevision]);
};

export const useEventTrail = () => {
  const [session, setSession] = useState(null);
  const [sessionKey, setSessionKey] = useState(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const [unsynced, setUnsynced] = useState(false);

  const generationRef = useRef(0);
  const sessionRef = useRef(null);
  const keyRef = useRef(null);
  const pendingRef = useRef(false);
  const controllerRef = useRef(null);

  const setCurrentSession = useCallback((nextSession, key = null) => {
    sessionRef.current = nextSession;
    keyRef.current = nextSession ? key : null;
    setSession(nextSession);
    setSessionKey(nextSession ? key : null);
  }, []);

  const invalidate = useCallback(() => {
    generationRef.current += 1;
    controllerRef.current?.abort();
    controllerRef.current = null;
    pendingRef.current = false;
    setPending(false);
    setUnsynced(false);
    setError(null);
  }, []);

  const cleanupLateSession = useCallback((lateSession) => {
    if (lateSession?.session_id && Number.isInteger(lateSession?.trail_revision)) {
      closeEventTrail(lateSession.session_id, {
        expectedTrailRevision: lateSession.trail_revision,
      }).catch(() => undefined);
    }
  }, []);

  const open = useCallback(
    async (context) => {
      const key = computeSessionKey(context);
      if (!key) {
        setError('EventTrail requires a valid snapshot, result, and KIS revision.');
        return null;
      }

      if (sessionRef.current && keyRef.current === key) {
        return sessionRef.current;
      }

      const previous = sessionRef.current;
      invalidate();
      setCurrentSession(null);

      if (previous?.session_id && Number.isInteger(previous?.trail_revision)) {
        closeEventTrail(previous.session_id, {
          expectedTrailRevision: previous.trail_revision,
        }).catch(() => undefined);
      }

      const generation = generationRef.current;
      const controller = new AbortController();
      controllerRef.current = controller;
      pendingRef.current = true;
      setPending(true);

      try {
        const opened = await openEventTrail(
          {
            snapshotId: context.snapshotId,
            resultId: context.resultId,
            expectedKisRevision: context.kisRevision,
            searchSessionId: context.searchSessionId,
          },
          { signal: controller.signal }
        );

        if (generation !== generationRef.current || controller.signal.aborted) {
          cleanupLateSession(opened);
          return null;
        }

        setCurrentSession(opened, key);
        setError(null);
        return opened;
      } catch (openError) {
        if (controller.signal.aborted) return null;
        if (generation === generationRef.current) {
          if (openError?.code === 'SNAPSHOT_EXPIRED' || openError?.status === 410) {
            setError(
              'EventTrail snapshot expired. Rerun the current search to create a fresh snapshot.'
            );
          } else if (openError?.code === 'SNAPSHOT_NOT_FOUND' || openError?.status === 404) {
            setError(
              'EventTrail snapshot not found (server was restarted or search expired). Please rerun the search.'
            );
          } else {
            setError(openError?.message || 'Failed to open EventTrail');
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
    },
    [invalidate, setCurrentSession, cleanupLateSession]
  );

  const act = useCallback(
    async (action) => {
      const current = sessionRef.current;
      if (!current?.session_id || pendingRef.current) return null;

      const generation = generationRef.current;
      const controller = new AbortController();
      controllerRef.current = controller;
      pendingRef.current = true;
      setPending(true);

      try {
        const updated = await actOnEventTrail(
          current.session_id,
          {
            expectedTrailRevision: current.trail_revision,
            action,
          },
          { signal: controller.signal }
        );

        if (generation !== generationRef.current || controller.signal.aborted) {
          return null;
        }

        setCurrentSession(updated, keyRef.current);
        setError(null);
        return updated;
      } catch (actError) {
        if (controller.signal.aborted) return null;
        if (generation !== generationRef.current) return null;

        if (
          actError?.code === 'TRAIL_REVISION_CONFLICT'
          || (actError?.status === 409 && actError?.code !== 'CONSTRAINT_CONFLICT')
        ) {
          try {
            const refreshed = await getEventTrail(current.session_id, {
              signal: controller.signal,
            });
            if (generation === generationRef.current) {
              setCurrentSession(refreshed, keyRef.current);
              setError(`Conflict detected: ${actError.message || 'revision updated'}`);
            }
          } catch (refreshErr) {
            if (
              refreshErr?.code === 'TRAIL_SESSION_EXPIRED'
              || refreshErr?.status === 410
              || refreshErr?.code === 'TRAIL_SESSION_NOT_FOUND'
              || refreshErr?.status === 404
            ) {
              setCurrentSession(null);
              setError('EventTrail session expired. Reopen it from this result.');
            } else {
              setUnsynced(true);
              setError(refreshErr?.message || 'Failed to refresh conflicting session');
            }
          }
        } else if (actError?.code === 'CONSTRAINT_CONFLICT') {
          setError(actError.message || 'Constraint conflict');
        } else if (actError?.code === 'TRAIL_SESSION_EXPIRED' || actError?.status === 410) {
          setCurrentSession(null);
          setError('EventTrail session expired. Reopen it from this result.');
        } else if (actError?.code === 'TRAIL_SESSION_NOT_FOUND' || actError?.status === 404) {
          setCurrentSession(null);
          setError(null);
        } else {
          setError(actError?.message || 'Action failed');
        }
        return null;
      } finally {
        if (controllerRef.current === controller) controllerRef.current = null;
        if (generation === generationRef.current) {
          pendingRef.current = false;
          setPending(false);
        }
      }
    },
    [setCurrentSession]
  );

  const undo = useCallback(() => act({ type: 'undo' }), [act]);

  const refresh = useCallback(async () => {
    const current = sessionRef.current;
    if (!current?.session_id || pendingRef.current) return null;

    const generation = generationRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    pendingRef.current = true;
    setPending(true);

    try {
      const refreshed = await getEventTrail(current.session_id, {
        signal: controller.signal,
      });
      if (generation !== generationRef.current || controller.signal.aborted) {
        return null;
      }
      setCurrentSession(refreshed, keyRef.current);
      setUnsynced(false);
      setError(null);
      return refreshed;
    } catch (refreshErr) {
      if (controller.signal.aborted) return null;
      if (generation === generationRef.current) {
        if (
          refreshErr?.code === 'TRAIL_SESSION_EXPIRED'
          || refreshErr?.status === 410
        ) {
          setCurrentSession(null);
          setError('EventTrail session expired. Reopen it from this result.');
        } else if (
          refreshErr?.code === 'TRAIL_SESSION_NOT_FOUND'
          || refreshErr?.status === 404
        ) {
          setCurrentSession(null);
          setError(null);
        } else {
          setUnsynced(true);
          setError(refreshErr?.message || 'Refresh failed');
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

  const close = useCallback(
    async ({ suppressError = false } = {}) => {
      const current = sessionRef.current;
      invalidate();
      setCurrentSession(null);
      if (!current?.session_id || !Number.isInteger(current?.trail_revision)) return;

      try {
        await closeEventTrail(current.session_id, {
          expectedTrailRevision: current.trail_revision,
        });
      } catch (closeError) {
        if (
          !suppressError
          && closeError?.status !== 404
          && closeError?.code !== 'TRAIL_SESSION_NOT_FOUND'
        ) {
          setError(`Could not close exploration: ${closeError.message || 'request failed'}`);
        }
      }
    },
    [invalidate, setCurrentSession]
  );

  const clearLocal = useCallback(() => {
    invalidate();
    setCurrentSession(null);
    setError(null);
  }, [invalidate, setCurrentSession]);

  const syncSession = useCallback((externalTrailState, key = null) => {
    setCurrentSession(externalTrailState, key ?? keyRef.current);
  }, [setCurrentSession]);

  return {
    session,
    sessionKey,
    pending,
    error,
    unsynced,
    open,
    act,
    undo,
    refresh,
    close,
    clearLocal,
    syncSession,
  };
};
