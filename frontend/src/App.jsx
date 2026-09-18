/** Application shell composed from modular feature components. */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AppHeader } from './features/header';
import { ImageModal } from './features/frames';
import { SearchWorkspace } from './features/search';
import { useHealthCheck } from './features/health';
import { ApiDocsModal } from './features/docs';
import { useEventTrail } from './features/event-trail';
import { VbsSessionProvider, useVbsSession } from './features/vbs/contexts/VbsSessionContext';
import { taskFamily } from './features/vbs/taskFamily';
import { SubmissionDialog, useDirectSubmission } from './features/submission';

const eventTrailSelectionKey = (selection) => {
  const c = selection?.eventTrailContext;
  return c ? JSON.stringify([c.snapshotId, c.resultId, c.kisRevision]) : null;
};

const AppShell = ({
  connectedUserId,
  draftUserId,
  invalidateSession,
  selectedTask,
  evaluations,
  setSelectedTask,
}) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [activeQuery, setActiveQuery] = useState('');
  const [modalQuery, setModalQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const [eventTrailAnnotations, setEventTrailAnnotations] = useState({});
  const eventTrailKeyRef = useRef(null);
  const userIdInputRef = useRef(null);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData } = useHealthCheck();
  const eventTrail = useEventTrail();
  const { close: closeEventTrailSession } = eventTrail;
  const submission = useDirectSubmission({
    userId: connectedUserId,
    selectedTask,
    onSessionRejected: invalidateSession,
  });

  const [workspaceOverride, setWorkspaceOverride] = useState(null);
  const activeTaskFamily = workspaceOverride || taskFamily(selectedTask);
  const isAvsTask = activeTaskFamily === 'AVS';

  const handleToggleWorkspace = useCallback((mode) => {
    setWorkspaceOverride(mode);
  }, []);

  const handleSetSelectedTask = useCallback((task) => {
    setWorkspaceOverride(null);
    setSelectedTask?.(task);
  }, [setSelectedTask]);

  const effectiveSelectedTask = selectedTask || (isAvsTask ? {
    evaluationId: 'local',
    evaluationName: 'Local Workspace',
    taskName: 'AVS Ad-Hoc Search',
    taskGroup: 'AVS',
    taskType: 'AVS',
    duration: 300,
  } : null);

  useEffect(() => {
    const session = eventTrail.session;
    if (!session) return;
    const key = selectedFrame?.eventTrailContext
      ? `${selectedFrame.eventTrailContext.snapshotId}:${session.result_id}`
      : null;
    if (!key) return;

    if (session.status === 'exhausted') {
      setEventTrailAnnotations((prev) => (prev[key] === 'exhausted' ? prev : { ...prev, [key]: 'exhausted' }));
    } else if (session.status === 'active') {
      setEventTrailAnnotations((prev) => (prev[key] === 'explored' ? prev : { ...prev, [key]: 'explored' }));
    }
  }, [eventTrail.session, selectedFrame]);

  const handleEventTrailInvalidated = useCallback(async () => {
    eventTrailKeyRef.current = null;
    setEventTrailAnnotations({});
    await closeEventTrailSession({ suppressError: true });
  }, [closeEventTrailSession]);

  const handleQueryFrameClick = (selection) => {
    const nextKey = eventTrailSelectionKey(selection);
    if (!nextKey || (eventTrailKeyRef.current && eventTrailKeyRef.current !== nextKey)) {
      eventTrailKeyRef.current = null;
      closeEventTrailSession({ suppressError: true });
    }
    if (nextKey) {
      eventTrailKeyRef.current = nextKey;
    }
    setSelectedFrame(selection);
    const frameObj = selection?.frame || selection;
    const targetQuery = selection?.query || frameObj?.eventText || frameObj?.event_text;
    setModalQuery(targetQuery || activeQuery);
  };

  useEffect(() => {
    const handleGlobalKeyDown = (event) => {
      // Ctrl + I -> Focus User ID input in header
      if (
        event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        (event.key.toLowerCase() === 'i' || event.code === 'KeyI')
      ) {
        event.preventDefault();
        event.stopPropagation();
        if (userIdInputRef.current) {
          userIdInputRef.current.focus();
          userIdInputRef.current.select?.();
        }
        return;
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown, true);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown, true);
  }, []);

  const workspaceEventTrail = useMemo(() => ({
    ...eventTrail,
    open: (ctx) => {
      if (ctx?.snapshotId && ctx?.resultId) {
        eventTrailKeyRef.current = eventTrailSelectionKey({ eventTrailContext: ctx });
      }
      return eventTrail.open(ctx);
    },
  }), [eventTrail]);

  return (
    <div className="app-wrapper">
      <AppHeader
        isHealthy={isHealthy}
        healthData={healthData}
        onOpenDocs={() => setIsDocsOpen(true)}
        userIdInputRef={userIdInputRef}
        isAvsTask={isAvsTask}
        onToggleWorkspace={handleToggleWorkspace}
      />
      {!connectedUserId && (
        <p className="submission-connect-hint" role="status">
          Connect a VBS participant before submitting answers.
        </p>
      )}

      <main className="app-container adhoc-app">
        <div className="workspace-panel">
          <SearchWorkspace
            isActive={true}
            onOpenSubmission={connectedUserId ? submission.open : undefined}
            isSubmissionOpening={submission.opening}
            userId={connectedUserId}
            topK={topK}
            setTopK={setTopK}
            onFrameClick={handleQueryFrameClick}
            onQueryChange={setActiveQuery}
            queryInputRef={queryInputRef}
            onEventTrailInvalidated={handleEventTrailInvalidated}
            eventTrailAnnotations={eventTrailAnnotations}
            eventTrail={workspaceEventTrail}
            workspaceMode={isAvsTask ? 'AVS' : 'KIS'}
            onToggleMode={handleToggleWorkspace}
            selectedTask={effectiveSelectedTask}
            setSelectedTask={handleSetSelectedTask}
            evaluations={evaluations}
            onSessionRejected={invalidateSession}
          />
        </div>
      </main>

      {selectedFrame && (
        <ImageModal
          frame={selectedFrame.frame || selectedFrame}
          events={
            selectedFrame.events
            || selectedFrame.frame?.events
            || selectedFrame.eventTrailContext?.events?.map(
              (event) => (typeof event === 'string' ? event : event?.text || event?.canonical_text || ''),
            )
            || []
          }
          query={modalQuery}
          initialTimestampMs={selectedFrame.initialTimestampMs}
          onOpenSubmission={!isAvsTask && connectedUserId ? submission.open : undefined}
          isSubmissionOpening={submission.opening}
          onClose={() => setSelectedFrame(null)}
          eventTrail={selectedFrame.eventTrailContext ? {
            context: selectedFrame.eventTrailContext,
            state: eventTrail.session,
            pending: eventTrail.pending,
            error: eventTrail.error,
            open: (ctx) => {
              eventTrailKeyRef.current = eventTrailSelectionKey(selectedFrame);
              return eventTrail.open(ctx || selectedFrame.eventTrailContext);
            },
            act: eventTrail.act,
            undo: eventTrail.undo,
            refresh: eventTrail.refresh,
            close: async (options) => {
              eventTrailKeyRef.current = null;
              await eventTrail.close(options);
            },
            back: async () => {
              setSelectedFrame(null);
              eventTrailKeyRef.current = null;
              await eventTrail.close({ suppressError: true });
            },
          } : undefined}
        />
      )}
      {submission.dialog && (
        <SubmissionDialog
          task={submission.dialog.task}
          initialValue={submission.dialog.value}
          outcome={submission.dialog.outcome}
          errorMessage={submission.dialog.error}
          isSubmitting={submission.dialog.submitting}
          onChange={submission.updateValue}
          onSubmit={submission.submit}
          onClose={submission.close}
        />
      )}
      {submission.openError && <div className="submission-open-error" role="alert">{submission.openError}</div>}
      <ApiDocsModal isOpen={isDocsOpen} onClose={() => setIsDocsOpen(false)} />
    </div>
  );
};

const AppContent = () => {
  const {
    connectedUserId,
    draftUserId,
    invalidateSession,
    selectedTask,
    evaluations,
    setSelectedTask,
  } = useVbsSession();
  return (
    <AppShell
      connectedUserId={connectedUserId}
      draftUserId={draftUserId}
      invalidateSession={invalidateSession}
      selectedTask={selectedTask}
      evaluations={evaluations}
      setSelectedTask={setSelectedTask}
    />
  );
};

const App = () => <VbsSessionProvider><AppContent /></VbsSessionProvider>;

export default App;
