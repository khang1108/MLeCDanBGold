/** Application shell composed from modular feature components. */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AppHeader } from './features/header';
import { ImageModal } from './features/frames';
import { SearchWorkspace } from './features/search';
import { WorkspacePage } from './features/workspace';
import { useHealthCheck } from './features/health';
import { ApiDocsModal } from './features/docs';
import { useEventTrail } from './features/event-trail';
import { VbsSessionProvider, useVbsSession } from './features/vbs/contexts/VbsSessionContext';
import { SubmissionDialog, useDirectSubmission } from './features/submission';

const eventTrailSelectionKey = (selection) => {
  const c = selection?.eventTrailContext;
  return c ? JSON.stringify([c.snapshotId, c.resultId, c.kisRevision]) : null;
};

const AppShell = ({ connectedUserId, draftUserId, invalidateSession, selectedTask }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [activeQuery, setActiveQuery] = useState('');
  const [activePage, setActivePage] = useState('query');
  const [modalQuery, setModalQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const [replayRequest, setReplayRequest] = useState(null);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const [eventTrailAnnotations, setEventTrailAnnotations] = useState({});
  const replayTokenRef = useRef(0);
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
    setModalQuery(activeQuery);
  };

  const handleReplay = (historyItem) => {
    handleEventTrailInvalidated();
    replayTokenRef.current += 1;
    setReplayRequest({ item: historyItem, token: replayTokenRef.current });
    setActivePage('query');
  };

  useEffect(() => {
    const handleGlobalKeyDown = (event) => {
      // Windows 1 / Meta + 1 -> Switch to Query tab
      if (
        event.metaKey &&
        !event.ctrlKey &&
        !event.altKey &&
        (event.key === '1' || event.code === 'Digit1')
      ) {
        event.preventDefault();
        event.stopPropagation();
        setActivePage('query');
        return;
      }

      // Windows 2 / Meta + 2 -> Switch to Workspace tab
      if (
        event.metaKey &&
        !event.ctrlKey &&
        !event.altKey &&
        (event.key === '2' || event.code === 'Digit2')
      ) {
        event.preventDefault();
        event.stopPropagation();
        setActivePage('workspace');
        return;
      }

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
  }, [setActivePage]);

  return (
    <div className="app-wrapper">
      <AppHeader
        isHealthy={isHealthy}
        healthData={healthData}
        onOpenDocs={() => setIsDocsOpen(true)}
        userIdInputRef={userIdInputRef}
        activePage={activePage}
        onSelectPage={setActivePage}
      />
      {!connectedUserId && (
        <p className="submission-connect-hint" role="status">
          Connect a VBS participant before submitting answers.
        </p>
      )}

      <main className="app-container adhoc-app">
        <div className="workspace-panel" hidden={activePage !== 'query'}>
          <SearchWorkspace
            isActive={activePage === 'query'}
            onOpenSubmission={connectedUserId ? submission.open : undefined}
            isSubmissionOpening={submission.opening}
            userId={connectedUserId}
            historyUserId={draftUserId}
            topK={topK}
            setTopK={setTopK}
            onFrameClick={handleQueryFrameClick}
            onQueryChange={setActiveQuery}
            queryInputRef={queryInputRef}
            onHistoryRefresh={() => setHistoryRefreshToken((token) => token + 1)}
            replayRequest={replayRequest}
            onEventTrailInvalidated={handleEventTrailInvalidated}
            eventTrailAnnotations={eventTrailAnnotations}
          />
        </div>
        <div className="workspace-panel" hidden={activePage !== 'workspace'}>
          <WorkspacePage
            isActive={activePage === 'workspace'}
            userId={draftUserId}
            historyRefreshToken={historyRefreshToken}
            onReplay={handleReplay}
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
          onOpenSubmission={connectedUserId ? submission.open : undefined}
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
  const { connectedUserId, draftUserId, invalidateSession, selectedTask } = useVbsSession();
  return <AppShell
    connectedUserId={connectedUserId}
    draftUserId={draftUserId}
    invalidateSession={invalidateSession}
    selectedTask={selectedTask}
  />;
};

const App = () => <VbsSessionProvider><AppContent /></VbsSessionProvider>;

export default App;
