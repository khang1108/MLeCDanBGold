/** Application shell composed from modular feature components. */
import React, { useCallback, useRef, useState } from 'react';
import { AppHeader } from './features/header';
import { ImageModal } from './features/frames';
import { SearchWorkspace } from './features/search';
import { WorkspacePage } from './features/workspace';
import { useHealthCheck } from './features/health';
import { useVimMode, TopKPromptModal, VimHelpModal } from './features/vim';
import { ApiDocsModal } from './features/docs';
import { useTemporalExploration } from './features/alignment/hooks/useTemporalExploration';
import { VbsSessionProvider, useVbsSession } from './features/vbs/contexts/VbsSessionContext';
import { SubmissionDialog, useDirectSubmission } from './features/submission';

const explorationSelectionKey = (selection) => {
  const seed = selection?.explorationSnapshot;
  const videoId = selection?.frame?.video_id;
  return seed && videoId ? JSON.stringify([seed, videoId]) : null;
};

const AppShell = ({ connectedUserId, draftUserId, invalidateSession }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [activeQuery, setActiveQuery] = useState('');
  const [activePage, setActivePage] = useState('query');
  const [modalQuery, setModalQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const [replayRequest, setReplayRequest] = useState(null);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const replayTokenRef = useRef(0);
  const explorationKeyRef = useRef(null);
  const userIdInputRef = useRef(null);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData } = useHealthCheck();
  const exploration = useTemporalExploration();
  const { close: closeTemporalExploration } = exploration;
  const submission = useDirectSubmission({
    userId: connectedUserId,
    onSessionRejected: invalidateSession,
  });
  const vim = useVimMode({
    onCloseAllModals: () => setSelectedFrame(null),
    queryInputRef,
    enableTopK: activePage === 'query',
  });

  const closeExploration = useCallback(async () => {
    explorationKeyRef.current = null;
    await closeTemporalExploration({ suppressError: true });
  }, [closeTemporalExploration]);

  const handleQueryFrameClick = (selection) => {
    const nextKey = explorationSelectionKey(selection);
    if (!nextKey || (explorationKeyRef.current && explorationKeyRef.current !== nextKey)) {
      closeExploration();
    }
    setSelectedFrame(selection);
    setModalQuery(activeQuery);
  };





  const handleReplay = (historyItem) => {
    closeExploration();
    replayTokenRef.current += 1;
    setReplayRequest({ item: historyItem, token: replayTokenRef.current });
    setActivePage('query');
  };

  return (
    <div className="app-wrapper">
      <AppHeader
        isHealthy={isHealthy}
        healthData={healthData}
        vimMode={vim.mode}
        onToggleVimMode={() => (vim.mode === 'NORMAL' ? vim.enterInsertMode() : vim.enterNormalMode())}
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
            onFocusQueryInput={() => vim.setMode('INSERT')}
            onBlurQueryInput={() => vim.setMode('NORMAL')}
            onHistoryRefresh={() => setHistoryRefreshToken((token) => token + 1)}
            replayRequest={replayRequest}
            onExplorationInvalidated={closeExploration}
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
          frame={selectedFrame.frame}
          query={modalQuery}
          initialTimestampMs={selectedFrame.initialTimestampMs}
          onOpenSubmission={connectedUserId ? submission.open : undefined}
          isSubmissionOpening={submission.opening}
          onClose={() => setSelectedFrame(null)}
          exploration={selectedFrame.explorationSnapshot ? {
            events: selectedFrame.explorationSnapshot.events.map(
              (event) => event.canonical_text,
            ),
            session: exploration.session,
            pending: exploration.pending,
            error: exploration.error,
            unsynced: exploration.unsynced,
            open: (durationSeconds) => {
              explorationKeyRef.current = explorationSelectionKey(selectedFrame);
              return exploration.open({
                snapshot: selectedFrame.explorationSnapshot,
                videoId: selectedFrame.frame.video_id,
                durationSeconds,
              });
            },
            act: exploration.act,
            undo: exploration.undo,
            refresh: exploration.refresh,
            onBack: async () => {
              setSelectedFrame(null);
              await closeExploration();
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
      <TopKPromptModal
        isOpen={vim.isTopKOpen && activePage === 'query'}
        currentTopK={topK}
        onSave={setTopK}
        onClose={() => vim.setIsTopKOpen(false)}
      />
      <VimHelpModal isOpen={vim.isHelpOpen} onClose={() => vim.setIsHelpOpen(false)} />
      <ApiDocsModal isOpen={isDocsOpen} onClose={() => setIsDocsOpen(false)} />
    </div>
  );
};

const AppContent = () => {
  const { connectedUserId, draftUserId, invalidateSession } = useVbsSession();
  return <AppShell
    connectedUserId={connectedUserId}
    draftUserId={draftUserId}
    invalidateSession={invalidateSession}
  />;
};

const App = () => <VbsSessionProvider><AppContent /></VbsSessionProvider>;

export default App;
