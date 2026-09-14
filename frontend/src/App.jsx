/** Application shell composed from modular feature components. */
import React, { useCallback, useRef, useState } from 'react';
import { AppHeader } from './features/header';
import { ImageModal } from './features/frames';
import { SearchWorkspace } from './features/search';
import { WorkspacePage } from './features/workspace';
import { DatabasePage } from './features/database';
import { useHealthCheck } from './features/health';
import { useVimMode, TopKPromptModal, VimHelpModal } from './features/vim';
import { ApiDocsModal } from './features/docs';
import { useTemporalExploration } from './features/alignment/hooks/useTemporalExploration';
import { VbsSessionProvider, useVbsSession } from './features/vbs/contexts/VbsSessionContext';
import AnswerWorkspaceProvider, {
  useAnswerWorkspace,
} from './features/answer-workspace/contexts/AnswerWorkspaceContext';
import AnswerCandidateDialog from './features/answer-workspace/components/AnswerCandidateDialog';

const explorationSelectionKey = (selection) => {
  const query = selection?.explorationSnapshot?.query;
  const videoId = selection?.frame?.video_id;
  return query && videoId ? `${query}\u0000${videoId}` : null;
};

const AppShell = ({ connectedUserId, draftUserId }) => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [activeQuery, setActiveQuery] = useState('');
  const [activePage, setActivePage] = useState('query');
  const [modalQuery, setModalQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const [replayRequest, setReplayRequest] = useState(null);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const [candidateDraft, setCandidateDraft] = useState(null);
  const [candidateDialogError, setCandidateDialogError] = useState('');
  const [isSavingCandidate, setIsSavingCandidate] = useState(false);
  const replayTokenRef = useRef(0);
  const explorationKeyRef = useRef(null);
  const userIdInputRef = useRef(null);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData } = useHealthCheck();
  const exploration = useTemporalExploration();
  const { close: closeTemporalExploration } = exploration;
  const answerWorkspace = useAnswerWorkspace();
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



  const handleAddAnswerCandidate = (initialValue) => {
    if (!connectedUserId) return;
    setCandidateDialogError('');
    setCandidateDraft({ kind: 'FRAME', initialValue });
  };

  const handleSaveAnswerCandidate = async (value) => {
    if (!connectedUserId || isSavingCandidate) return;
    setIsSavingCandidate(true);
    setCandidateDialogError('');
    try {
      const nextWorkspace = value.kind === 'FRAME'
        ? await answerWorkspace.addFrame({ videoId: value.videoId, timestampMs: value.timestampMs })
        : await answerWorkspace.addText({ text: value.text });
      const candidate = nextWorkspace?.candidates?.find((item) => item.kind === value.kind && (
        value.kind === 'FRAME'
          ? item.video_id === value.videoId && item.timestamp_ms === value.timestampMs
          : item.text === value.text
      ));
      if (candidate?.candidate_id) {
        window.dispatchEvent(new CustomEvent('answer-workspace-focus-candidate', {
          detail: { candidateId: candidate.candidate_id },
        }));
      }
      setCandidateDraft(null);
    } catch (error) {
      setCandidateDialogError(error.message || 'Could not add the answer to the shared workspace.');
    } finally {
      setIsSavingCandidate(false);
    }
  };

  const handleManualVideo = ({ frame, requestedTimestampMs }) => {
    closeExploration();
    setSelectedFrame({
      frame,
      initialTimestampMs: requestedTimestampMs,
    });
    setModalQuery('');
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

      <main className="app-container adhoc-app">
        <div className="workspace-panel" hidden={activePage !== 'query'}>
          <SearchWorkspace
            isActive={activePage === 'query'}
            onAddCandidate={connectedUserId ? handleAddAnswerCandidate : undefined}
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
            onOpenManualVideo={handleManualVideo}
          />
        </div>
        <div className="workspace-panel" hidden={activePage !== 'database'}>
          <DatabasePage isActive={activePage === 'database'} />
        </div>
      </main>

      {selectedFrame && (
        <ImageModal
          frame={selectedFrame.frame}
          query={modalQuery}
          initialTimestampMs={selectedFrame.initialTimestampMs}
          workspaceAction={connectedUserId ? 'add-candidate' : undefined}
          onAddCandidate={handleAddAnswerCandidate}
          onClose={() => setSelectedFrame(null)}
          exploration={selectedFrame.explorationSnapshot ? {
            events: selectedFrame.explorationSnapshot.events,
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
      {candidateDraft && (
        <AnswerCandidateDialog
          kind={candidateDraft.kind}
          initialValue={candidateDraft.initialValue}
          onSave={handleSaveAnswerCandidate}
          onCancel={() => {
            if (isSavingCandidate) return;
            setCandidateDraft(null);
            setCandidateDialogError('');
          }}
          errorMessage={candidateDialogError}
          isSaving={isSavingCandidate}
        />
      )}
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
  const { connectedUserId, draftUserId } = useVbsSession();
  return (
    <AnswerWorkspaceProvider connectedUserId={connectedUserId}>
      <AppShell connectedUserId={connectedUserId} draftUserId={draftUserId} />
    </AnswerWorkspaceProvider>
  );
};

const App = () => <VbsSessionProvider><AppContent /></VbsSessionProvider>;

export default App;
