/** Application shell composed from modular feature components. */
import React, { useRef, useState } from 'react';
import { AppHeader } from './features/header';
import { ImageModal } from './features/frames';
import { SearchWorkspace, ImageSearchWorkspace } from './features/search';
import { FilterWorkspace } from './features/filter';
import { WorkspacePage } from './features/workspace';
import { DatabasePage } from './features/database';
import { useHealthCheck } from './features/health';
import { useVimMode, TopKPromptModal, VimHelpModal } from './features/vim';
import { ApiDocsModal } from './features/docs';
import { SubmissionProvider, SubmissionDialogProvider, useSubmissionDialog } from './features/submission';

const USER_ID_STORAGE_KEY = 'hcmai_user_id';

const getStoredUserId = () => {
  try {
    return window.localStorage.getItem(USER_ID_STORAGE_KEY) || '';
  } catch {
    return '';
  }
};

const persistUserId = (val) => {
  try {
    if (val) {
      window.localStorage.setItem(USER_ID_STORAGE_KEY, val);
    } else {
      window.localStorage.removeItem(USER_ID_STORAGE_KEY);
    }
  } catch {
    // Ignore storage errors in restricted contexts
  }
};

const AppContent = () => {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [activeQuery, setActiveQuery] = useState('');
  const [activePage, setActivePage] = useState('query');
  const [modalQuery, setModalQuery] = useState('');
  const [userId, setUserId] = useState(getStoredUserId);
  const [userIdError, setUserIdError] = useState(null);
  const [topK, setTopK] = useState(20);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const [replayRequest, setReplayRequest] = useState(null);
  const [historyRefreshToken, setHistoryRefreshToken] = useState(0);
  const replayTokenRef = useRef(0);
  const userIdInputRef = useRef(null);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData } = useHealthCheck();
  const { requestSubmission } = useSubmissionDialog();
  const vim = useVimMode({
    onCloseAllModals: () => setSelectedFrame(null),
    queryInputRef,
    enableTopK: activePage === 'query' || activePage === 'image-search',
  });

  const handleQueryFrameClick = (selection) => {
    setSelectedFrame(selection);
    setModalQuery(activeQuery);
  };

  const handleFilterFrameClick = (frame) => {
    setSelectedFrame({ frame, submissionMode: 'kis' });
    setModalQuery('');
  };

  const handleManualVideo = ({ frame, requestedTimestampMs }) => {
    setSelectedFrame({
      frame,
      initialTimestampMs: requestedTimestampMs,
      submissionMode: 'none',
    });
    setModalQuery('');
  };

  const handleReplay = (historyItem) => {
    replayTokenRef.current += 1;
    setReplayRequest({ item: historyItem, token: replayTokenRef.current });
    setActivePage('query');
  };

  const handleFocusUserId = () => {
    setUserIdError('A User ID is required before searching.');
    userIdInputRef.current?.focus();
  };

  const handleInspectorSubmit = (intent) => {
    requestSubmission({
      ...intent,
      history: selectedFrame?.history || (selectedFrame?.frame?.frame_id ? { frameIds: [selectedFrame.frame.frame_id] } : undefined),
    });
  };

  return (
    <div className="app-wrapper">
      <AppHeader
        isHealthy={isHealthy}
        healthData={healthData}
        vimMode={vim.mode}
        onToggleVimMode={() => (vim.mode === 'NORMAL' ? vim.enterInsertMode() : vim.enterNormalMode())}
        onOpenDocs={() => setIsDocsOpen(true)}
        userId={userId}
        onChangeUserId={(event) => {
          const nextUserId = event.target.value;
          setUserId(nextUserId);
          persistUserId(nextUserId);
          if (nextUserId.trim()) setUserIdError(null);
        }}
        userIdError={userIdError}
        userIdInputRef={userIdInputRef}
        activePage={activePage}
        onSelectPage={setActivePage}
      />

      <main className="app-container adhoc-app">
        <div className="workspace-panel" hidden={activePage !== 'query'}>
          <SearchWorkspace
            isActive={activePage === 'query'}
            userId={userId}
            topK={topK}
            setTopK={setTopK}
            onFrameClick={handleQueryFrameClick}
            onQueryChange={setActiveQuery}
            queryInputRef={queryInputRef}
            onFocusQueryInput={() => vim.setMode('INSERT')}
            onBlurQueryInput={() => vim.setMode('NORMAL')}
            onFocusUserId={handleFocusUserId}
            onHistoryRefresh={() => setHistoryRefreshToken((token) => token + 1)}
            replayRequest={replayRequest}
          />
        </div>
        <div className="workspace-panel" hidden={activePage !== 'image-search'}>
          <ImageSearchWorkspace
            isActive={activePage === 'image-search'}
            topK={topK}
            setTopK={setTopK}
            onFrameClick={handleQueryFrameClick}
          />
        </div>
        <div className="workspace-panel" hidden={activePage !== 'filter'}>
          <FilterWorkspace isActive={activePage === 'filter'} onFrameClick={handleFilterFrameClick} />
        </div>
        <div className="workspace-panel" hidden={activePage !== 'workspace'}>
          <WorkspacePage
            isActive={activePage === 'workspace'}
            userId={userId}
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
          onSubmit={selectedFrame.submissionMode === 'kis' ? handleInspectorSubmit : undefined}
          onClose={() => setSelectedFrame(null)}
        />
      )}
      <TopKPromptModal
        isOpen={vim.isTopKOpen && (activePage === 'query' || activePage === 'image-search')}
        currentTopK={topK}
        onSave={setTopK}
        onClose={() => vim.setIsTopKOpen(false)}
      />
      <VimHelpModal isOpen={vim.isHelpOpen} onClose={() => vim.setIsHelpOpen(false)} />
      <ApiDocsModal isOpen={isDocsOpen} onClose={() => setIsDocsOpen(false)} />
    </div>
  );
};

const App = () => (
  <SubmissionProvider>
    <SubmissionDialogProvider>
      <AppContent />
    </SubmissionDialogProvider>
  </SubmissionProvider>
);

export default App;
