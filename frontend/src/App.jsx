import React, { useCallback, useEffect, useRef, useState } from 'react';
import ConversationPanel from './features/conversation/components/ConversationPanel';
import { useRequestGuard } from './features/conversation/hooks/useRequestGuard';
import { useConversationSession } from './features/conversation/hooks/useConversationSession';
import { useFeedbackDraft } from './features/conversation/hooks/useFeedbackDraft';
import { useSessionHistory } from './features/conversation/hooks/useSessionHistory';
import { useSearchWorkspace } from './features/conversation/hooks/useSearchWorkspace';
import FramesBox from './features/frames/components/FramesBox';
import ImageModal from './features/frames/components/ImageModal';
import OptionsDrawer from './features/search-controls/components/OptionsDrawer';

// App composes features; endpoint and state details live in their owning hooks.
function App() {
  const [query, setQuery] = useState('');
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [isOptionsOpen, setIsOptionsOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [topK, setTopK] = useState(20);
  const [searchMode, setSearchMode] = useState('accurate');
  const initialRequestRef = useRef(false);
  const { isPending, runRequest } = useRequestGuard();
  const { session, sessionError, create, load, setSession } = useConversationSession(runRequest);
  const feedback = useFeedbackDraft(session, isPending);
  const history = useSessionHistory();
  const search = useSearchWorkspace({
    session,
    topK,
    searchMode,
    draftFeedback: feedback.draftFeedback,
    feedbackDirty: feedback.feedbackDirty,
    runRequest,
    setSession,
  });
  const {
    results, warnings, latencyMs, lastRequestId, error, isSearching,
    clear: clearResults, submit: submitSearch,
  } = search;

  const resetWorkspace = useCallback(() => {
    clearResults();
    setSelectedFrame(null);
    setQuery('');
  }, [clearResults]);

  const createSession = useCallback(async () => {
    const next = await create();
    if (next) {
      resetWorkspace();
      setIsHistoryOpen(false);
    }
  }, [create, resetWorkspace]);

  const loadSession = useCallback(async (sessionId) => {
    const next = await load(sessionId);
    if (next) {
      resetWorkspace();
      setIsHistoryOpen(false);
    }
  }, [load, resetWorkspace]);

  const submit = useCallback(async (value) => {
    if (await submitSearch(value)) setQuery('');
  }, [submitSearch]);

  useEffect(() => {
    if (!initialRequestRef.current) {
      initialRequestRef.current = true;
      createSession();
    }
  }, [createSession]);

  const toolbar = {
    history: { ...history, isOpen: isHistoryOpen },
    onNew: createSession,
    onOptions: () => setIsOptionsOpen(true),
    onToggleHistory: () => {
      if (isHistoryOpen) setIsHistoryOpen(false);
      else {
        setIsHistoryOpen(true);
        history.loadHistory();
      }
    },
    onSelectHistory: loadSession,
  };
  const debug = {
    requestId: lastRequestId,
    topK,
    searchMode,
    resultCount: results.length,
    committedFeedback: feedback.committedFeedback,
    draftFeedback: feedback.draftFeedback,
    feedbackDirty: feedback.feedbackDirty,
  };

  return (
    <div className="app-wrapper">
      <main className="app-container conversation-app">
        <div className="workspace-layout conversation-layout">
          <ConversationPanel
            toolbar={toolbar}
            session={session}
            sessionError={sessionError}
            isPending={isPending}
            debug={debug}
            query={query}
            setQuery={setQuery}
            onSubmit={submit}
            canSubmit={Boolean(session) && (Boolean(query.trim()) || feedback.feedbackDirty)}
          />
          <section className="results-workspace">
            <FramesBox
              results={results}
              isLoading={isSearching}
              error={error}
              latencyMs={latencyMs}
              warnings={warnings}
              feedbackState={feedback.stateFor}
              onPromising={(id) => feedback.toggle(id, 'promising')}
              onReject={(id) => feedback.toggle(id, 'rejected')}
              onFrameClick={setSelectedFrame}
            />
          </section>
        </div>
      </main>
      <OptionsDrawer
        isOpen={isOptionsOpen}
        onClose={() => setIsOptionsOpen(false)}
        topK={topK}
        setTopK={setTopK}
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        onReset={() => { setTopK(20); setSearchMode('accurate'); }}
      />
      {selectedFrame && <ImageModal frame={selectedFrame} onClose={() => setSelectedFrame(null)} />}
    </div>
  );
}

export default App;