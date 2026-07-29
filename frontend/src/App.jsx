import React, { useCallback, useState } from "react";
import ConversationPanel from "./features/conversation/components/ConversationPanel";
import { useRequestGuard } from "./features/conversation/hooks/useRequestGuard";
import { useConversationSession } from "./features/conversation/hooks/useConversationSession";
import { useFeedbackDraft } from "./features/conversation/hooks/useFeedbackDraft";
import { useSessionHistory } from "./features/conversation/hooks/useSessionHistory";
import { useSearchWorkspace } from "./features/conversation/hooks/useSearchWorkspace";
import FramesBox from "./features/frames/components/FramesBox";
import ImageModal from "./features/frames/components/ImageModal";
import TabNavigation from "./features/navigation/components/TabNavigation";
import AdHocSearchWorkspace from "./features/search/components/AdHocSearchWorkspace";
import OptionsDrawer from "./features/search-controls/components/OptionsDrawer";
import { useHealthCheck } from "./features/health/hooks/useHealthCheck";
import HealthBadge from "./features/health/components/HealthBadge";
import DeleteSessionModal from "./features/conversation/components/DeleteSessionModal";
import { deleteSession } from "./api/sessions";
import "./styles/gif-loader.css";

// App composes features; endpoint and state details live in their owning hooks.
function App() {
  const [activeTab, setActiveTab] = useState("conversation");
  const [query, setQuery] = useState("");
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [isOptionsOpen, setIsOptionsOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState(null);
  const [isDeletingSession, setIsDeletingSession] = useState(false);
  const [topK, setTopK] = useState(20);
  const [searchMode, setSearchMode] = useState("accurate");
  const { isHealthy, healthData, isChecking } = useHealthCheck();
  const { isPending, runRequest } = useRequestGuard();
  const { session, sessionError, create, load, setSession } =
    useConversationSession(runRequest);
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
    results,
    warnings,
    latencyMs,
    lastRequestId,
    error,
    isSearching,
    clear: clearResults,
    submit: submitSearch,
  } = search;

  const resetWorkspace = useCallback(() => {
    clearResults();
    setSelectedFrame(null);
    setQuery("");
  }, [clearResults]);

  const createSession = useCallback(async () => {
    const next = await create();
    if (next) {
      resetWorkspace();
      setIsHistoryOpen(false);
    }
  }, [create, resetWorkspace]);

  const loadSession = useCallback(
    async (sessionId) => {
      const next = await load(sessionId);
      if (next) {
        resetWorkspace();
        setIsHistoryOpen(false);
      }
    },
    [load, resetWorkspace],
  );

  const handleDeleteRequest = useCallback((targetId) => {
    setDeleteTargetId(targetId);
  }, []);

  const handleConfirmDelete = useCallback(
    async (targetId) => {
      setIsDeletingSession(true);
      try {
        await deleteSession(targetId);
        history.removeSessionId(targetId);
        if (session?.session_id === targetId) {
          setSession(null);
          resetWorkspace();
        }
        setDeleteTargetId(null);
      } catch (err) {
        // preserve modal for retry if deletion fails
      } finally {
        setIsDeletingSession(false);
      }
    },
    [history, resetWorkspace, session?.session_id, setSession],
  );

  const submit = useCallback(
    async (value) => {
      let activeSession = session;
      if (!activeSession) {
        activeSession = await create();
      }
      if (activeSession && (await submitSearch(value, activeSession))) {
        setQuery("");
      }
    },
    [create, session, submitSearch],
  );

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
    onDeleteHistory: handleDeleteRequest,
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
      <header className="app-header">
        <div className="app-title-group">
          <h1 className="app-title">HCMAI 2026 Frame Retrieval</h1>
          <HealthBadge
            isHealthy={isHealthy}
            healthData={healthData}
            isChecking={isChecking}
          />
        </div>
        <TabNavigation activeTab={activeTab} onSelectTab={setActiveTab} />
      </header>

      {activeTab === "conversation" ? (
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
              canSubmit={Boolean(query.trim()) || feedback.feedbackDirty}
            />
            <section className="results-workspace">
              <FramesBox
                results={results}
                isLoading={isSearching}
                error={error}
                latencyMs={latencyMs}
                warnings={warnings}
                feedbackState={feedback.stateFor}
                onPromising={(id) => feedback.toggle(id, "promising")}
                onReject={(id) => feedback.toggle(id, "rejected")}
                onFrameClick={setSelectedFrame}
              />
            </section>
          </div>
        </main>
      ) : (
        <main className="app-container adhoc-app">
          <AdHocSearchWorkspace
            topK={topK}
            setTopK={setTopK}
            searchMode={searchMode}
            setSearchMode={setSearchMode}
            onFrameClick={setSelectedFrame}
          />
        </main>
      )}

      <OptionsDrawer
        isOpen={isOptionsOpen}
        onClose={() => setIsOptionsOpen(false)}
        topK={topK}
        setTopK={setTopK}
        searchMode={searchMode}
        setSearchMode={setSearchMode}
        onReset={() => {
          setTopK(20);
          setSearchMode("accurate");
        }}
      />
      {selectedFrame && (
        <ImageModal
          frame={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}
      <DeleteSessionModal
        sessionId={deleteTargetId}
        onConfirm={handleConfirmDelete}
        onClose={() => setDeleteTargetId(null)}
        isDeleting={isDeletingSession}
      />
    </div>
  );
}

export default App;
