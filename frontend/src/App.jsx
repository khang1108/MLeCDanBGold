import React, { useRef, useState } from "react";
import ImageModal from "./features/frames/components/ImageModal";
import SearchWorkspace from "./features/search/components/SearchWorkspace";
import FilterWorkspace from "./features/filter/components/FilterWorkspace";
import { useHealthCheck } from "./features/health/hooks/useHealthCheck";
import HealthBadge from "./features/health/components/HealthBadge";
import { useVimMode } from "./features/vim/hooks/useVimMode";
import VimModeBadge from "./features/vim/components/VimModeBadge";
import TopKPromptModal from "./features/vim/components/TopKPromptModal";
import VimHelpModal from "./features/vim/components/VimHelpModal";
import ApiDocsModal from "./features/docs/components/ApiDocsModal";
import { SubmissionProvider, useSubmission } from "./features/submission/contexts/SubmissionContext";

function AppContent() {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [activeQuery, setActiveQuery] = useState("");
  const [activePage, setActivePage] = useState("query");
  const [modalQuery, setModalQuery] = useState("");
  const [topK, setTopK] = useState(20);
  const [isDocsOpen, setIsDocsOpen] = useState(false);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData } = useHealthCheck();
  const { requestSubmission } = useSubmission();
  const vim = useVimMode({
    onCloseAllModals: () => setSelectedFrame(null),
    queryInputRef,
    enableTopK: activePage === "query",
  });

  const handleQueryFrameClick = (selection) => {
    setSelectedFrame(selection);
    setModalQuery(activeQuery);
  };

  const handleFilterFrameClick = (frame) => {
    setSelectedFrame({ frame, submissionMode: "kis" });
    setModalQuery("");
  };

  return (
    <div className="app-wrapper">
      <header className="app-header">
        <div className="app-title-group">
          <h1 className="app-title">HCMAI 2026 Frame Retrieval</h1>
          <HealthBadge
            isHealthy={isHealthy}
            healthData={healthData}
          />
          <VimModeBadge
            mode={vim.mode}
            onToggleMode={() =>
              vim.mode === "NORMAL"
                ? vim.enterInsertMode()
                : vim.enterNormalMode()
            }
          />
          <button
            type="button"
            className="api-docs-badge-btn"
            onClick={() => setIsDocsOpen(true)}
            title="Interactive API Docs / FastAPI Specs"
          >
            <span className="api-docs-icon"></span>
            <span>API Docs</span>
          </button>
        </div>
        <nav className="workspace-nav" aria-label="Workspace selection">
          <button
            type="button"
            className={`workspace-nav-btn ${activePage === "query" ? "active" : ""}`}
            onClick={() => setActivePage("query")}
            aria-pressed={activePage === "query"}
          >
            Query
          </button>
          <button
            type="button"
            className={`workspace-nav-btn ${activePage === "filter" ? "active" : ""}`}
            onClick={() => setActivePage("filter")}
            aria-pressed={activePage === "filter"}
          >
            Filter
          </button>
        </nav>
      </header>

      <main className="app-container adhoc-app">
        <div className="workspace-panel" hidden={activePage !== "query"}>
          <SearchWorkspace
            isActive={activePage === "query"}
            topK={topK}
            setTopK={setTopK}
            onFrameClick={handleQueryFrameClick}
            onQueryChange={setActiveQuery}
            queryInputRef={queryInputRef}
            onFocusQueryInput={() => vim.setMode("INSERT")}
            onBlurQueryInput={() => vim.setMode("NORMAL")}
          />
        </div>
        <div className="workspace-panel" hidden={activePage !== "filter"}>
          <FilterWorkspace
            isActive={activePage === "filter"}
            onFrameClick={handleFilterFrameClick}
          />
        </div>
      </main>

      {selectedFrame && (
        <ImageModal
          frame={selectedFrame.frame}
          query={modalQuery}
          onSubmit={selectedFrame.submissionMode === "kis" ? requestSubmission : undefined}
          onClose={() => setSelectedFrame(null)}
        />
      )}
      <TopKPromptModal
        isOpen={vim.isTopKOpen && activePage === "query"}
        currentTopK={topK}
        onSave={setTopK}
        onClose={() => vim.setIsTopKOpen(false)}
      />
      <VimHelpModal
        isOpen={vim.isHelpOpen}
        onClose={() => vim.setIsHelpOpen(false)}
      />
      <ApiDocsModal
        isOpen={isDocsOpen}
        onClose={() => setIsDocsOpen(false)}
      />
    </div>
  );
}

function App() {
  return (
    <SubmissionProvider>
      <AppContent />
    </SubmissionProvider>
  );
}

export default App;
