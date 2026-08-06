import React, { useEffect, useRef, useState } from "react";
import ImageModal from "./features/frames/components/ImageModal";
import AdHocSearchWorkspace from "./features/search/components/AdHocSearchWorkspace";
import VqaSearchWorkspace from "./features/vqa/components/VqaSearchWorkspace";
import { useHealthCheck } from "./features/health/hooks/useHealthCheck";
import HealthBadge from "./features/health/components/HealthBadge";
import { useVimMode } from "./features/vim/hooks/useVimMode";
import VimModeBadge from "./features/vim/components/VimModeBadge";
import TopKPromptModal from "./features/vim/components/TopKPromptModal";
import VimHelpModal from "./features/vim/components/VimHelpModal";
import "./styles/gif-loader.css";
import "./styles/vim.css";

export const taskFromPathname = (pathname) => pathname === "/qa" ? "qa" : "kis";

function App() {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [topK, setTopK] = useState(20);
  const [pathname, setPathname] = useState(() => window.location.pathname);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData, isChecking } = useHealthCheck();
  const vim = useVimMode({
    onCloseAllModals: () => setSelectedFrame(null),
    queryInputRef,
  });
  const task = taskFromPathname(pathname);

  useEffect(() => {
    const handlePopState = () => setPathname(window.location.pathname);
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigate = (event, targetPath) => {
    event.preventDefault();
    if (window.location.pathname !== targetPath) {
      window.history.pushState({}, "", targetPath);
      setPathname(targetPath);
      setSelectedFrame(null);
    }
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
          <VimModeBadge
            mode={vim.mode}
            onToggleMode={() =>
              vim.mode === "NORMAL"
                ? vim.enterInsertMode()
                : vim.enterNormalMode()
            }
          />
        </div>
        <nav className="task-navigation" aria-label="Task selection">
          <a
            className={`task-navigation-link${task === "kis" ? " active" : ""}`}
            href="/"
            aria-current={task === "kis" ? "page" : undefined}
            onClick={(event) => navigate(event, "/")}
          >
            KIS
          </a>
          <a
            className={`task-navigation-link${task === "qa" ? " active" : ""}`}
            href="/qa"
            aria-current={task === "qa" ? "page" : undefined}
            onClick={(event) => navigate(event, "/qa")}
          >
            QA
          </a>
        </nav>
      </header>

      <main className="app-container adhoc-app">
        {task === "qa" ? (
          <VqaSearchWorkspace topK={topK} setTopK={setTopK} />
        ) : (
          <AdHocSearchWorkspace
            topK={topK}
            setTopK={setTopK}
            onFrameClick={setSelectedFrame}
            queryInputRef={queryInputRef}
            onFocusQueryInput={() => vim.setMode("INSERT")}
            onBlurQueryInput={() => vim.setMode("NORMAL")}
          />
        )}
      </main>

      {selectedFrame && (
        <ImageModal
          frame={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}
      <TopKPromptModal
        isOpen={vim.isTopKOpen}
        currentTopK={topK}
        onSave={setTopK}
        onClose={() => vim.setIsTopKOpen(false)}
      />
      <VimHelpModal
        isOpen={vim.isHelpOpen}
        onClose={() => vim.setIsHelpOpen(false)}
      />
    </div>
  );
}

export default App;
