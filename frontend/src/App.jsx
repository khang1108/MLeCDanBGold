import React, { useRef, useState } from "react";
import ImageModal from "./features/frames/components/ImageModal";
import AdHocSearchWorkspace from "./features/search/components/AdHocSearchWorkspace";
import { useHealthCheck } from "./features/health/hooks/useHealthCheck";
import HealthBadge from "./features/health/components/HealthBadge";
import { useVimMode } from "./features/vim/hooks/useVimMode";
import VimModeBadge from "./features/vim/components/VimModeBadge";
import TopKPromptModal from "./features/vim/components/TopKPromptModal";
import VimHelpModal from "./features/vim/components/VimHelpModal";
import "./styles/gif-loader.css";
import "./styles/vim.css";

// App exposes only the standalone ad-hoc competition search workspace.
function App() {
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [topK, setTopK] = useState(20);
  const queryInputRef = useRef(null);
  const { isHealthy, healthData, isChecking } = useHealthCheck();
  const vim = useVimMode({
    onCloseAllModals: () => setSelectedFrame(null),
    queryInputRef,
  });

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
      </header>

      <main className="app-container adhoc-app">
        <AdHocSearchWorkspace
          topK={topK}
          setTopK={setTopK}
          onFrameClick={setSelectedFrame}
          queryInputRef={queryInputRef}
          onFocusQueryInput={() => vim.setMode("INSERT")}
          onBlurQueryInput={() => vim.setMode("NORMAL")}
        />
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
