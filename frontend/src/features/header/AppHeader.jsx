import React from 'react';
import HealthBadge from '../health/components/HealthBadge';
import VbsUserControl from '../vbs/components/VbsUserControl';

export const AppHeader = ({
  isHealthy,
  healthData,
  onOpenDocs,
  userIdInputRef,
  isAvsTask = false,
  onToggleWorkspace,
}) => (
  <header className="app-header">
    <div className="app-title-group">
      <h1 className="app-title">VBS 2027 Video Retrieval</h1>
      <div className="workspace-mode-toggle" role="group" aria-label="Workspace Mode">
        <button
          type="button"
          className={`workspace-mode-btn ${!isAvsTask ? 'active' : ''}`}
          onClick={() => onToggleWorkspace?.('KIS')}
          title="Switch to KIS (Ctrl+Shift+1)"
        >
          KIS
        </button>
        <button
          type="button"
          className={`workspace-mode-btn ${isAvsTask ? 'active' : ''}`}
          onClick={() => onToggleWorkspace?.('AVS')}
          title="Switch to AVS (Ctrl+Shift+2)"
        >
          AVS
        </button>
      </div>
      <HealthBadge isHealthy={isHealthy} healthData={healthData} />
      <button
        type="button"
        className="api-docs-badge-btn"
        onClick={onOpenDocs}
        title="Interactive API Docs / FastAPI Specs"
      >
        <span className="api-docs-icon" />
        <span>API Docs</span>
      </button>
    </div>
    <div className="app-header-tools">
      <VbsUserControl inputRef={userIdInputRef} />
    </div>
  </header>
);

export default AppHeader;
