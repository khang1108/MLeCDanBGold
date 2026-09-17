import React from 'react';
import HealthBadge from '../health/components/HealthBadge';
import VbsUserControl from '../vbs/components/VbsUserControl';

export const AppHeader = ({
  isHealthy,
  healthData,
  onOpenDocs,
  userIdInputRef,
  activePage,
  onSelectPage,
}) => (
  <header className="app-header">
    <div className="app-title-group">
      <h1 className="app-title">VBS 2027 Video Retrieval</h1>
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
      <nav className="workspace-nav" aria-label="Workspace selection">
        {[
          ['query', 'Query'],
          ['workspace', 'Workspace'],
        ].map(([page, label]) => (
          <button
            key={page}
            type="button"
            className={`workspace-nav-btn ${activePage === page ? 'active' : ''}`}
            onClick={() => onSelectPage(page)}
            aria-pressed={activePage === page}
          >
            {label}
          </button>
        ))}
      </nav>
    </div>
  </header>
);

export default AppHeader;
