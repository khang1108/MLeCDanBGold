import React from 'react';
import HealthBadge from '../health/components/HealthBadge';
import VimModeBadge from '../vim/components/VimModeBadge';

export const AppHeader = ({
  isHealthy,
  healthData,
  vimMode,
  onToggleVimMode,
  onOpenDocs,
  userId,
  onChangeUserId,
  userIdError,
  userIdInputRef,
  activePage,
  onSelectPage,
}) => (
  <header className="app-header">
    <div className="app-title-group">
      <h1 className="app-title">HCMAI 2026 Frame Retrieval</h1>
      <HealthBadge isHealthy={isHealthy} healthData={healthData} />
      <VimModeBadge mode={vimMode} onToggleMode={onToggleVimMode} />
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
      <label className="user-id-field" htmlFor="app-user-id">
        <span>User ID</span>
        <input
          ref={userIdInputRef}
          id="app-user-id"
          className="input-text"
          value={userId}
          onChange={onChangeUserId}
          placeholder="team-a"
          autoComplete="off"
          aria-describedby={userIdError ? 'app-user-id-error' : undefined}
        />
      </label>
      {userIdError && (
        <span id="app-user-id-error" className="user-id-error" role="alert">
          {userIdError}
        </span>
      )}
      <nav className="workspace-nav" aria-label="Workspace selection">
        {[
          ['query', 'Query'],
          ['image-search', 'Image Search'],
          ['filter', 'Filter'],
          ['workspace', 'Workspace'],
          ['database', 'Database'],
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
