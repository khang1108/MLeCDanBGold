import React from 'react';
import HealthBadge from '../health/components/HealthBadge';
import VbsUserControl from '../vbs/components/VbsUserControl';

export const AppHeader = ({
  isHealthy,
  healthData,
  onOpenDocs,
  userIdInputRef,
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
    </div>
  </header>
);

export default AppHeader;
