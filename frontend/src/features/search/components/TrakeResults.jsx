/** Render live or replayed TRAKE paths with the canonical Query layout.

This component owns only the result-section presentation. Search orchestration
and replay state stay in SearchWorkspace and ReplayResults respectively.
*/
import React from 'react';
import TrakePathCard from './TrakePathCard';

export const TrakeResults = ({
  events = [],
  paths = [],
  warnings = [],
  error = null,
  hasSearched = false,
  onFrameClick,
  onTrakeSubmit,
  getFrameClassName,
}) => (
  <section className="frames-container" aria-label="TRAKE ordered paths">
    {error && (
      <div className="error-alert" role="alert">
        <div className="error-details">
          <h4 className="error-title">TRAKE Search Error</h4>
          <p className="error-message">{error}</p>
        </div>
      </div>
    )}
    {!error && warnings.length > 0 && (
      <div className="search-warning" role="status">
        <span>Server note:</span>
        <ul>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
      </div>
    )}
    {!error && paths.length > 0 && (
      <div className="frames-scroll-region">
        {paths.map((path, index) => (
          <TrakePathCard
            key={`${path.video_id}-${index}`}
            events={events}
            path={path}
            onFrameClick={onFrameClick}
            onSubmit={onTrakeSubmit}
            getFrameClassName={getFrameClassName}
          />
        ))}
      </div>
    )}
    {!error && hasSearched && paths.length === 0 && (
      <div className="frames-empty-state">
        <p className="body-md frames-empty-text">No ordered TRAKE paths found</p>
      </div>
    )}
  </section>
);

export default TrakeResults;
