import React from 'react';

/**
 * Present the committed canonical query and revision for the active KIS session.
 */
const IntentSummary = ({ currentIntent }) => {
  if (!currentIntent) return null;

  return (
    <div className="kis-intent-summary" data-testid="kis-intent-summary">
      <div className="kis-intent-summary-header">
        {typeof currentIntent.revision === 'number' && currentIntent.revision > 0 && (
          <span className="kis-revision-tag" title={`Revision ${currentIntent.revision}`}>
            Rev {currentIntent.revision}
          </span>
        )}
        {currentIntent.query_text && (
          <div className="kis-intent-canonical" data-testid="kis-intent-canonical">
            <span className="kis-intent-label">Resolved Query:</span>
            <span className="kis-intent-query-text">{currentIntent.query_text}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default IntentSummary;
