import React from 'react';
import { useVbsSession } from '../contexts/VbsSessionContext';

/** Let the operator connect one mapped participant and explicitly unlock it. */
const VbsUserControl = ({ inputRef }) => {
  const {
    draftUserId,
    setDraftUserId,
    connectionState,
    error,
    dresLogStatus,
    connect,
    disconnect,
  } = useVbsSession();
  const isLocked = connectionState === 'connected';
  const isConnecting = connectionState === 'connecting';
  const dresStatus = isConnecting
    ? { label: 'Connecting', className: 'connecting' }
    : !isLocked
      ? { label: 'Disconnected', className: 'disconnected' }
      : dresLogStatus === 'sent'
        ? { label: 'Log sent', className: 'sent' }
        : dresLogStatus === 'failed'
          ? { label: 'Last log failed', className: 'failed' }
          : { label: 'Connected', className: 'connected' };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!isLocked && !isConnecting) connect();
  };

  const handleUnlock = () => {
    disconnect();
    window.setTimeout(() => inputRef?.current?.focus(), 0);
  };

  return (
    <form className="vbs-user-control" onSubmit={handleSubmit}>
      <label className="user-id-field" htmlFor="app-user-id">
        <span>User ID</span>
        <input
          ref={inputRef}
          id="app-user-id"
          className="input-text"
          value={draftUserId}
          onChange={(event) => setDraftUserId(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') handleSubmit(event);
          }}
          placeholder="team-a"
          autoComplete="off"
          disabled={isLocked || isConnecting}
          aria-describedby={error ? 'app-user-id-error' : undefined}
        />
      </label>
      {isLocked ? (
        <button
          type="button"
          className="vbs-user-status connected"
          onClick={handleUnlock}
          title="Disconnect and unlock participant"
        >
          OK
        </button>
      ) : (
        <button type="submit" className="vbs-user-status" disabled={isConnecting || !draftUserId.trim()}>
          {isConnecting ? 'Connecting…' : 'Connect'}
        </button>
      )}
      <span
        className={`vbs-dres-status ${dresStatus.className}`}
        role="status"
        aria-label="DRES status"
      >
        <span className="vbs-dres-status-dot" aria-hidden="true" />
        {dresStatus.label}
      </span>
      {error && <span id="app-user-id-error" className="user-id-error" role="alert">{error}</span>}
    </form>
  );
};

export default VbsUserControl;
