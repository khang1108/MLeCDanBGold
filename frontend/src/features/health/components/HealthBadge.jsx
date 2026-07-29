import React from 'react';

const HealthBadge = ({ isHealthy, healthData, isChecking }) => {
  if (isHealthy === null) {
    return (
      <div className="health-badge checking" title="Checking backend health status...">
        <span className="health-dot pulsing" />
        <span className="health-text">Checking...</span>
      </div>
    );
  }

  if (isHealthy) {
    const readyText = healthData?.ready ? 'Ready' : 'Online';
    const totalFrames = healthData?.total_frames ?? 0;
    const tooltip = `Status: OK | System: ${readyText} | Frames: ${totalFrames.toLocaleString()}`;

    return (
      <div className="health-badge online" title={tooltip}>
        <span className="health-dot online-dot" />
        <span className="health-text">System Online</span>
      </div>
    );
  }

  return (
    <div className="health-badge offline" title="Backend service is offline or unreachable">
      <span className="health-dot offline-dot" />
      <span className="health-text">System Offline</span>
    </div>
  );
};

export default HealthBadge;
