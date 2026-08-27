import React from 'react';

const HealthBadge = ({ isHealthy, healthData }) => {
  if (isHealthy === null) {
    return (
      <div className="health-badge checking" title="Checking backend health status...">
        <span className="health-dot pulsing" />
        <span className="health-text">Checking...</span>
      </div>
    );
  }

  if (isHealthy) {
    const totalFrames = healthData?.total_frames ?? 0;
    const tooltip = `Status: OK | System: Ready | Frames: ${totalFrames.toLocaleString()}`;

    return (
      <div className="health-badge online" title={tooltip}>
        <span className="health-dot online-dot" />
        <span className="health-text">System Ready</span>
      </div>
    );
  }

  return (
    <div className="health-badge offline" title="Backend is unavailable or not ready for search">
      <span className="health-dot offline-dot" />
      <span className="health-text">System Not Ready</span>
    </div>
  );
};

export default HealthBadge;
